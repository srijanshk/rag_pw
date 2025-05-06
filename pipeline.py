import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import BartTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
from torch.cuda.amp import GradScaler
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logger_rag = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, dense_retriever, device,
                 train_generator=True,
                 train_retriever_end_to_end=False, 
                 total_steps_for_scheduler=10000,
                 k_retrieval_for_inference=10, 
                 retriever_temperature_for_training=1.0, 
                 ):
        self.dense_retriever = dense_retriever
        self.device = device
        self.train_generator = train_generator
        self.train_retriever_end_to_end = train_retriever_end_to_end
        self.total_steps_for_scheduler = total_steps_for_scheduler
        self.k_retrieval_for_inference = k_retrieval_for_inference

        self.retriever_temperature_for_training = retriever_temperature_for_training

        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

        if self.train_generator or self.train_retriever_end_to_end:
            self._init_optimizers_and_scheduler()
        
        if not self.train_generator:
            self.generator.eval()
        if not self.train_retriever_end_to_end:
            self.dense_retriever.query_encoder.eval()


        self.scaler = GradScaler()
    
    def _init_optimizers_and_scheduler(self):
        param_groups = []
        if self.train_generator:
            param_groups.append({"params": self.generator.parameters(), "lr": 5e-6, "name": "generator"})
            self.generator.train()
            logger_rag.info("Generator parameters included in RAG optimizer.")
        
        if self.train_retriever_end_to_end:
            self.dense_retriever.enable_fine_tuning_mode() 
            param_groups.append({
                "params": self.dense_retriever.query_encoder.parameters(), 
                "lr": 1e-5,
                "name": "retriever_query_encoder"
            })
            logger_rag.info("Retriever's query_encoder parameters included in RAG optimizer for end-to-end training.")

        if not param_groups:
            logger_rag.warning("RAGPipeline initiated for training, but neither generator nor retriever is set to be trained. Optimizer will be empty.")
            self.optimizer = None
            self.scheduler = None
            return

        self.optimizer = AdamW(param_groups)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_steps_for_scheduler),
            num_training_steps=self.total_steps_for_scheduler
        )
        logger_rag.info("RAG Optimizer and Scheduler initialized.")

    def _get_differentiable_retrieval_outputs_for_training(
        self, query_text: str, positive_doc_texts: List[str], negative_doc_texts: List[str]
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Embeds query and a given set of positive/negative documents, calculates differentiable
        scores, and returns document texts and their probabilities p(z|x).
        This is used during RAG training to enable backpropagation to the retriever.
        """
        if not positive_doc_texts:
            logger_rag.warning(f"No positive documents for query '{query_text}' in training. Returning empty retrieval.")
            return [], torch.empty(0, device=self.device)

        # 1. Embed Query (Trainable)
        # embed_texts_for_training handles train mode and prefixes
        query_embedding = self.dense_retriever.embed_texts_for_training(
            [query_text], "query", self.dense_retriever.query_encoder, is_encoder_trainable=True
        ) # Shape: [1, D]

        # 2. Prepare and Embed Candidate Documents (Doc encoder is frozen)
        candidate_doc_texts = positive_doc_texts + negative_doc_texts
        doc_embeddings = self.dense_retriever.embed_texts_for_training(
            candidate_doc_texts, "passage", self.dense_retriever.doc_encoder, is_encoder_trainable=False # doc_encoder is frozen
        ) # Shape: [num_candidate_docs, D]

        # 3. Normalize Embeddings (for cosine similarity)
        query_embedding_norm = F.normalize(query_embedding, p=2, dim=1)
        doc_embeddings_norm = F.normalize(doc_embeddings, p=2, dim=1)

        # 4. Calculate Differentiable Scores (logits for softmax)
        # retriever_scores shape: [num_candidate_docs]
        retriever_scores = torch.matmul(query_embedding_norm, doc_embeddings_norm.T).squeeze(0) 

        # 5. Calculate Probabilities p(z|x) via Softmax
        # doc_probs_p_z_given_x shape: [num_candidate_docs]
        doc_probs_p_z_given_x = F.softmax(retriever_scores / self.retriever_temperature_for_training, dim=0)
        
        return candidate_doc_texts, doc_probs_p_z_given_x


    def _calculate_rag_nll_loss(self, query_text: str, answer_text: str, 
                                conditioning_doc_texts: List[str], 
                                doc_probs_p_z_given_x: torch.Tensor):
        """
        Calculates the Negative Log-Likelihood for RAG based on pre-computed doc_probs_p_z_given_x.
        doc_probs_p_z_given_x: Differentiable tensor of probabilities for conditioning_doc_texts.
        """
        if not conditioning_doc_texts or doc_probs_p_z_given_x.numel() == 0:
            logger_rag.warning(f"No conditioning documents or probabilities for query '{query_text}'. Returning high loss.")
            return torch.tensor(10.0, device=self.device, dtype=torch.float32, requires_grad=True if self.train_generator else False)

        prompts = [f"question: {query_text} context: {doc_text}" for doc_text in conditioning_doc_texts]
        
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        
        targets = self.tokenizer(
            [answer_text] * len(prompts), return_tensors="pt", padding=True, truncation=True, max_length=128
        ).input_ids.to(self.device)

        labels = targets.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100 # For CrossEntropyLoss

        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            outputs = self.generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())
            
            num_actual_tokens = (shift_labels != -100).sum(dim=1).clamp(min=1) # Avoid div by zero
            per_sample_loss = per_token_loss.sum(dim=1) / num_actual_tokens

        log_likelihoods_y_given_xz = -per_sample_loss  # log p(y|x,z)
        log_pz_given_x = torch.log(doc_probs_p_z_given_x.clamp(min=1e-9)) # log p(z|x)

        # Ensure log_likelihoods and log_pz have compatible shapes for broadcasting if needed,
        # though here they should both be [num_conditioning_docs]
        marginal_log_likelihood = torch.logsumexp(log_likelihoods_y_given_xz + log_pz_given_x, dim=0)
        nll_loss = -marginal_log_likelihood
        return nll_loss


    def train_on_batch(self, batch_data: List[Dict], batch_idx=None, epoch=None, stage_name=None):
        if not self.optimizer:
            logger_rag.warning("train_on_batch called, but RAG optimizer is not initialized (nothing to train). Skipping.")
            return None

        # Set modes for components being trained
        if self.train_generator: self.generator.train()
        if self.train_retriever_end_to_end: self.dense_retriever.query_encoder.train() # Already handled by enable_fine_tuning_mode

        self.optimizer.zero_grad()
        total_loss_for_batch_items = 0.0
        valid_items_count = 0

        for item in batch_data:
            query = item["query"]
            answer = item["answers"][0] # Assuming one answer
            positive_docs = item.get("positive_docs", []) # Must be provided for retriever training
            negative_docs = item.get("negative_docs", []) # Optional, but good for retriever

            if self.train_retriever_end_to_end and not positive_docs:
                logger_rag.warning(f"Skipping item for query '{query}' due to missing positive_docs required for retriever E2E training.")
                continue

            # Differentiable retrieval for training
            # This path is used if train_retriever_end_to_end is True.
            # If train_retriever_end_to_end is False, we'd need a different way to get docs (e.g. Faiss)
            # but then p(z|x) wouldn't be differentiable.
            # For simplicity, this example assumes if we are in train_on_batch, and train_retriever_end_to_end is True,
            # we use the differentiable path. If train_retriever_end_to_end is False, this setup won't train retriever.
            
            # The critical part: get differentiable p(z|x)
            # For RAG, we marginalize over a set of documents 'z'.
            # This set should include the positive document(s) and some negatives.
            # If train_retriever_end_to_end is False, this part doesn't contribute to retriever grads.
            # However, to calculate RAG loss, we still need some p(z|x).
            if not positive_docs:
                 logger_rag.warning(f"Skipping item for query '{query}' due to missing positive_docs required for RAG loss calculation.")
                 continue

            candidate_doc_texts, doc_probs_p_z_given_x = self._get_differentiable_retrieval_outputs_for_training(
                query, positive_docs, negative_docs
            )
            
            if not candidate_doc_texts or doc_probs_p_z_given_x.numel() == 0:
                logger_rag.warning(f"No valid documents/probabilities retrieved for query '{query}' during training. Skipping item.")
                continue

            loss_for_item = self._calculate_rag_nll_loss(query, answer, candidate_doc_texts, doc_probs_p_z_given_x)

            if torch.isnan(loss_for_item) or torch.isinf(loss_for_item):
                logger_rag.warning(f"Skipping NaN/Inf loss for query: {query}")
                continue
            
            # Here, we'll average at the end. Backward pass is on per-item scaled loss for numerical stability with AMP.
            self.scaler.scale(loss_for_item / len(batch_data)).backward() # Average loss for backward
            total_loss_for_batch_items += loss_for_item.item()
            valid_items_count += 1

        if valid_items_count == 0:
            logger_rag.warning(f"No valid items in batch {batch_idx}, skipping optimizer step.")
            return None

        self.scaler.unscale_(self.optimizer) # Unscale before clipping
        
        if self.train_generator:
            clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
        if self.train_retriever_end_to_end:
            clip_grad_norm_(self.dense_retriever.query_encoder.parameters(), max_norm=1.0)
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler:
            self.scheduler.step()
        
        avg_loss = total_loss_for_batch_items / valid_items_count
        # Log avg_loss with wandb or other logger
        if stage_name:
            wandb.log({"stage": stage_name, "avg_loss": avg_loss}, step=batch_idx)
        return avg_loss


    def generate_answer(self, query: str, generation_strategy="thorough", num_beams_generation=5, max_length_generation=64):
        self.generator.eval()
        self.dense_retriever.query_encoder.eval() # Use Faiss search for generation

        # _retrieve_topk_docs_for_inference uses Faiss, returns list of {"text":.., "score":..}
        retrieved_docs_inference = self.dense_retriever.search(query, self.k_retrieval_for_inference)

        if not retrieved_docs_inference:
            logger_rag.warning(f"No docs for '{query}' in generation. Using query only.")
            prompt_text = f"{query}"
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        elif generation_strategy == "fast":
            top_doc_text = retrieved_docs_inference[0]["text"]
            prompt_text = f"{query} {top_doc_text}"
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        else: # thorough
            prompt_texts = [f"{query} {doc_info['text']}" for doc_info in retrieved_docs_inference]
            inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        with torch.no_grad():
            generated_ids = self.generator.generate(
                inputs.input_ids, attention_mask=inputs.attention_mask,
                max_length=max_length_generation, num_beams=num_beams_generation,
                early_stopping=True, no_repeat_ngram_size=3,
            )
        decoded_answers = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

        if generation_strategy == "fast" or not retrieved_docs_inference:
            return decoded_answers[0]
        else: # thorough - simple re-ranking based on original Faiss scores (could be more complex)
            candidate_answer_scores = {}
            # Using original Faiss scores for re-ranking multiple generated answers
            for i, answer_text in enumerate(decoded_answers):
                # This assumes one generated answer per retrieved doc if inputs had multiple prompts
                original_retriever_score = retrieved_docs_inference[i]["score"] if i < len(retrieved_docs_inference) else 0
                candidate_answer_scores[answer_text] = candidate_answer_scores.get(answer_text, 0.0) + original_retriever_score
            
            if not candidate_answer_scores: return "Could not generate a confident answer."
            return max(candidate_answer_scores.items(), key=lambda x: x[1])[0]


    def train_pipeline(self, dataset, batch_size=8, epochs=3, stage_name="RAG_E2E_Training"):
        if not self.train_generator and not self.train_retriever_end_to_end:
            logger_rag.info("Neither generator nor retriever is set for training. Skipping train_pipeline.")
            return []

        all_epoch_losses = []
        for epoch_num in range(epochs):
            logger_rag.info(f"🔁 Starting RAG Training Epoch {epoch_num + 1}/{epochs} for Stage: {stage_name}")
            if isinstance(dataset, list): np.random.shuffle(dataset)
            
            num_batches_in_epoch = (len(dataset) + batch_size - 1) // batch_size
            epoch_losses_this_epoch = []

            with tqdm(range(num_batches_in_epoch), desc=f"Epoch {epoch_num+1}", leave=True, dynamic_ncols=True) as pbar:
                for batch_i in pbar:
                    start_idx, end_idx = batch_i * batch_size, (batch_i + 1) * batch_size
                    current_batch_data = dataset[start_idx:end_idx]
                    if not current_batch_data: continue

                    loss = self.train_on_batch(current_batch_data, batch_idx=batch_i + 1, epoch=epoch_num + 1, stage_name=stage_name)
                    if loss is not None:
                        epoch_losses_this_epoch.append(loss)
                        pbar.set_postfix(avg_loss_batch=f"{loss:.4f}")
            
            if epoch_losses_this_epoch:
                mean_loss_this_epoch = np.mean(epoch_losses_this_epoch)
                all_epoch_losses.append(mean_loss_this_epoch)
                wandb.log({"epoch": epoch_num + 1, "mean_loss": mean_loss_this_epoch}, step=epoch_num + 1)
                logger_rag.info(f"✅ Epoch {epoch_num + 1} Mean Loss: {mean_loss_this_epoch:.4f}")
            else:
                 logger_rag.warning(f"Epoch {epoch_num + 1} had no valid batches.")
        logger_rag.info(f"RAG training completed for stage: {stage_name}.")
        return all_epoch_losses
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import BartTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup
import numpy as np
from typing import List
from tqdm import tqdm
import logging
from torch.cuda.amp import GradScaler
import wandb

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

logger_rag = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, dense_retriever, device, generator_fine_tune=False, total_steps=10000, k=10, inference_table_train=None, inference_table_infer=None):
        self.dense_retriever = dense_retriever
        self.device = device

        self.generator_fine_tune = generator_fine_tune
        self.k = k
        self.total_steps = total_steps
        self.inference_table_train = inference_table_train
        self.inference_table_infer = inference_table_infer

        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        self.generator = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)


        if self.generator_fine_tune:
            self._init_generator_optimizer() # Renamed
            self.generator.train() # Set generator to train mode
        else:
            self.generator.eval() # Set generator to eval mode

        self.scaler = GradScaler()
    
    def _init_generator_optimizer(self):
        logger_rag.info("Optimizer: Initializing for Generator parameters only for RAG training.")
        param_groups = [
            {"params": self.generator.parameters(), "lr": 5e-6}
        ]
        self.optimizer = AdamW(param_groups)

        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_steps_for_generator_scheduler),
            num_training_steps=self.total_steps_for_generator_scheduler
        )
        logger_rag.info(f"Optimizer and Scheduler initialized for RAGPipeline's generator.")

    
    # def _init_optimizer(self):
    #     self.generator.train()

    #     param_groups = [
    #         {"params": self.generator.parameters(), "lr": 5e-6}
    #     ]

    #     if self.dense_retriever.fine_tune:
    #         print("Optimizer: Including Retriever parameters for End-to-End training.")
    #         self.dense_retriever.query_encoder.train()
    #         param_groups.append({"params": self.dense_retriever.query_encoder.parameters(), "lr": 1e-5})
    #     else:
    #         print("Optimizer: Including Generator parameters only.")

    #     self.optimizer = AdamW(param_groups)

    #     self.scheduler = get_cosine_schedule_with_warmup(
    #         self.optimizer,
    #         num_warmup_steps=int(0.1 * self.total_steps),
    #         num_training_steps=self.total_steps
    #     )

    def _retrieve_topk_docs(self, query: str, k: int) -> List[dict]:
        """
        Retrieves top-k documents using the retriever's standard search.
        NOTE: This uses the numpy-based Faiss search from DenseRetriever.search(), 
              which breaks gradients back to the retriever. 
              The retriever is effectively frozen during RAG training.
        """
        # Ensure retriever's query encoder is in eval mode for this step
        self.dense_retriever.query_encoder.eval()
        results_from_retriever = self.dense_retriever.search(query, k) # List[{"id":_, "text":_, "score":_}]

        # The dense_retriever.search already returns a list of dicts with 'text' and 'score'
        # We need to normalize scores to be probabilities for the RAG loss
        processed_results_for_rag = []
        raw_scores = []

        for res_item in results_from_retriever:
            text = res_item.get("text", "")
            score = res_item.get("score", 0.0) # Faiss scores (e.g. inner product)
            if text: # Ensure text is not empty
                processed_results_for_rag.append({"text": text, "original_score": score})
                raw_scores.append(score)
        
        if not processed_results_for_rag:
            return []

        # Convert scores to probabilities (e.g., using softmax if scores are logits, or simple normalization)
        # Assuming higher score is better (like cosine similarity/inner product on normalized vectors)
        # If scores can be negative or are distances, transformation might be needed before softmax/normalization.
        # For simplicity, let's use softmax on the raw scores.
        # If scores are already well-behaved (e.g. positive and sum to a reasonable number), direct normalization is fine.
        
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)
        # Using softmax to convert scores to probabilities
        # Add a temperature parameter if needed for score sharpness.
        probabilities = F.softmax(scores_tensor / 1.0, dim=0) # Temp=1.0

        final_retrieved_with_probs = []
        for i, item in enumerate(processed_results_for_rag):
            final_retrieved_with_probs.append({
                "text": item["text"], 
                "p_z_given_x": probabilities[i].item(), # p(z|x)
                "original_score": item["original_score"]
            })
            
        return final_retrieved_with_probs

    def _log_marginal_likelihood(self, query: str, answer: str):
        retrieved_docs_with_probs  = self._retrieve_topk_docs(query, self.k)
        if wandb.run is not None and self.inference_table_train is not None:
            retrieved_texts = "<br><br>".join([doc for doc, _ in retrieved_docs_with_probs][:10])
            self.inference_table_train.add_data(query, answer, wandb.Html(retrieved_texts))


        # if not retrieved:
        #     logger.warning(f"No documents retrieved for query: '{query}'. Skipping loss calculation, returning high loss.")
        #     return torch.tensor(10.0, device=self.device, dtype=torch.float32, requires_grad=True)

        # # Prepare prompts at once
        # prompts = [f"{query} {doc}" for doc, _ in retrieved]
        # p_z_scores = [p_z for _, p_z in retrieved]

        # # Tokenize all prompts together
        # inputs = self.tokenizer(
        #     prompts,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=512
        # ).to(self.device)

        # # Tokenize targets once
        # targets = self.tokenizer(
        #     [answer] * len(prompts),
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=True,
        #     max_length=128
        # ).input_ids.to(self.device)

        # # Prepare labels (ignore pad tokens)
        # labels = targets.clone()
        # labels[labels == self.tokenizer.pad_token_id] = -100

        # # Forward pass: batch
        # with torch.cuda.amp.autocast():
        #     outputs = self.generator(
        #         input_ids=inputs.input_ids,
        #         attention_mask=inputs.attention_mask,
        #         labels=labels
        #     )
        #     # outputs.loss is average over batch — not directly useful
        #     # We need per-example loss: get logits and manually compute loss

        #     logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
        #     shift_logits = logits[..., :-1, :].contiguous()
        #     shift_labels = labels[..., 1:].contiguous()

        #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        #     per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        #     per_token_loss = per_token_loss.view(shift_labels.size())

        #     # Now per_token_loss: [batch_size, seq_len]
        #     per_sample_loss = per_token_loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)

        # # per_sample_loss: shape [batch_size]
        # log_likelihoods = -per_sample_loss  # negate because we need log p(y|x,z)

        # p_z = torch.tensor(p_z_scores, device=self.device, dtype=torch.float32)
        # p_z = torch.clamp(p_z, min=1e-8)
        # p_z_normalized = p_z / p_z.sum()
        # log_pz = torch.log(p_z_normalized)

        # marginal_log_likelihood = torch.logsumexp(log_likelihoods + log_pz, dim=0)

        # nll = -marginal_log_likelihood

        # return nll
        if not retrieved_docs_with_probs:
            logger_rag.warning(f"No documents retrieved for query: '{query}'. Returning high loss.")
            # Return a tensor that requires grad, otherwise .backward() might complain if this is the only loss.
            return torch.tensor(10.0, device=self.device, dtype=torch.float32, requires_grad=self.generator.training)


        prompts = [f"{query} {doc_info['text']}" for doc_info in retrieved_docs_with_probs]
        p_z_values = [doc_info['p_z_given_x'] for doc_info in retrieved_docs_with_probs]

        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        
        targets = self.tokenizer(
            [answer] * len(prompts), return_tensors="pt", padding=True, truncation=True, max_length=128
        ).input_ids.to(self.device)

        labels = targets.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Generator forward pass
        # Context manager for autocast should be based on whether scaler is enabled
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
            
            # Average loss per sample (sequence)
            num_actual_tokens = (shift_labels != -100).sum(dim=1)
            # Avoid division by zero if a sequence has no valid labels (e.g. all padding after shifting)
            num_actual_tokens = torch.max(num_actual_tokens, torch.ones_like(num_actual_tokens))
            per_sample_loss = per_token_loss.sum(dim=1) / num_actual_tokens


        log_likelihoods_y_given_xz = -per_sample_loss  # log p(y|x, z) = -Loss(y | x, z)
        log_pz_given_x = torch.log(torch.tensor(p_z_values, device=self.device, dtype=torch.float32).clamp(min=1e-9)) # log p(z|x)

        marginal_log_likelihood = torch.logsumexp(log_likelihoods_y_given_xz + log_pz_given_x, dim=0)
        nll_loss = -marginal_log_likelihood
        return nll_loss

    # def train_on_batch(self, batch, batch_idx=None, epoch=None, stage=None):
    #     if not self.optimizer:
    #         raise RuntimeError("Optimizer not initialized. Ensure fine_tune=True during RAGPipeline init.")
        

    #     self.generator.train()
    #     if self.dense_retriever.fine_tune and any(p is rp for p in self.optimizer.param_groups[0]['params'] for rp in self.dense_retriever.query_encoder.parameters()):
    #          self.dense_retriever.query_encoder.train()
        
    #     total_loss = 0.0
    #     valid_items = 0

    #     self.optimizer.zero_grad()

    #     for item in batch:
    #         query = item["query"]
    #         answer = item["answers"][0]

    #         loss = self._log_marginal_likelihood(query, answer)

    #         if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
    #             logger.warning(f"[Epoch {epoch}] Batch {batch_idx} → Skipping NaN loss for query: {query}")
    #             continue

    #         self.scaler.scale(loss / len(batch)).backward()
    #         total_loss += loss.item()
    #         valid_items += 1

    #     if valid_items == 0:
    #         logger.warning(f"[Epoch {epoch}] Batch {batch_idx} → No valid samples, skipping optimizer step.")
    #         return None

    #     self.scaler.unscale_(self.optimizer)
    #     clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

    #     is_retriever_in_optimizer = False
    #     retriever_params = list(self.dense_retriever.query_encoder.parameters())
    #     if retriever_params:
    #         for group in self.optimizer.param_groups:
    #             if any(p is rp for p in group['params'] for rp in retriever_params):
    #                  is_retriever_in_optimizer = True
    #                  break
    #         if is_retriever_in_optimizer and self.dense_retriever.fine_tune:
    #             clip_grad_norm_(self.dense_retriever.query_encoder.parameters(), max_norm=1.0)

    #     self.scaler.step(self.optimizer)
    #     self.scaler.update()
    #     self.scheduler.step()
    #     self.optimizer.zero_grad()

    #     avg_loss = total_loss / valid_items

    #     wandb.log({"RAG_Loss": avg_loss, "Batch": batch_idx, "stage": stage})

    #     return avg_loss
    def train_on_batch(self, batch_data, batch_idx=None, epoch=None, stage_name=None): # renamed params
        if not self.generator_fine_tune or not hasattr(self, "optimizer"):
            raise RuntimeError("Generator optimizer not initialized. Ensure generator_fine_tune=True during RAGPipeline init.")
        
        self.generator.train() # Ensure generator is in training mode
        # Retriever is used in eval mode via its .search() method, which handles its internal state.

        self.optimizer.zero_grad()
        
        accumulated_loss_for_batch = 0.0
        valid_items_in_batch = 0

        for item in batch_data:
            query = item["query"]
            answer = item["answers"][0] # Assuming answers is a list and we take the first one

            loss_for_item = self._log_marginal_likelihood(query, answer)

            if torch.isnan(loss_for_item) or torch.isinf(loss_for_item) or loss_for_item.item() == 0.0 and not loss_for_item.requires_grad : # check for problematic loss
                logger_rag.warning(f"[Epoch {epoch}] Batch {batch_idx} → Skipping problematic loss (NaN, Inf, or zero with no grad) for query: {query}")
                continue
            
            # Accumulate loss, scale for gradient accumulation if batch_data is small or for averaging
            # Here, we average the loss over the number of items in the batch for the backward pass.
            # This means each item contributes equally to the gradient update.
            self.scaler.scale(loss_for_item / len(batch_data)).backward()
            accumulated_loss_for_batch += loss_for_item.item()
            valid_items_in_batch += 1

        if valid_items_in_batch == 0:
            logger_rag.warning(f"[Epoch {epoch}] Batch {batch_idx} → No valid samples in batch, skipping optimizer step.")
            return None # Or 0.0, depending on how you want to log

        self.scaler.unscale_(self.optimizer) # Unscale before clipping
        clip_grad_norm_(self.generator.parameters(), max_norm=1.0) # Clip gradients for the generator
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if hasattr(self, "scheduler"):
            self.scheduler.step()
        
        # self.optimizer.zero_grad() # Already done at the beginning of the next call or at start of this one.

        avg_loss_for_batch = accumulated_loss_for_batch / valid_items_in_batch
        
        # Wandb logging
        log_data = {"RAG_Generator_Loss": avg_loss_for_batch}
        if batch_idx is not None: log_data["Batch_ID"] = batch_idx
        if stage_name is not None: log_data["Stage"] = stage_name
        if wandb.run is not None: wandb.log(log_data)

        return avg_loss_for_batch


    # def generate_answer(self, query: str, strategy="thorough", top_k=None):
    #     top_k = top_k or self.k
    #     retrieved = self._retrieve_topk_docs(query, top_k)

    #     if strategy == "fast":
    #         doc, _ = retrieved[0]
    #         prompt = f"{query} {doc}"
    #         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
    #         outputs = self.generator.generate(
    #             **inputs,
    #             max_length=64,
    #             num_beams=5,
    #             early_stopping=True,
    #             no_repeat_ngram_size=3,
    #             forced_bos_token_id=0
    #         )

    #         return self.tokenizer.decode(output[0], skip_special_tokens=True)

    #     elif strategy == "thorough":
    #         prompts = [f"{query} {doc}" for doc, _ in retrieved]
    #         p_z = [score for _, score in retrieved]

    #         inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
    #         with torch.no_grad():
    #             outputs = self.generator.generate(
    #                 **inputs,
    #                 max_length=64,
    #                 num_beams=5,
    #                 early_stopping=True,
    #                 no_repeat_ngram_size=3,
    #                 forced_bos_token_id=0
    #             )
    #         answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

    #         candidate_scores = {}
    #         for answer, score in zip(answers, p_z):
    #             candidate_scores[answer] = candidate_scores.get(answer, 0.0) + score
            
    #         if wandb.run is not None and self.inference_table_infer is not None:
    #             final_answer, final_score = max(candidate_scores.items(), key=lambda x: x[1])

    #             top_docs = retrieved[:5]
    #             html_rows = []
    #             for i, (doc, score) in enumerate(top_docs):
    #                 html_rows.append(f"<b>[{i+1}]</b> <code>Score: {score:.4f}</code><br>{doc}")

    #             html_display = "<hr>".join(html_rows)
    #             input_prompt = f"{query} {top_docs[0][0]}" if top_docs else "[empty]"

    #             self.inference_table_infer.add_data(
    #                 query,
    #                 final_answer,
    #                 wandb.Html(html_display),
    #                 input_prompt
    #             )


    #         return max(candidate_scores.items(), key=lambda x: x[1])[0]

    def generate_answer(self, query: str, generation_strategy="thorough", num_beams_generation=5, max_length_generation=64): # Renamed params
        self.generator.eval() # Ensure generator is in eval mode for generation
        self.dense_retriever.query_encoder.eval() # Ensure retriever is also in eval

        # retrieved_docs_with_probs: List[{"text": str, "p_z_given_x": float, "original_score": float}]
        retrieved_docs_with_probs = self._retrieve_topk_docs(query, self.k_retrieval)

        if not retrieved_docs_with_probs:
            logger_rag.warning(f"No documents retrieved for query: '{query}' during generation. Generating based on query alone or returning empty.")
            # Fallback: generate from query alone or return a default message
            prompt_text = f"question: {query} context: " # Empty context
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        elif generation_strategy == "fast":
            # Use only the top document for "fast" strategy
            top_doc_text = retrieved_docs_with_probs[0]["text"]
            prompt_text = f"question: {query} context: {top_doc_text}"
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        elif generation_strategy == "thorough":
            # Prepare inputs for all retrieved documents to generate multiple candidate answers
            prompt_texts = [f"question: {query} context: {doc_info['text']}" for doc_info in retrieved_docs_with_probs]
            inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        else:
            raise ValueError(f"Unknown generation_strategy: {generation_strategy}")

        with torch.no_grad(): # Ensure no gradients are computed during generation
            generated_ids = self.generator.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length_generation,
                num_beams=num_beams_generation,
                early_stopping=True,
                no_repeat_ngram_size=3,
                # forced_bos_token_id=0 # BART doesn't typically need this unless issues with BOS
            )
        
        decoded_answers = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

        if generation_strategy == "fast" or not retrieved_docs_with_probs:
            return decoded_answers[0] # Return the single generated answer
        
        elif generation_strategy == "thorough":
            # Re-rank or select among generated answers based on p_z_given_x
            # Simple re-ranking: sum p_z_given_x for identical answers, then pick highest.
            candidate_answer_scores = {}
            doc_probabilities_p_z = [doc_info['p_z_given_x'] for doc_info in retrieved_docs_with_probs]

            for i, answer_text in enumerate(decoded_answers):
                # Ensure index i is valid for doc_probabilities_p_z, happens if len(decoded_answers) <= len(doc_probabilities_p_z)
                # This should be the case as we generate one answer per prompt (derived from a retrieved doc)
                current_doc_prob = doc_probabilities_p_z[i] if i < len(doc_probabilities_p_z) else 0
                candidate_answer_scores[answer_text] = candidate_answer_scores.get(answer_text, 0.0) + current_doc_prob
            
            if not candidate_answer_scores: return "Could not generate a confident answer." # Fallback
            
            final_selected_answer = max(candidate_answer_scores.items(), key=lambda x: x[1])[0]

            # Wandb logging for inference
            if wandb.run is not None and self.inference_table_infer is not None:
                top_docs_display_html = "<hr>".join([f"<b>Score (p(z|x)): {doc_info['p_z_given_x']:.4f}</b> (Orig: {doc_info['original_score']:.2f})<br>{doc_info['text'][:200]}..." for doc_info in retrieved_docs_with_probs[:3]])
                self.inference_table_infer.add_data(query, final_selected_answer, wandb.Html(top_docs_display_html), inputs['input_ids'].shape[1] if 'input_ids' in inputs else 'N/A')
            
            return final_selected_answer
        

    # def train(self, dataset, batch_size=8, epochs=3, stage ="Stage=2"):
    #     all_losses = []
    #     for epoch in range(epochs):
    #         logger.info(f"🔁 Starting Epoch {epoch + 1}/{epochs}")
    #         np.random.shuffle(dataset)

    #         num_batches = len(dataset) // batch_size
    #         batch_indices = range(0, len(dataset), batch_size)

    #         # tqdm for batches
    #         with tqdm(total=num_batches, desc=f"Epoch {epoch+1}", leave=True, dynamic_ncols=True) as pbar:
    #             for i in batch_indices:
    #                 batch = dataset[i: i + batch_size]
    #                 if len(batch) < batch_size:
    #                     continue
    #                 batch_idx = i // batch_size + 1
    #                 loss = self.train_on_batch(batch, batch_idx=batch_idx, epoch=epoch + 1, stage=stage)

    #                 pbar.set_postfix(loss=f"{loss:.4f}")
    #                 pbar.update(1)

    #                 all_losses.append(loss)

    #         epoch_loss = np.mean(all_losses[-num_batches:])
    #         wandb.log({"epoch_loss": epoch_loss, "epoch": {epoch + 1},"stage": stage})
    #         logger.info(f"✅ Finished Epoch {epoch + 1} — Mean Loss: {epoch_loss:.4f}")
        
    #     return all_losses

    def train_generator(self, dataset, batch_size=8, epochs=3, stage_name="GeneratorTraining"): # Renamed method and params
        if not self.generator_fine_tune:
            logger_rag.info("Generator fine-tuning is disabled for this RAGPipeline instance. Skipping training.")
            return []

        all_epoch_losses = []
        for epoch_num in range(epochs):
            logger_rag.info(f"🔁 Starting Generator Training Epoch {epoch_num + 1}/{epochs} for Stage: {stage_name}")
            
            # Simple shuffle for list-based dataset
            if isinstance(dataset, list):
                np.random.shuffle(dataset) 
            # If using a Hugging Face Dataset, shuffling is often handled by DataLoader or .shuffle() method

            num_batches_in_epoch = (len(dataset) + batch_size - 1) // batch_size # Handles partial last batch
            
            epoch_losses_this_epoch = []

            with tqdm(range(num_batches_in_epoch), desc=f"Epoch {epoch_num+1}", leave=True, dynamic_ncols=True) as pbar:
                for batch_i in pbar:
                    start_idx = batch_i * batch_size
                    end_idx = start_idx + batch_size
                    current_batch_data = dataset[start_idx:end_idx]

                    if not current_batch_data: continue # Skip if batch is empty

                    loss = self.train_on_batch(current_batch_data, batch_idx=batch_i + 1, epoch=epoch_num + 1, stage_name=stage_name)

                    if loss is not None:
                        epoch_losses_this_epoch.append(loss)
                        pbar.set_postfix(avg_loss_batch=f"{loss:.4f}")
            
            if epoch_losses_this_epoch:
                mean_loss_this_epoch = np.mean(epoch_losses_this_epoch)
                all_epoch_losses.append(mean_loss_this_epoch)
                logger_rag.info(f"✅ Finished Epoch {epoch_num + 1} — Mean RAG Generator Loss: {mean_loss_this_epoch:.4f}")
                # Wandb logging for epoch
                if wandb.run is not None: wandb.log({"epoch_generator_loss": mean_loss_this_epoch, "epoch_num": epoch_num + 1, "stage_name": stage_name})
            else:
                 logger_rag.warning(f"Epoch {epoch_num + 1} had no valid batches. No average loss to report.")


        logger_rag.info(f"Generator training completed for stage: {stage_name}.")
        return all_epoch_losses



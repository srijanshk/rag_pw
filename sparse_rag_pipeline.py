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

logger_rag_sparse = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SparseRAGPipeline:
    def __init__(self,
                 sparse_retriever,
                 device,
                 generator_model_name="facebook/bart-base",
                 k_retrieval_for_training_marginalization=10,
                 k_retrieval_for_inference=50,          
                 generator_lr=3e-5,
                 total_steps_for_scheduler=10000,
                 max_context_length=512,
                 max_answer_length=128,
                 use_xapian_scores_for_loss_weights=True,
                 loss_weighting_temperature=0.1
                ):

        self.sparse_retriever = sparse_retriever
        self.device = device
        # N for training: number of documents to pass through generator and marginalize over
        self.k_marginalization = k_retrieval_for_training_marginalization
        # K for inference: number of documents to retrieve and generate candidate answers from
        self.k_inference_candidates = k_retrieval_for_inference

        self.generator_model_name = generator_model_name
        self.max_context_length = max_context_length
        self.max_answer_length = max_answer_length
        self.use_xapian_scores_for_loss_weights = use_xapian_scores_for_loss_weights
        self.loss_weighting_temperature = loss_weighting_temperature
        self.total_steps_for_scheduler = total_steps_for_scheduler


        logger_rag_sparse.info(f"Initializing SparseRAGPipeline with generator: {self.generator_model_name}")
        logger_rag_sparse.info(f"Training will marginalize over {self.k_marginalization} retrieved documents.")
        logger_rag_sparse.info(f"Inference will generate candidates from {self.k_inference_candidates} retrieved documents.")


        self.tokenizer = BartTokenizer.from_pretrained(self.generator_model_name)
        self.generator = BartForConditionalGeneration.from_pretrained(self.generator_model_name).to(device)
        self.generator.train()

        self.optimizer = AdamW(self.generator.parameters(), lr=generator_lr)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps_for_scheduler),
            num_training_steps=total_steps_for_scheduler
        )
        self.scaler = GradScaler()
        logger_rag_sparse.info("Optimizer and Scheduler initialized for the generator.")

    def _prepare_single_generator_input(self, query: str, doc_text: str, doc_title: str = "") -> str:
        """ Prepares a single input string for the generator from one document. """
        title_prefix = f"title: {doc_title} " if doc_title else ""
        return f"question: {query} context: {title_prefix}text: {doc_text}"
    
    def train_on_batch(self, batch_of_pre_retrieved_data: List[Dict], batch_idx=None, epoch=None, stage_name=None):
        self.generator.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        all_generator_input_texts_for_batch = []

        all_target_answer_texts_for_generator_batch = []
        all_xapian_scores_for_batch_contexts = []
        sample_context_counts = []

        for original_sample_idx, item in enumerate(batch_of_pre_retrieved_data):
            query = item["query"]
            target_answer_text = item["answers"][0] if item.get("answers") else "" 

            if not target_answer_text.strip():
                continue

            pre_retrieved_contexts = item.get("retrieved_contexts", [])
            valid_contexts_for_this_item = []
            
            

            for ctx in pre_retrieved_contexts:
                doc_text = ctx.get("text", "").strip()
                if doc_text:
                    score = ctx.get("score", 1.0)
                    valid_contexts_for_this_item.append((ctx, score))
                if len(valid_contexts_for_this_item) >= self.k_marginalization:
                    break

            
            if not valid_contexts_for_this_item:
                continue

            num_contexts_added_for_this_sample = 0
            for doc_meta, xapian_score in valid_contexts_for_this_item:
                doc_text = doc_meta.get('text', '')
                doc_title = doc_meta.get('title', '')
                
                generator_input_str = self._prepare_single_generator_input(query, doc_text, doc_title)
                all_generator_input_texts_for_batch.append(generator_input_str)

                all_target_answer_texts_for_generator_batch.append(target_answer_text)
                all_xapian_scores_for_batch_contexts.append(xapian_score)
                num_contexts_added_for_this_sample += 1
            
            if num_contexts_added_for_this_sample > 0:
                sample_context_counts.append((original_sample_idx, num_contexts_added_for_this_sample))

        if not all_generator_input_texts_for_batch:
            logger_rag_sparse.debug(f"Batch {batch_idx} (Epoch {epoch}): No processable generator inputs created.")
            return None 

        # Tokenize all generator inputs together
        batched_generator_inputs = self.tokenizer(
            all_generator_input_texts_for_batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_context_length
        ).to(self.device)

        tokenized_targets = self.tokenizer(
            all_target_answer_texts_for_generator_batch,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_answer_length
        ).to(self.device)
        
        batched_labels = tokenized_targets.input_ids.clone()
        batched_labels[batched_labels == self.tokenizer.pad_token_id] = -100
        
        if (batched_labels == -100).all():
             logger_rag_sparse.warning(f"Batch {batch_idx} (Epoch {epoch}): All labels became -100 after tokenizing all answers. Skipping batch.")
             return None


        with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
            outputs = self.generator(
                input_ids=batched_generator_inputs.input_ids,
                attention_mask=batched_generator_inputs.attention_mask,
                labels=batched_labels
            )
            
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()

            shift_labels = batched_labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='none')

            per_token_neg_log_likelihoods = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1) # Flatten labels
            )

            per_token_neg_log_likelihoods = per_token_neg_log_likelihoods.view(logits.size(0), -1)
            

            actual_token_mask = (shift_labels != -100) # Shape: (TOTAL_CONTEXTS_IN_BATCH, ANS_SEQ_LEN-1)

            num_actual_tokens = actual_token_mask.sum(dim=1).clamp(min=1)
            neg_log_likelihood_sequences = (per_token_neg_log_likelihoods * actual_token_mask).sum(dim=1) / num_actual_tokens
            # Log average loss per token to WandB
            if wandb.run and stage_name:
                avg_token_loss = neg_log_likelihood_sequences.mean().item()
                wandb.log({f"{stage_name}_avg_token_loss": avg_token_loss}, step=wandb.run.step)
            log_probs_y_given_xz_all = -neg_log_likelihood_sequences 

            current_pos_in_flat_outputs = 0
            marginalized_nll_list_for_batch = []

            for original_sample_idx_in_batch_data, num_contexts_for_this_sample in sample_context_counts:
                start_idx = current_pos_in_flat_outputs
                end_idx = current_pos_in_flat_outputs + num_contexts_for_this_sample

                current_log_probs_gen = log_probs_y_given_xz_all[start_idx:end_idx]
                current_xapian_s = torch.tensor(
                    all_xapian_scores_for_batch_contexts[start_idx:end_idx],
                    device=self.device, dtype=torch.float32
                )

                if self.use_xapian_scores_for_loss_weights:
                    doc_weights_p_z = F.softmax(current_xapian_s / self.loss_weighting_temperature, dim=0)
                else: # Uniform weighting if not using Xapian scores
                    doc_weights_p_z = torch.full_like(current_log_probs_gen, 1.0 / num_contexts_for_this_sample, device=self.device)
                
                log_doc_weights = torch.log(doc_weights_p_z.detach().clamp(min=1e-9)) # Ensure no grad and numerical stability
                
                marginal_log_likelihood_for_sample = torch.logsumexp(log_doc_weights + current_log_probs_gen, dim=0)
                single_item_nll = -marginal_log_likelihood_for_sample
                
                if not (torch.isnan(single_item_nll) or torch.isinf(single_item_nll)):
                    marginalized_nll_list_for_batch.append(single_item_nll)
                else:
                    logger_rag_sparse.warning(f"NaN/Inf NLL for a sample in Batch {batch_idx} (Epoch {epoch}). Query: {batch_of_pre_retrieved_data[original_sample_idx_in_batch_data]['query'][:30]}...")

                current_pos_in_flat_outputs = end_idx
            
            if not marginalized_nll_list_for_batch:
                logger_rag_sparse.warning(f"Batch {batch_idx} (Epoch {epoch}) resulted in no valid NLLs after marginalization. Skipping optimizer step.")
                return None # No loss to backward/step
            
            final_batch_loss = torch.stack(marginalized_nll_list_for_batch).mean() # Mean of NLLs of original samples

        # Scale, backward, step for the single aggregated batch loss
        if self.scaler.is_enabled():
            self.scaler.scale(final_batch_loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else: # CPU or AMP not enabled
            final_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.optimizer.step()

        if self.scheduler:
            self.scheduler.step()

        loss_item = final_batch_loss.item() # Get scalar value for logging
        # WandB logging (ensure wandb.run is active)
        if wandb.run and stage_name:
            wandb.log({
                f"{stage_name}_batch_loss_OptA_TrueBatch": loss_item,
                # "batch_idx_in_epoch": batch_idx, # This is passed from train_pipeline
                # "current_epoch": epoch,         # This is passed from train_pipeline
                "learning_rate_gen": self.scheduler.get_last_lr()[0] if self.scheduler else self.optimizer.param_groups[0]['lr']
            }, step=wandb.run.step) # Use global step for batch losses if preferred
        return loss_item



    def generate_answer(self,
                        query: str,
                        num_beams_generation=4,
                        strategy=None, # strategy might not be very relevant here
                        top_k=None,    # For controlling how many docs are retrieved if live
                        pre_retrieved_docs_with_scores: List[Tuple[dict, float]] = None):
        """
        Generates an answer for a given query.
        If pre_retrieved_docs_with_scores is provided, it uses them.
        Otherwise, performs a live search using self.sparse_retriever.
        """
        self.generator.eval() # Set generator to evaluation mode

        actual_retrieved_docs_with_scores = []

        if pre_retrieved_docs_with_scores is not None:
            logger_rag_sparse.debug(f"Using {len(pre_retrieved_docs_with_scores)} pre-retrieved docs for query: {query[:30]}...")
            # If pre_retrieved_docs are provided, respect top_k if it further limits them
            num_docs_to_consider = top_k if top_k is not None else len(pre_retrieved_docs_with_scores)
            actual_retrieved_docs_with_scores = pre_retrieved_docs_with_scores[:num_docs_to_consider]
        else:
            # Perform live retrieval
            num_docs_to_retrieve_live = top_k if top_k is not None else self.k_inference_candidates
            logger_rag_sparse.debug(f"Performing live Xapian search for {num_docs_to_retrieve_live} docs for query: {query[:30]}...")
            actual_retrieved_docs_with_scores = self.sparse_retriever.search(query, k=num_docs_to_retrieve_live)

        if not actual_retrieved_docs_with_scores:
            # Fallback: generate from query only if no docs are retrieved/provided
            logger_rag_sparse.warning(f"No docs available (live or pre-retrieved) for '{query}' in inference. Using query only.")
            inputs = self.tokenizer(f"question: {query} context: ", return_tensors="pt",
                                    truncation=True, padding=True, max_length=self.max_context_length).to(self.device)
            with torch.no_grad():
                 generated_ids = self.generator.generate(
                    inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_length=self.max_answer_length or 128, num_beams=num_beams_generation,
                    early_stopping=True, no_repeat_ngram_size=3
                )
            answer_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            return answer_text if answer_text else "Could not generate an answer."

        candidate_answers_data = []
        for ctx in actual_retrieved_docs_with_scores:
            doc_metadata = ctx
            xapian_score = ctx.get("score", 1.0)
            doc_text = doc_metadata.get('text', '')
            doc_title = doc_metadata.get('title', '')
            if not doc_text.strip():
                continue

            generator_input_text = self._prepare_single_generator_input(query, doc_text, doc_title)
            inputs = self.tokenizer(
                generator_input_text, return_tensors="pt",
                padding=True, truncation=True, max_length=self.max_context_length
            ).to(self.device)

            with torch.no_grad():
                generated_ids = self.generator.generate(
                    inputs.input_ids, attention_mask=inputs.attention_mask,
                    max_length=self.max_answer_length, num_beams=num_beams_generation,
                    early_stopping=True, no_repeat_ngram_size=3
                )
            answer_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
            if answer_text.strip(): # Only consider non-empty generations
                candidate_answers_data.append({"text": answer_text, "retriever_score": xapian_score, "doc_title": doc_title})

        if not candidate_answers_data:
            # This can happen if all retrieved docs were empty or generation failed for all
            logger_rag_sparse.warning(f"Generation failed for all {len(actual_retrieved_docs_with_scores)} retrieved/provided contexts for query '{query[:30]}...'")
            # Fallback to query-only generation as a last resort if desired, or return specific message
            # For now, returning a specific message:
            return "Could not generate a confident answer from the provided/retrieved documents."


        # Simple re-ranking: pick the answer generated from the context of the highest-scoring Xapian doc.
        best_candidate = sorted(candidate_answers_data, key=lambda x: x["retriever_score"], reverse=True)[0]
        return best_candidate["text"]




    def train_pipeline(self, dataset_pre_retrieved, batch_size=8, epochs=3, stage_name="SparseRAG_Training"):
        # Ensure scheduler is initialized or re-initialized if total_steps change
        num_batches_per_epoch = (len(dataset_pre_retrieved) + batch_size - 1) // batch_size
        current_total_steps = num_batches_per_epoch * epochs
        if self.scheduler is None or self.total_steps_for_scheduler != current_total_steps:
            self.total_steps_for_scheduler = current_total_steps
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * self.total_steps_for_scheduler),
                num_training_steps=self.total_steps_for_scheduler
            )
            logger_rag_sparse.info(f"Scheduler re-initialized with {self.total_steps_for_scheduler} total steps for stage: {stage_name}.")

        all_batch_losses_cumulative = []
        for epoch_num in range(epochs):
            logger_rag_sparse.info(f"🔁 Starting Epoch {epoch_num + 1}/{epochs} for Stage: {stage_name}")
            # Data is already loaded, just shuffle if it's a list
            if isinstance(dataset_pre_retrieved, list):
                np.random.shuffle(dataset_pre_retrieved)

            epoch_batch_losses = []
            # Use a plain range for batch indices, data slicing handles the actual batch
            batch_indices = range(0, len(dataset_pre_retrieved), batch_size)
            with tqdm(batch_indices, desc=f"Epoch {epoch_num+1}/{epochs} [{stage_name}]", leave=True, dynamic_ncols=True) as pbar:
                for i_batch_start_idx in pbar:
                    current_batch_data_slice = dataset_pre_retrieved[i_batch_start_idx : i_batch_start_idx + batch_size]
                    if not current_batch_data_slice:
                        continue
                    
                    # The batch_idx for logging should be 1-based for the current epoch
                    current_batch_idx_in_epoch = (i_batch_start_idx // batch_size) + 1

                    loss = self.train_on_batch(current_batch_data_slice,
                                               batch_idx=current_batch_idx_in_epoch,
                                               epoch=epoch_num + 1,
                                               stage_name=stage_name)
                    if loss is not None:
                        epoch_batch_losses.append(loss)
                        all_batch_losses_cumulative.append(loss) # For overall plotting
                        pbar.set_postfix(avg_loss_batch=f"{loss:.4f}")
                        # LR logging is now inside train_on_batch after scheduler.step()

            if epoch_batch_losses:
                mean_epoch_loss = np.mean(epoch_batch_losses)
                logger_rag_sparse.info(f"✅ Epoch {epoch_num + 1} Mean Batch Loss: {mean_epoch_loss:.4f}")
                if wandb.run:
                    wandb.log({f"{stage_name}_epoch_mean_loss": mean_epoch_loss,
                               "epoch_num_completed": epoch_num + 1}) # Use a clear epoch key
            else:
                 logger_rag_sparse.warning(f"Epoch {epoch_num + 1} had no valid batches with loss for {stage_name}.")
        logger_rag_sparse.info(f"Training completed for stage: {stage_name}.")
        return all_batch_losses_cumulative

    def save_model(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        generator_path = os.path.join(output_dir, "generator")
        tokenizer_path = os.path.join(output_dir, "generator")
        self.generator.save_pretrained(generator_path)
        self.tokenizer.save_pretrained(tokenizer_path)
        logger_rag_sparse.info(f"SparseRAGPipelineOptionA generator and tokenizer saved to {output_dir}")

    @classmethod
    def load_model(cls, model_dir, sparse_retriever_instance, device, **kwargs): # Allow kwargs for other init params
        generator_path = os.path.join(model_dir, "generator")
        if not os.path.exists(generator_path):
            raise FileNotFoundError(f"Generator or tokenizer not found in {model_dir}")
        
        # Pass through other necessary args from kwargs or set defaults
        pipeline = cls(
            sparse_retriever=sparse_retriever_instance,
            device=device,
            generator_model_name=generator_path,
            k_retrieval_for_training_marginalization=kwargs.get('k_retrieval_for_training_marginalization', 3),
            k_retrieval_for_inference=kwargs.get('k_retrieval_for_inference', 5),
            generator_lr=kwargs.get('generator_lr', 3e-5), # Not used if not training, but good to have
            total_steps_for_scheduler=kwargs.get('total_steps_for_scheduler',1), # Placeholder, re-init on train
            max_context_length=kwargs.get('max_context_length', 312),
            max_answer_length=kwargs.get('max_answer_length', 128),
            use_xapian_scores_for_loss_weights=kwargs.get('use_xapian_scores_for_loss_weights', True),
            loss_weighting_temperature=kwargs.get('loss_weighting_temperature',0.1)
        )
        logger_rag_sparse.info(f"SparseRAGPipelineOptionA loaded from {model_dir}")
        return pipeline
import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
import logging
from torch.amp import GradScaler
import math

import wandb
import contextlib

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
                 model_name="facebook/bart-base",
                 use_fp16: bool = True
                 ):
        self.dense_retriever = dense_retriever
        self.device = device
        self.train_generator = train_generator
        self.train_retriever_end_to_end = train_retriever_end_to_end
        self.total_steps_for_scheduler = total_steps_for_scheduler
        self.k_retrieval_for_inference = k_retrieval_for_inference
        self.passage_max_length = 512
        self.answer_max_length = 128
        self.model_name = model_name
        self.use_fp16 = use_fp16

        self.retriever_temperature_for_training = retriever_temperature_for_training

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.generator = BartForConditionalGeneration.from_pretrained(self.model_name).to(device)
        # Enable gradient checkpointing to reduce memory consumption
        if hasattr(self.generator, "gradient_checkpointing_enable"):
            self.generator.gradient_checkpointing_enable()
        # Set chunk size for generator loss computation
        self.gen_loss_chunk_size = 8

        if self.train_generator or self.train_retriever_end_to_end:
            self._init_optimizers_and_scheduler()
        
        if not self.train_generator:
            self.generator.eval()
        if not self.train_retriever_end_to_end:
            self.dense_retriever.query_encoder.eval()


        self.scaler = GradScaler(enabled=(self.use_fp16 and self.device.type == 'cuda'))
        logger_rag.info(f"RAGPipeline GradScaler enabled: {self.scaler.is_enabled()}")
    
    def _init_optimizers_and_scheduler(self):
        param_groups = []
        if self.train_generator:
            param_groups.append({"params": self.generator.parameters(), "lr": 2e-5, "name": "generator"})
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
            candidate_doc_texts, "passage", self.dense_retriever.doc_encoder, is_encoder_trainable=False
        ) # Shape: [num_candidate_docs, D]

        with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
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


    def _calculate_rag_nll_loss(
        self,
        doc_input_ids: torch.LongTensor,            # shape [n_docs, passage_len]
        doc_attention_mask: torch.LongTensor,       # shape [n_docs, passage_len]
        answer_text: str,
        doc_probs_p_z_given_x: torch.Tensor,        # shape [n_docs]
    ):
        """
        Calculates the RAG NLL loss: -logsumexp_z [ log p(y|x,z) + log p(z|x) ]
        """
        n_docs = doc_input_ids.size(0)
        if n_docs == 0 or doc_probs_p_z_given_x.numel() == 0 or doc_probs_p_z_given_x.dim() != 1 or doc_probs_p_z_given_x.shape[0] != n_docs:
            logger_rag.warning(
                f"Invalid inputs for RAG loss (tokenized docs). "
                f"n_docs={n_docs}, probs shape={doc_probs_p_z_given_x.shape}. Returning high loss."
            )
            dummy_loss = torch.tensor(10.0, device=self.device, dtype=torch.float32)
            if self.train_generator or self.train_retriever_end_to_end:
                dummy_loss.requires_grad_()
            return dummy_loss

        # prepare labels once
        labels = self.tokenizer(
            [answer_text] * n_docs,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.answer_max_length
        ).input_ids.to(self.device)
        token_sums_list = []
        encoder = self.generator.get_encoder()
        for i in range(0, n_docs, self.gen_loss_chunk_size):
            chunk_ids = doc_input_ids[i : i + self.gen_loss_chunk_size]
            chunk_mask = doc_attention_mask[i : i + self.gen_loss_chunk_size]
            enc_out = encoder(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
                return_dict=True
            )
            with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
                outputs = self.generator(
                    encoder_outputs=enc_out,
                    decoder_input_ids=labels[i : i + self.gen_loss_chunk_size],
                    decoder_attention_mask=(labels[i : i + self.gen_loss_chunk_size] != self.tokenizer.pad_token_id),
                    return_dict=True
                )
            seq_logits = outputs.logits  # [chunk, target_len, vocab]
            logprobs = F.log_softmax(seq_logits, dim=-1)
            ll = logprobs.gather(dim=-1, index=labels[i : i + self.gen_loss_chunk_size].unsqueeze(-1)).squeeze(-1)
            ll.masked_fill_(labels[i : i + self.gen_loss_chunk_size].eq(self.tokenizer.pad_token_id), 0.0)
            token_sums_list.append(ll.sum(dim=1))
            del outputs, seq_logits, logprobs, enc_out
            torch.cuda.empty_cache()
        token_sums = torch.cat(token_sums_list, dim=0)
        log_doc_probs = torch.log(doc_probs_p_z_given_x.clamp_min(1e-9))
        marginal_ll = torch.logsumexp(token_sums + log_doc_probs, dim=0)
        nll_loss = -marginal_ll
        if (self.train_generator and any(p.requires_grad for p in self.generator.parameters())) or \
           (self.train_retriever_end_to_end and any(p.requires_grad for p in self.dense_retriever.query_encoder.parameters())):
            if not nll_loss.requires_grad:
                nll_loss = nll_loss.clone().requires_grad_(True)
        return nll_loss


    def train_on_batch(self, batch_data: List[Dict], batch_idx=None, epoch=None, stage_name=None):
        """
        Trains the RAG pipeline on a batch of data using differentiable retrieval
        and the marginal NLL loss.
        Requires batch_data items to have 'query', 'answers' (list),
        'positive_doc' (str), and 'negative_docs' (List[str]).
        """
        if not self.optimizer:
            logger_rag.warning("train_on_batch called, but RAG optimizer is not initialized (nothing to train). Skipping.")
            return None

        # --- Ensure Correct Modes ---
        # Generator training mode is handled within _calculate_rag_nll_loss if needed
        # Retriever query encoder needs to be in train mode *if* it's being trained E2E
        original_retriever_mode = self.dense_retriever.query_encoder.training
        if self.train_retriever_end_to_end:
            self.dense_retriever.query_encoder.train()
        else:
            # If not training retriever E2E, keep it in eval mode for consistency
             self.dense_retriever.query_encoder.eval()


        self.optimizer.zero_grad()
        
        # Accumulate loss over the batch before scaling and backward pass
        # This is often more stable than calling backward on each item, especially with AMP
        accumulated_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        valid_items_count = 0

        for item in batch_data:
            query = item.get("query")
            answers = item.get("answers")
            positive_docs = item.get("positive_docs", []) 
            negative_docs = item.get("negative_docs", []) # Expecting a list of negative doc strings

            # --- Input Validation ---
            if not query or not answers or positive_docs is None: # Check positive_doc explicitly for None
                logger_rag.warning(f"Skipping item due to missing query, answer, or positive_doc. Query: {query is not None}, Answer: {answers is not None}, PosDoc: {positive_docs is not None}")
                continue
            answer = answers[0] # Assuming the first answer is the target

            # --- Retrieval depending on mode ---
            if not self.train_retriever_end_to_end:
                # Live FAISS retrieval path
                retrieved = self._retrieve_topk_docs(query, self.k_retrieval_for_inference)
                if not retrieved:
                    logger_rag.warning(f"No docs retrieved for query '{query[:50]}...', skipping.")
                    continue
                doc_input_ids      = torch.stack([d["input_ids"] for d in retrieved])
                doc_attention_mask = torch.stack([d["attention_mask"] for d in retrieved])
                doc_probs          = torch.stack([d["p_z_given_x"] for d in retrieved])
            else:
                # Differentiable retrieval path
                candidate_texts, doc_probs = self._get_differentiable_retrieval_outputs_for_training(
                    query, positive_docs, negative_docs
                )
                if not candidate_texts or doc_probs.numel() == 0:
                    logger_rag.warning(f"No valid differentiable docs for '{query[:50]}...', skipping.")
                    continue
                tok = self.tokenizer(
                    candidate_texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=self.passage_max_length
                ).to(self.device)
                doc_input_ids      = tok.input_ids
                doc_attention_mask = tok.attention_mask

            # Compute RAG loss
            loss_for_item = self._calculate_rag_nll_loss(
                doc_input_ids,
                doc_attention_mask,
                answer,
                doc_probs.to(self.device)
            )

            if torch.isnan(loss_for_item) or torch.isinf(loss_for_item):
                logger_rag.warning(f"Skipping NaN/Inf loss for query: {query[:50]}...")
                self.optimizer.zero_grad(set_to_none=True)
                continue # Skip this item

            accumulated_loss = accumulated_loss + loss_for_item
            valid_items_count += 1


        # --- Backward Pass and Optimizer Step (if any valid items) ---
        if valid_items_count > 0:
            average_loss = accumulated_loss / valid_items_count
            
            # Scale the average loss and perform backward pass
            self.scaler.scale(average_loss).backward()

            # Unscale gradients before clipping
            self.scaler.unscale_(self.optimizer)

            # Gradient Clipping
            if self.train_generator:
                clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            if self.train_retriever_end_to_end:
                clip_grad_norm_(self.dense_retriever.query_encoder.parameters(), max_norm=1.0)

            # Optimizer Step
            self.scaler.step(self.optimizer)

            # Update GradScaler
            self.scaler.update()

            # Scheduler Step (if applicable)
            if self.scheduler:
                self.scheduler.step()

            avg_loss_item = average_loss.item() # Get Python float for logging
        else:
            logger_rag.warning(f"No valid items processed in batch {batch_idx}. Skipping optimizer step.")
            avg_loss_item = None # Indicate no loss computed

        # --- Restore Retriever Mode ---
        self.dense_retriever.query_encoder.train(original_retriever_mode)


        # --- Logging ---
        if avg_loss_item is not None:
            log_data = {"stage": stage_name, "avg_loss": avg_loss_item, "batch_idx": batch_idx, "epoch": epoch, "scaler_scale": self.scaler.get_scale()}
            if self.optimizer and self.optimizer.param_groups:
                lr_logged = False
                for i, pg in enumerate(self.optimizer.param_groups):
                    name = pg.get('name', f'group_{i}')
                    log_data[f"lr_{name}"] = pg.get('lr', 0)
                    lr_logged = True
                if not lr_logged: # Fallback if names aren't set
                     log_data["lr_group0"] = self.optimizer.param_groups[0].get('lr',0)


            if stage_name and wandb.run:
                wandb.log(log_data)
            return avg_loss_item
        else:
            return None # No loss calculated for this batch


    def _retrieve_topk_docs(self, query: str, k: int) -> List[dict]:
        """
        Retrieves top-k documents using the retriever's standard search, returns tokenized inputs and probabilities.
        """
        self.dense_retriever.query_encoder.eval()
        results = self.dense_retriever.search(query, k)  # Each item has 'input_ids','attention_mask','score','text'
        if not results:
            return []
        # Convert raw FAISS scores to probabilities
        raw_scores = [float(r["score"]) for r in results]
        scores_tensor = torch.tensor(raw_scores, dtype=torch.float32, device=self.device)
        probs = F.softmax(scores_tensor, dim=0)
        final = []
        for r, p in zip(results, probs):
            final.append({
                "input_ids":       r["input_ids"].to(self.device),
                "attention_mask":  r["attention_mask"].to(self.device),
                "p_z_given_x":     p,
                "original_score":  float(r["score"]),
                "text":            r.get("text", "")
            })
        return final
    
    def generate_answer(self, query: str, strategy="thorough", num_beams_generation=5, max_length_generation=128, top_k=None):
        """
        Generates an answer for a single query using the RAG pipeline.

        Args:
            query (str): The input query.
            strategy (str): 'fast' (use top-1 doc) or 'thorough' (use top-k docs and re-rank).
            num_beams_generation (int): Number of beams for generation.
            max_length_generation (int): Maximum length of the generated answer.
            top_k (int, optional): Number of documents to retrieve. Defaults to self.k_retrieval_for_inference.

        Returns:
            str: The generated answer string.
        """
        # 1. Set Eval Modes
        self.generator.eval()
        self.dense_retriever.query_encoder.eval()

        # 2. Retrieve Documents (using non-differentiable Faiss search)
        k = top_k or self.k_retrieval_for_inference
        retrieved_docs_with_probs = self._retrieve_topk_docs(query, k)

        # 3. Prepare Generator Input
        if not retrieved_docs_with_probs:
            logger_rag.warning(f"No documents retrieved for query: '{query}' during generation. Generating based on query alone.")
            prompt_text = f"question: {query} context: "
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            n_docs = 1
            input_batch_size = 1 # Only one prompt
        elif strategy == "fast":
            top_doc_text = retrieved_docs_with_probs[0]["text"]
            prompt_text = f"question: {query} context: {top_doc_text}"
            inputs = self.tokenizer(prompt_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(self.device)
            n_docs = 1
            input_batch_size = 1
        elif strategy == "thorough":
            prompt_texts = [f"question: {query} context: {doc_info['text']}" for doc_info in retrieved_docs_with_probs]
            inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            n_docs = len(retrieved_docs_with_probs)
            input_batch_size = n_docs # Batch size for generator is number of docs
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 4. Generate Answers - MODIFIED to get scores for re-ranking
        autocast_enabled_inference = getattr(self, 'use_fp16', False) and self.device.type == 'cuda'
        gen_output = None
        sequences = None
        score_logits = None

        with torch.no_grad():
            with torch.amp.autocast(device_type=self.device.type, enabled=autocast_enabled_inference):
                try:
                     gen_kwargs = {
                         "input_ids": inputs.input_ids,
                         "attention_mask": inputs.attention_mask,
                         "max_length": max_length_generation,
                         "num_beams": num_beams_generation,
                         "early_stopping": True,
                         "no_repeat_ngram_size": 3,
                         "output_scores": True,          # <<< Request scores for log p(y|x,z)
                         "return_dict_in_generate": True # <<< Get structured output
                     }
                     gen_output = self.generator.generate(**gen_kwargs)
                     sequences = gen_output.sequences
                     score_logits = gen_output.scores # List of tensors (batch, vocab) per step
                except Exception as e:
                     logger_rag.error(f"Generation failed for query '{query[:50]}...': {e}", exc_info=True)
                     return f"Error during generation: {e}" # Return error message

        # 5. Decode
        if sequences is None:
             logger_rag.error("Generation did not produce sequences.")
             return "Could not generate an answer (no sequences)."

        decoded_answers = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sequences]

        if not decoded_answers:
            return "Could not generate an answer (decoding failed)."

        # 6. Post-process / Select Final Answer based on Strategy
        if strategy == "fast" or not retrieved_docs_with_probs:
            # Return the single generated answer
            return decoded_answers[0]

        elif strategy == "thorough":
            # --- Updated Re-ranking Logic using log p(y|x,z) + log p(z|x) ---
            final_selected_answer = "Error: Re-ranking failed." # Default fallback

            if score_logits is None:
                 logger_rag.warning("Generator did not return scores. Cannot calculate log p(y|x,z). Selecting first answer.")
                 # Fallback: just return the answer generated from the top document's prompt
                 final_selected_answer = decoded_answers[0]

            else:
                 # Calculate log p(y|x,z) for each generated sequence
                 log_py_list = []
                 for j in range(sequences.shape[0]): # Iterate through batch items (n_docs)
                     seq = sequences[j]
                     log_probs = []
                     for t, logits in enumerate(score_logits):
                         logit_row = logits[j]
                         log_softmax = torch.log_softmax(logit_row, dim=-1)
                         token_id = seq[t + 1].item() if (t + 1) < len(seq) else None
                         # Check if token_id is valid index for log_softmax
                         if token_id is not None and token_id >= 0 and token_id < log_softmax.shape[-1]:
                             log_probs.append(log_softmax[token_id].item())
                         else:
                             break # Stop if padding, EOS, or invalid index encountered
                     total_logprob = sum(log_probs)
                     log_py_list.append(total_logprob)

                 # Combine scores and rank
                 ranked_candidates = []
                 doc_probabilities_p_z = [doc_info['p_z_given_x'] for doc_info in retrieved_docs_with_probs]

                 min_len = min(len(decoded_answers), len(doc_probabilities_p_z), len(log_py_list))
                 if min_len != n_docs:
                      logger_rag.warning(f"Mismatch in lengths during thorough re-ranking (generate_answer): "
                                         f"Answers={len(decoded_answers)}, "
                                         f"DocProbs={len(doc_probabilities_p_z)}, "
                                         f"LogPy={len(log_py_list)}, "
                                         f"Expected={n_docs}")

                 for i in range(min_len):
                     answer_text = decoded_answers[i]
                     pz = doc_probabilities_p_z[i]
                     logpy = log_py_list[i]
                     score = float('-inf') # Default score
                     if pz > 1e-9: # Avoid log(0)
                          # Combined score: log p(y|x,z) + log p(z|x)
                          score = logpy + math.log(pz)
                     ranked_candidates.append((answer_text, score, logpy, pz)) # Store parts for potential debugging

                 if not ranked_candidates:
                      # Fallback if something went wrong
                      logger_rag.warning("No candidates to rank in thorough strategy. Returning first answer.")
                      final_selected_answer = decoded_answers[0]
                 else:
                      # Sort by the combined score (higher is better)
                      ranked_candidates.sort(key=lambda x: x[1], reverse=True)
                      # Select the answer text with the highest combined score
                      final_selected_answer = ranked_candidates[0][0]
            # --- End Updated Re-ranking Logic ---

            # Optional Wandb logging
            # Ensure inference_table_infer exists and is a wandb.Table if using this
            if wandb.run is not None and hasattr(self, 'inference_table_infer') and isinstance(getattr(self, 'inference_table_infer', None), wandb.Table):
                 try:
                     top_score_display = 'N/A'
                     if 'ranked_candidates' in locals() and ranked_candidates:
                          top_score = ranked_candidates[0][1]
                          top_score_display = f"{top_score:.4f}" if isinstance(top_score, float) else str(top_score)

                     top_docs_display_html = "<hr>".join([
                         f"<b>Score p(z|x): {doc_info['p_z_given_x']:.4f}</b> (Orig: {doc_info['original_score']:.2f})<br>{doc_info['text'][:200]}..."
                         for doc_info in retrieved_docs_with_probs[:3] # Log top 3 for brevity
                     ])
                     input_len_for_log = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') and inputs.input_ids is not None else 'N/A'
                     # Check if table has expected columns before adding data
                     expected_cols = ["Query", "Generated Answer", "Top Docs", "Input Length", "Top Score"] # Adjust if needed
                     if all(col in self.inference_table_infer.columns for col in expected_cols):
                         self.inference_table_infer.add_data(query, final_selected_answer, wandb.Html(top_docs_display_html), input_len_for_log, top_score_display)
                     else:
                          logger_rag.warning(f"Wandb table columns mismatch. Expected {expected_cols}, got {self.inference_table_infer.columns}. Skipping add_data.")

                 except Exception as e:
                     logger_rag.error(f"Wandb logging failed during generate_answer: {e}", exc_info=True)


            return final_selected_answer
        else:
            # Should not be reached
            logger_rag.error(f"Reached unexpected part of generate_answer logic with strategy: {strategy}")
            return decoded_answers[0]



    
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
                wandb.log({"epoch": epoch_num + 1, "mean_loss": mean_loss_this_epoch})
                logger_rag.info(f"✅ Epoch {epoch_num + 1} Mean Loss: {mean_loss_this_epoch:.4f}")
            else:
                 logger_rag.warning(f"Epoch {epoch_num + 1} had no valid batches.")
        logger_rag.info(f"RAG training completed for stage: {stage_name}.")
        return all_epoch_losses
    
    def generate_answers_batch(
        self,
        queries: List[str],
        strategy: str = "thorough",   # "fast" or "thorough"
        num_beams_generation: int = 3,
        max_length_generation: int = 128,
        top_k: int = None,
        use_sampling: bool = False,
        generator_batch_size: int = 8
    ) -> List[str]:
        """
        Batch version of generate_answer that reuses _retrieve_topk_docs for retrieval,
        then batches the generation step for efficiency. Ranks with log p(y|x,z) + log p(z|x).
        """
        self.generator.eval()
        self.dense_retriever.query_encoder.eval()

        if not queries:
            return []

        k = top_k or self.k_retrieval_for_inference

        # 1) Retrieve per-query using the same logic as generate_answer
        all_docs_with_probs = [
            self._retrieve_topk_docs(q, k)
            for q in queries
        ]
        # all_docs_with_probs: List[List[{"text","p_z_given_x","original_score"}]]

        # 2) Build flat map of prompts and document probabilities
        flat_map = []
        for qi, docs in enumerate(all_docs_with_probs):
            q = queries[qi]
            if not docs:
                flat_map.append({"idx": qi, "prompt": f"question: {q} context: ", "p_z": 0.0, "fast": True})
            elif strategy == "fast":
                top = docs[0]
                flat_map.append({"idx": qi, "prompt": f"question: {q} context: {top['text']}", "p_z": top["p_z_given_x"], "fast": True})
            else:  # thorough
                for d in docs:
                    flat_map.append({"idx": qi, "prompt": f"question: {q} context: {d['text']}", "p_z": d["p_z_given_x"], "fast": False})

        # 3) Batch-generate answers and compute log p(y|x,z) for each
        prompts = [item["prompt"] for item in flat_map]
        answers = [None] * len(prompts)
        log_py = [None] * len(prompts)
        autocast_inf = self.use_fp16 and self.device.type == "cuda"
        with torch.no_grad(), torch.amp.autocast(device_type=self.device.type, enabled=autocast_inf):
            for i in range(0, len(prompts), generator_batch_size):
                chunk_prompts = prompts[i: i + generator_batch_size]
                inputs = self.tokenizer(
                    chunk_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)
                gen_kwargs = {
                    "input_ids": inputs.input_ids,
                    "attention_mask": inputs.attention_mask,
                    "max_length": max_length_generation,
                    "output_scores": True,
                    "return_dict_in_generate": True
                }
                if use_sampling:
                    gen_output = self.generator.generate(**gen_kwargs, do_sample=True, top_p=0.9, temperature=0.8)
                else:
                    gen_output = self.generator.generate(**gen_kwargs, num_beams=num_beams_generation, early_stopping=True, no_repeat_ngram_size=3)
                sequences = gen_output.sequences
                score_logits = gen_output.scores  # List of length L-1, each is (batch, vocab)
                # Decode
                decoded = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sequences]
                # Compute log p(y|x,z)
                # score_logits: list of tensors, each (batch, vocab), one per generated token (not including start token)
                # For each sequence, sum log_softmax at each generated token position for the generated token
                # The first token in sequences is the start token, so skip it
                # sequences: (batch, seq_len)
                # For each sequence, get token at position t+1, and get score_logits[t][:, token]
                # score_logits is a list of length seq_len-1
                for j in range(sequences.shape[0]):
                    seq = sequences[j]
                    # For each position t, get log_softmax over vocab, then select logprob of actual token
                    log_probs = []
                    for t, logits in enumerate(score_logits):
                        # logits: (batch, vocab)
                        logit_row = logits[j]
                        log_softmax = torch.log_softmax(logit_row, dim=-1)
                        token_id = seq[t+1].item() if t+1 < len(seq) else None
                        if token_id is not None:
                            log_probs.append(log_softmax[token_id].item())
                    total_logprob = sum(log_probs)
                    answers[i + j] = decoded[j]
                    log_py[i + j] = total_logprob

        # 4) Re-rank per query using log p(y|x,z) + log p(z|x)
        buckets: Dict[int, List[Tuple[str, float, float, bool]]] = {}
        # Store (answer, p_z, log_py, fast)
        for idx, item in enumerate(flat_map):
            ans = answers[idx]
            pz = item["p_z"]
            fast = item["fast"]
            logpy = log_py[idx]
            buckets.setdefault(item["idx"], []).append((ans, pz, logpy, fast))

        final_answers = []
        for qi in range(len(queries)):
            entries = buckets.get(qi, [])
            if not entries:
                final_answers.append("No answer generated.")
                continue
            # fast strategy or single entry
            if entries[0][3] or len(entries) == 1:
                final_answers.append(entries[0][0])
            else:
                # Use log p(y|x,z) + log p(z|x) for re-ranking, fallback if p(z|x)==0
                best_entry = None
                best_score = None
                for ans, pz, logpy, fast in entries:
                    if pz > 0:
                        score = logpy + math.log(pz)
                    else:
                        score = float('-inf')
                    if best_entry is None or score > best_score:
                        best_entry = ans
                        best_score = score
                if best_entry is not None and best_score > float('-inf'):
                    final_answers.append(best_entry)
                else:
                    # fallback: use first entry
                    final_answers.append(entries[0][0])

        return final_answers
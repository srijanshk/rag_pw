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

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, dense_retriever, device, alpha=0.5, fine_tune=False, total_steps=10000, k=10, inference_table_train=None, inference_table_infer=None, model_name="facebook/bart-base"):
        self.dense_retriever = dense_retriever
        self.device = device
        self.alpha = alpha
        self.fine_tune = fine_tune
        self.k = k
        self.total_steps = total_steps
        self.inference_table_train = inference_table_train
        self.inference_table_infer = inference_table_infer
        self.model_name = model_name

        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.generator = BartForConditionalGeneration.from_pretrained(self.model_name).to(device)

        if self.fine_tune:
            self._init_optimizer()
        self.scaler = GradScaler()
    
    def _init_optimizer(self):
        self.generator.train()
        combined_params = list(self.generator.parameters())


        if self.dense_retriever.fine_tune:
            print("Optimizer: Including Retriever parameters for End-to-End training.")
            self.dense_retriever.model.train() # Ensure retriever model is in train mode
            combined_params.extend(list(self.dense_retriever.model.parameters()))
        else:
            print("Optimizer: Including Generator parameters only.")

        # Initialize optimizer with potentially combined parameters
        self.optimizer = AdamW(combined_params, lr=3e-5)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * self.total_steps),
            num_training_steps=self.total_steps
        )

    def _retrieve_topk_docs(self, query: str, k: int):
        # results = self.dense_retriever.search(query, k)
        # total_score = sum(score for _, score in results)
        # if total_score == 0:
        #     total_score = 1e-6
        # return [(r["text"], score / total_score) for r, score in results]
        """
        Retrieves top-k documents using the retriever's standard search.
        NOTE: This uses the numpy-based Faiss search, which breaks gradients
              back to the retriever for implicit RAG loss updates.
        """
        results = self.dense_retriever.search(query, k) # Returns List[Tuple[metadata_dict, score]]

        # Process results for compatibility (extract text, normalize scores if needed)
        processed_results = []
        scores = []
        for metadata, score in results:
            text = metadata.get("text", "")
            if text: # Ensure text is not empty
                 processed_results.append((text, score))
                 scores.append(score)

        if not processed_results:
            return []

        # Normalize scores to be probabilities (sum to 1)
        # Faiss scores are often distances (lower is better), convert if necessary
        # Assuming higher score is better and already somewhat probability-like (e.g., cosine sim)
        # If scores are distances, they need inversion/transformation first.
        total_score = sum(s for s in scores)
        if total_score <= 0: # Avoid division by zero or negative scores
            # Fallback: uniform probability if scores are invalid
            num_results = len(processed_results)
            return [(text, 1.0 / num_results) for text, _ in processed_results]
        else:
             # Normalize scores to sum to 1
             return [(text, score / total_score) for text, score in processed_results]

    # def _log_marginal_likelihood(self, query: str, answer: str):
    #     """
    #     Calculates the negative marginal log-likelihood of the answer,
    #     marginalized over the retrieved documents.
    #     -\log \sum_{z} p(z|x) p(y|x, z)
    #     """
    #     # 1. Retrieve documents
    #     retrieved = self._retrieve_topk_docs(query, self.k)

    #     # Handle case where no documents are retrieved
    #     if not retrieved:
    #         logger.warning(f"No documents retrieved for query: '{query}'. Skipping loss calculation, returning high loss.")
    #         return torch.tensor(0.0, device=self.device, dtype=torch.float32, requires_grad=True)

    #     # 2. Prepare target labels once (same answer for all docs)
    #     targets_tokenized = self.tokenizer(
    #         answer,
    #         return_tensors="pt",
    #         padding=True,
    #         truncation=True,
    #         max_length=128 
    #     ).input_ids.to(self.device)

    #     # Prepare labels for loss calculation (ignore padding tokens)
    #     labels = targets_tokenized.clone()
    #     labels[labels == self.tokenizer.pad_token_id] = -100 # Common practice for HF models

    #     log_likelihoods_list = []
    #     p_z_scores = [] 

    #     # 3. Loop through each retrieved document to get individual log likelihoods
    #     for doc, p_z_score in retrieved:
    #         prompt = f"question: {query} context: {doc}"
    #         p_z_scores.append(p_z_score) # Collect score

    #         # Tokenize individual prompt
    #         inputs = self.tokenizer(
    #             prompt,
    #             return_tensors="pt",
    #             padding=True, # Pad this single input if needed, though usually handled by model
    #             truncation=True,
    #             max_length=512 # Ensure this matches generator's capability
    #         ).to(self.device)

    #         # Use mixed precision context for the forward pass
    #         with torch.cuda.amp.autocast():
    #             # Call generator for *this specific document*
    #             # Assumes generator is in train mode if fine-tuning
    #             outputs = self.generator(
    #                 input_ids=inputs.input_ids,
    #                 attention_mask=inputs.attention_mask,
    #                 labels=labels # Use the prepared labels
    #             )
    #             # outputs.loss for a single item batch IS the correct loss for that item
    #             log_likelihood_doc = -outputs.loss # log p(y|x, z) = -Loss(y | x, z)
    #             log_likelihoods_list.append(log_likelihood_doc)

    #     # Check if any likelihoods were calculated (handles edge cases if loop didn't run)
    #     if not log_likelihoods_list:
    #          logger.warning(f"Log likelihood list is empty for query: '{query}'. Returning high loss.")
    #          return torch.tensor(10.0, device=self.device, dtype=torch.float32, requires_grad=True)

    #     # 4. Convert collected scores and likelihoods to tensors
    #     log_likelihoods = torch.stack(log_likelihoods_list) # Tensor of [-loss_doc1, -loss_doc2, ...]
    #     p_z = torch.tensor(p_z_scores, device=self.device, dtype=torch.float32)

    #     # 5. Process retriever probabilities p(z|x)
    #     # Ensure scores are valid probabilities (non-negative, sum to 1)
    #     p_z = torch.clamp(p_z, min=1e-8) # Avoid zero scores before normalization/log
    #     # Normalize scores to get probabilities IF they weren't already probabilities
    #     # The current _retrieve_topk_docs *does* normalize, so this might be redundant, but safe.
    #     p_z_normalized = p_z / p_z.sum()

    #     log_pz = torch.log(p_z_normalized) # Take log of normalized probabilities

    #     # 6. Calculate marginal log likelihood using logsumexp
    #     # Formula: log_likelihoods = log p(y|x,z), log_pz = log p(z|x)
    #     # We want log sum_z exp(log p(y|x,z) + log p(z|x))
    #     marginal_log_likelihood = torch.logsumexp(log_likelihoods + log_pz, dim=0)

    #     # 7. Return the negative marginal log likelihood (the final loss for this query-answer pair)
    #     nll_loss = -marginal_log_likelihood
    #     return nll_loss

    def _log_marginal_likelihood(self, query: str, answer: str):
        retrieved = self._retrieve_topk_docs(query, self.k)
        if wandb.run is not None and self.inference_table_train is not None:
            retrieved_texts = "<br><br>".join([doc for doc, _ in retrieved][:10])
            self.inference_table_train.add_data(query, answer, wandb.Html(retrieved_texts))


        if not retrieved:
            logger.warning(f"No documents retrieved for query: '{query}'. Skipping loss calculation, returning high loss.")
            return torch.tensor(10.0, device=self.device, dtype=torch.float32, requires_grad=True)

        # Prepare prompts at once
        prompts = [f"question: {query} passage: {doc}" for doc, _ in retrieved]
        p_z_scores = [p_z for _, p_z in retrieved]

        # Tokenize all prompts together
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Tokenize targets once
        targets = self.tokenizer(
            [answer] * len(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).input_ids.to(self.device)

        # Prepare labels (ignore pad tokens)
        labels = targets.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # Forward pass: batch
        with torch.cuda.amp.autocast():
            outputs = self.generator(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            # outputs.loss is average over batch — not directly useful
            # We need per-example loss: get logits and manually compute loss

            logits = outputs.logits  # shape: [batch_size, seq_len, vocab_size]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            per_token_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            per_token_loss = per_token_loss.view(shift_labels.size())

            # Now per_token_loss: [batch_size, seq_len]
            per_sample_loss = per_token_loss.sum(dim=1) / (shift_labels != -100).sum(dim=1)

        # per_sample_loss: shape [batch_size]
        log_likelihoods = -per_sample_loss  # negate because we need log p(y|x,z)

        p_z = torch.tensor(p_z_scores, device=self.device, dtype=torch.float32)
        p_z = torch.clamp(p_z, min=1e-8)
        p_z_normalized = p_z / p_z.sum()
        log_pz = torch.log(p_z_normalized)

        marginal_log_likelihood = torch.logsumexp(log_likelihoods + log_pz, dim=0)

        nll = -marginal_log_likelihood

        return nll

    def train_on_batch(self, batch, batch_idx=None, epoch=None, stage=None):
        if not self.optimizer:
            raise RuntimeError("Optimizer not initialized. Ensure fine_tune=True during RAGPipeline init.")
        

        self.generator.train()
        if self.dense_retriever.fine_tune and any(p is rp for p in self.optimizer.param_groups[0]['params'] for rp in self.dense_retriever.model.parameters()):
             self.dense_retriever.model.train()
        
        total_loss = 0.0
        valid_items = 0

        self.optimizer.zero_grad()

        for item in batch:
            query = item["query"]
            answer = item["answers"][0]

            loss = self._log_marginal_likelihood(query, answer)

            if torch.isnan(loss) or torch.isinf(loss) or loss.item() == 0.0:
                logger.warning(f"[Epoch {epoch}] Batch {batch_idx} → Skipping NaN loss for query: {query}")
                continue

            self.scaler.scale(loss / len(batch)).backward()
            total_loss += loss.item()
            valid_items += 1

        if valid_items == 0:
            logger.warning(f"[Epoch {epoch}] Batch {batch_idx} → No valid samples, skipping optimizer step.")
            return None

        self.scaler.unscale_(self.optimizer)
        clip_grad_norm_(self.generator.parameters(), max_norm=1.0)

        is_retriever_in_optimizer = False
        retriever_params = list(self.dense_retriever.model.parameters())
        if retriever_params:
            for group in self.optimizer.param_groups:
                if any(p is rp for p in group['params'] for rp in retriever_params):
                     is_retriever_in_optimizer = True
                     break
            if is_retriever_in_optimizer and self.dense_retriever.fine_tune:
                clip_grad_norm_(self.dense_retriever.model.parameters(), max_norm=1.0)

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad()

        avg_loss = total_loss / valid_items

        wandb.log({"RAG_Loss": avg_loss, "Batch": batch_idx, "stage": stage})

        return avg_loss


    def generate_answer(self, query: str, strategy="thorough", top_k=None):
        top_k = top_k or self.k
        retrieved = self._retrieve_topk_docs(query, top_k)

        if strategy == "fast":
            doc, _ = retrieved[0]
            prompt = f"question: {query} passage: {doc}"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            output = self.generator.generate(**inputs, max_length=64)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        elif strategy == "thorough":
            prompts = [f"question: {query} passage: {doc}" for doc, _ in retrieved]
            p_z = [score for _, score in retrieved]

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.generator.generate(
                    **inputs,
                    max_length=64,
                    num_beams=5,
                    early_stopping=True,
                    no_repeat_ngram_size=2
                )
            answers = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

            candidate_scores = {}
            for answer, score in zip(answers, p_z):
                candidate_scores[answer] = candidate_scores.get(answer, 0.0) + score
            
            if wandb.run is not None and self.inference_table_infer is not None:
                retrieved_texts = "<br><br>".join([doc for doc, _ in retrieved][:5])
                final_answer = max(candidate_scores.items(), key=lambda x: x[1])[0]
                self.inference_table_infer.add_data(query, final_answer, wandb.Html(retrieved_texts))

            return max(candidate_scores.items(), key=lambda x: x[1])[0]

    def train(self, dataset, batch_size=8, epochs=3, stage ="Stage=2"):
        all_losses = []
        for epoch in range(epochs):
            logger.info(f"🔁 Starting Epoch {epoch + 1}/{epochs}")
            np.random.shuffle(dataset)

            num_batches = len(dataset) // batch_size
            batch_indices = range(0, len(dataset), batch_size)

            # tqdm for batches
            with tqdm(total=num_batches, desc=f"Epoch {epoch+1}", leave=True, dynamic_ncols=True) as pbar:
                for i in batch_indices:
                    batch = dataset[i: i + batch_size]
                    if len(batch) < batch_size:
                        continue
                    batch_idx = i // batch_size + 1
                    loss = self.train_on_batch(batch, batch_idx=batch_idx, epoch=epoch + 1, stage=stage)

                    pbar.set_postfix(loss=f"{loss:.4f}")
                    pbar.update(1)

                    all_losses.append(loss)

            epoch_loss = np.mean(all_losses[-num_batches:])
            wandb.log({"epoch_loss": epoch_loss, "epoch": {epoch + 1},"stage": stage})
            logger.info(f"✅ Finished Epoch {epoch + 1} — Mean Loss: {epoch_loss:.4f}")
        
        return all_losses



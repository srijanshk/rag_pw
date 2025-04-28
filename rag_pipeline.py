import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers import T5Tokenizer, T5ForConditionalGeneration, get_cosine_schedule_with_warmup
import numpy as np
from typing import List
from tqdm import tqdm
import logging

logging.basicConfig(
    format='[%(asctime)s] %(levelname)s: %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, dense_retriever, device, alpha=0.5, fine_tune=False, total_steps=10000, k=10):
        self.dense_retriever = dense_retriever
        self.device = device
        self.alpha = alpha
        self.fine_tune = fine_tune
        self.k = k

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.generator = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)

        if self.fine_tune:
            self.generator.train()
            self.optimizer = AdamW(self.generator.parameters(), lr=3e-5)
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=int(0.1 * total_steps),
                num_training_steps=total_steps
            )

    def _retrieve_topk_docs(self, query: str, k: int):
        results = self.dense_retriever.search(query, k)
        total_score = sum(score for _, score in results)
        if total_score == 0:
            total_score = 1e-6
        return [(r["text"], score / total_score) for r, score in results]

    def _log_marginal_likelihood(self, query: str, answer: str):
        retrieved = self._retrieve_topk_docs(query, self.k)
        losses = []
        weights = []

        for doc, p_z in retrieved:
            prompt = f"question: {query} context: {doc}"
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
            targets = self.tokenizer(answer, return_tensors="pt", padding=True, truncation=True, max_length=128).input_ids.to(self.device)

            output = self.generator(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=targets)
            log_p_y_given_xz = -output.loss
            losses.append(log_p_y_given_xz)
            weights.append(torch.tensor(p_z, device=self.device))

        log_probs = torch.stack(losses)
        weights = torch.stack(weights)
        return -torch.logsumexp(log_probs + torch.log(weights), dim=0)

    def train_on_batch(self, batch, batch_idx=None, epoch=None):
        total_loss = 0.0
        total_gen_loss = 0.0
        total_ret_loss = 0.0

        for item in batch:
            query = item["query"]
            answer = item["answers"][0]  # Using first answer as target

            retriever_loss = 0.0
            if self.fine_tune and self.dense_retriever.fine_tune:
                retriever_loss = self.dense_retriever.fine_tune_on_batch([item])

            gen_loss = self._log_marginal_likelihood(query, answer)
            total = gen_loss + self.alpha * retriever_loss

            total.backward()
            clip_grad_norm_(self.generator.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            total_loss += total.item()
            total_gen_loss += gen_loss.item()
            total_ret_loss += retriever_loss

        avg_gen_loss = total_gen_loss / len(batch)
        avg_ret_loss = total_ret_loss / len(batch)
        avg_total = total_loss / len(batch)

        logger.info(
            f"[Epoch {epoch}] Batch {batch_idx} | Gen Loss: {avg_gen_loss:.4f} | "
            f"Ret Loss: {avg_ret_loss:.4f} | Total: {avg_total:.4f}"
        )

        return avg_total


    def generate_answer(self, query: str, strategy="thorough", top_k=None):
        top_k = top_k or self.k
        retrieved = self._retrieve_topk_docs(query, top_k)

        if strategy == "fast":
            doc, _ = retrieved[0]
            prompt = f"question: {query} context: {doc}"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
            output = self.generator.generate(**inputs, max_length=64)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)

        elif strategy == "thorough":
            candidate_scores = {}
            for doc, p_z in retrieved:
                prompt = f"question: {query} context: {doc}"
                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(self.device)
                output = self.generator.generate(**inputs, max_length=64)
                answer = self.tokenizer.decode(output[0], skip_special_tokens=True)
                candidate_scores[answer] = candidate_scores.get(answer, 0.0) + p_z
            return max(candidate_scores.items(), key=lambda x: x[1])[0]

    def train(self, dataset, batch_size=8, epochs=3):
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
                    loss = self.train_on_batch(batch, batch_idx=batch_idx, epoch=epoch + 1)

                    pbar.set_postfix(loss=f"{loss:.4f}")
                    pbar.update(1)

                    all_losses.append(loss)

            epoch_loss = np.mean(all_losses[-num_batches:])
            logger.info(f"✅ Finished Epoch {epoch + 1} — Mean Loss: {epoch_loss:.4f}")
        
        return all_losses



import os
from typing import List
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch.cuda.amp import autocast, GradScaler
import logging

logger_retriever = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, index_path, metadata_path, device, model_name="intfloat/e5-large-v2", fine_tune_flag=False):
        self.model_name = model_name
        self.fine_tune_enabled = fine_tune_flag
        self.index = faiss.read_index(index_path) # Used for inference/search
        self.metadata = self._load_metadata(metadata_path)
        self.device = device

        self.query_encoder = SentenceTransformer(model_name).to(device)
        self.doc_encoder = SentenceTransformer(model_name).to(device) # Used for on-the-fly encoding
        self._freeze_encoder(self.doc_encoder) # Document encoder typically frozen for RAG retriever tuning

        if self.fine_tune_enabled:
            self._init_optimizer_and_scaler()
            self.query_encoder.train()
        else:
            self.query_encoder.eval()

    def _freeze_encoder(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _init_optimizer_and_scaler(self):
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.AdamW(self.query_encoder.parameters(), lr=2e-5)
            self.scaler = GradScaler()
            logger_retriever.info(f"Optimizer/Scaler initialized for DenseRetriever's query_encoder ({self.model_name}).")

    def enable_fine_tuning_mode(self):
        self.fine_tune_enabled = True
        if not hasattr(self, "optimizer"):
            self._init_optimizer_and_scaler()
        self.query_encoder.train()
        logger_retriever.info("DenseRetriever fine-tuning mode enabled. Query encoder set to train().")

    def disable_fine_tuning_mode(self):
        self.fine_tune_enabled = False
        self.query_encoder.eval()
        logger_retriever.info("DenseRetriever fine-tuning mode disabled. Query encoder set to eval().")

    def _load_metadata(self, path):
        meta = {}
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    meta[str(entry["id"])] = entry
                except json.JSONDecodeError as e:
                    logger_retriever.warning(f"Skipping malformed JSON line: {line.strip()} - Error: {e}")
        return meta

    def _apply_model_specific_prefixes(self, texts, text_type="query"):
        """Applies prefixes like 'query: ' or 'passage: ' based on model type."""
        if "e5" in self.model_name.lower():
            if text_type == "query":
                return [f"query: {t}" for t in texts]
            elif text_type == "passage":
                return [f"passage: {t}" for t in texts]
        return texts # No prefix for other models or if type is unknown

    def embed_texts_for_training(self, texts: List[str], text_type: str, encoder_to_use, is_encoder_trainable: bool):
        """
        Embeds texts (queries or passages) and returns tensors with gradient history
        if the encoder is trainable and in train mode.
        text_type: "query" or "passage" for prefixing.
        encoder_to_use: self.query_encoder or self.doc_encoder.
        is_encoder_trainable: True if this encoder is part of the backward pass.
        """
        prefixed_texts = self._apply_model_specific_prefixes(texts, text_type)

        original_training_state = encoder_to_use.training
        grad_context = torch.no_grad()

        if is_encoder_trainable:
            encoder_to_use.train() # Ensure it's in train mode for grads
            if self.fine_tune_enabled or encoder_to_use is self.query_encoder: # Enable grad if retriever FT is on OR specifically for query encoder
                 grad_context = torch.enable_grad()
        else:
            encoder_to_use.eval()

        with grad_context:
            embeddings = self._forward_pass(
                prefixed_texts,
                encoder_to_use,
                training_mode_for_encoder=is_encoder_trainable and encoder_to_use.training # Pass precise training mode
            )
        
        encoder_to_use.train(original_training_state) # Restore state
        return embeddings


    def _forward_pass(self, texts, encoder, batch_size=64, training_mode_for_encoder=False):
        embeddings_list = []
        original_encoder_training_state = encoder.training

        if training_mode_for_encoder:
            encoder.train()
            grad_context = torch.enable_grad()
        else:
            encoder.eval()
            grad_context = torch.no_grad()

        with grad_context:
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]

                batch_features = encoder.tokenize(batch_texts)
                batch_features = batch_to_device(batch_features, self.device)
                model_output = encoder.forward(batch_features)
                
                token_embeddings = model_output["token_embeddings"]
                attention_mask = batch_features["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                pooled_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1).clamp(min=1e-9)
                embeddings_list.append(pooled_embeddings)
        
        encoder.train(original_encoder_training_state)
        return torch.cat(embeddings_list, dim=0)

    def search(self, query: str, k=5): # This is for INFERENCE
        self.query_encoder.eval()
        prefixed_queries = self._apply_model_specific_prefixes([query], "query")
        
        query_embedding_np = self.query_encoder.encode(
            prefixed_queries,
            convert_to_numpy=True,
            normalize_embeddings=True, # Important for Faiss with inner product/cosine
            show_progress_bar=False,
            device=self.device
        )

        distances, indices = self.index.search(query_embedding_np, k)
        results = []
        if indices.size == 0 or distances.size == 0: return results
        for i in range(len(indices[0])):
            idx, score = indices[0][i], distances[0][i]
            if idx == -1: continue
            metadata = self.metadata.get(str(idx), {})
            text = metadata.get("text", "")
            if isinstance(text, str) and len(text.strip()) > 0:
                results.append({"id": idx, "text": text, "score": score, "title": metadata.get("title", "")})
        return results

    def fine_tune_on_batch(self, batch_of_triplets):
        if not self.fine_tune_enabled:
            logger_retriever.warning("fine_tune_on_batch called, but fine-tuning mode is not enabled. Skipping.")
            return 0.0
        if not hasattr(self, "optimizer"):
            logger_retriever.error("Optimizer not initialized for Retriever. Call enable_fine_tuning_mode() or init with fine_tune_flag=True.")
            return 0.0

        self.query_encoder.train()
        queries, docs_for_encoding, doc_groups_lengths = [], [], []

        for item in batch_of_triplets:
            query_text = item["query"]
            positive_doc_text = item["positive_doc"]
            negative_doc_texts = item.get("negative_docs", [])

            prefixed_query_list = self._apply_model_specific_prefixes([query_text], "query")
            prefixed_positive_doc_list = self._apply_model_specific_prefixes([positive_doc_text], "passage")
            prefixed_negative_docs_list = self._apply_model_specific_prefixes(negative_doc_texts, "passage")
            
            queries.append(prefixed_query_list[0])
            current_docs_in_group = prefixed_positive_doc_list + prefixed_negative_docs_list
            docs_for_encoding.extend(current_docs_in_group)
            doc_groups_lengths.append(len(current_docs_in_group))

        query_embeddings = self._forward_pass(queries, self.query_encoder, training_mode_for_encoder=True)
        doc_embeddings_flat = self._forward_pass(docs_for_encoding, self.doc_encoder, training_mode_for_encoder=False) # doc_encoder is frozen

        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings_flat = F.normalize(doc_embeddings_flat, p=2, dim=1)

        grouped_doc_embeddings, current_pos = [], 0
        for length in doc_groups_lengths:
            grouped_doc_embeddings.append(doc_embeddings_flat[current_pos : current_pos + length])
            current_pos += length
        
        total_loss = 0.0
        for i in range(len(query_embeddings)):
            q_emb = query_embeddings[i].unsqueeze(0)
            doc_group_embs = grouped_doc_embeddings[i]
            scores = torch.mm(q_emb, doc_group_embs.transpose(0, 1)).squeeze(0)
            targets = torch.zeros(1, dtype=torch.long, device=self.device)
            loss = F.cross_entropy(scores.unsqueeze(0), targets)
            total_loss += loss
        
        if not batch_of_triplets: return 0.0
        average_loss = total_loss / len(batch_of_triplets)

        self.optimizer.zero_grad()
        self.scaler.scale(average_loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.query_encoder.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return average_loss.item()

    def save_query_encoder(self, path):
        logger_retriever.info(f"Saving DenseRetriever's query_encoder to {path}")
        self.query_encoder.save(path)

    @classmethod
    def load_from_paths(cls, model_load_path, index_path, metadata_path, device, base_model_name_for_doc_encoder="intfloat/e5-large-v2"):
        instance = cls(index_path, metadata_path, device, model_name=base_model_name_for_doc_encoder, fine_tune_flag=False)
        logger_retriever.info(f"Loading query_encoder from {model_load_path} into DenseRetriever.")
        instance.query_encoder = SentenceTransformer(model_load_path, device=str(device))
        instance.query_encoder.to(device)
        return instance
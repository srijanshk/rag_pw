import os
from typing import List, Tuple
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import json
import numpy as np
import json as _json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch.amp import GradScaler
import logging
from transformers import AutoTokenizer

logger_retriever = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self,
                 index_path,
                 metadata_path,
                 device,
                 model_name="intfloat/e5-large-v2",
                 fine_tune=False,
                 use_fp16: bool = True,
                 ef_search: int = 1500,
                 ef_construction: int = 200, doc_encoder_model=None):
        self.model_name = model_name
        self.doc_encoder_model = doc_encoder_model if doc_encoder_model else model_name
        self.fine_tune_enabled = fine_tune
        self.index = faiss.read_index(index_path) # Used for inference/search

        # unwrap any wrapper indices (IDMap, IDMap2, Replicas, Shards) to get the raw HNSW index
        inner = self.index
        while hasattr(inner, "index"):
            inner = inner.index
        # ensure we have the proper subclass wrapper for HNSW
        inner = faiss.downcast_index(inner)

        # ensure it's truly an HNSW index
        if not hasattr(inner, "hnsw"):
            raise ValueError(f"Index at {index_path} isn’t an HNSW index (found {type(inner)})")

        # tune HNSW parameters
        inner.hnsw.efSearch = ef_search
        inner.hnsw.efConstruction = ef_construction
        logger_retriever.info(f"Using HNSW index; set efSearch={ef_search}, efConstruction={ef_construction}")
        

        self.metadata = self._load_metadata(metadata_path)


        self.device = device
        self.use_fp16 = use_fp16

        self.query_encoder = SentenceTransformer(model_name).to(device)
        # Enable gradient checkpointing if available
        if hasattr(self.query_encoder, "gradient_checkpointing_enable"):
            self.query_encoder.gradient_checkpointing_enable()
            logger_retriever.info("Gradient checkpointing enabled on query encoder.")
        self.doc_encoder = SentenceTransformer(doc_encoder_model).to(device)

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
    
    def update_hnsw_params(self, ef_search: int = 50, ef_construction: int = 200):
        """
        Update the HNSW parameters for the FAISS index.
        :param ef_search: The number of nearest neighbors to search for.
        :param ef_construction: The number of neighbors to consider during index construction.
        """
        inner = self.index
        while hasattr(inner, "index"):
            inner = inner.index
        # ensure we have the proper subclass wrapper for HNSW
        inner = faiss.downcast_index(inner)
        
        inner.hnsw.efSearch = ef_search
        inner.hnsw.efConstruction = ef_construction
        logger_retriever.info(f"Updated HNSW parameters: efSearch={ef_search}, efConstruction={ef_construction}")

    def _init_optimizer_and_scaler(self):
        if not hasattr(self, "optimizer"):
            self.optimizer = torch.optim.AdamW(self.query_encoder.parameters(), lr=2e-5)
            self.scaler = GradScaler(enabled=(self.use_fp16 and self.device.type == 'cuda'))
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
    
    def search_batch_with_embeddings(self, 
                                 query_embeddings_batch: np.ndarray, 
                                 k: int,
                                 normalize_query_embeddings: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Searches the FAISS index with a batch of query embeddings.

        Args:
            query_embeddings_batch: A NumPy array of query embeddings (batch_size, embedding_dim).
            k: Number of documents to retrieve per query.
            normalize_query_embeddings: Whether to L2 normalize query embeddings before search. 
                                        Set to True if your index was built with normalized vectors 
                                        and uses inner product/cosine similarity. Your existing



                                        `search` method does normalize for E5.
        Returns:
            A tuple of (distances_batch, indices_batch).
        """
        self.query_encoder.eval()

        embeddings_to_search = query_embeddings_batch
        if normalize_query_embeddings and "e5" in self.model_name.lower():
            # L2 normalize
            faiss.normalize_L2(embeddings_to_search)

        return self.index.search(embeddings_to_search, k)

    def save_query_encoder(self, path):
        logger_retriever.info(f"Saving DenseRetriever's query_encoder to {path}")
        self.query_encoder.save(path)

    @classmethod
    def load_from_paths(
        cls,
        model_load_path: str,
        index_path: str,
        metadata_path: str,
        device,
        base_model_name_for_doc_encoder: str = "intfloat/e5-large-v2",
    ):
        instance = cls(
            index_path,
            metadata_path,
            device,
            model_name=base_model_name_for_doc_encoder,
            fine_tune=False
        )
        logger_retriever.info(f"Loading query_encoder from {model_load_path} into DenseRetriever.")
        instance.query_encoder = SentenceTransformer(model_load_path, device=str(device))
        instance.query_encoder.to(device)
        return instance
    

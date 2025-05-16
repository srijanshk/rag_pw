import os
from typing import List
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
                 passage_tokenizer_name_or_path: str = None,
                 id2row_path: str = None,
                 input_ids_path: str = None,
                 attention_mask_path: str = None,
                 num_passages: int = None,
                 passage_max_len: int = 512,
                 fine_tune=False,
                 use_fp16: bool = True,
                 ef_search: int = 50,
                 ef_construction: int = 200):
        self.model_name = model_name
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

        # Load pre-tokenized passages via memmap if provided
        if id2row_path and input_ids_path and attention_mask_path and num_passages is not None:
            with open(id2row_path, 'r') as f:
                self.id2row = _json.load(f)
            shape = (num_passages, passage_max_len)
            self.passage_input_ids = np.memmap(input_ids_path, mode='r', dtype=np.int64, shape=shape)
            self.passage_attention_mask = np.memmap(attention_mask_path, mode='r', dtype=np.int64, shape=shape)
        else:
            self.passage_input_ids = None
            self.passage_attention_mask = None

        self.device = device
        self.use_fp16 = use_fp16

        self.query_encoder = SentenceTransformer(model_name).to(device)
        # Enable gradient checkpointing if available
        if hasattr(self.query_encoder, "gradient_checkpointing_enable"):
            self.query_encoder.gradient_checkpointing_enable()
            logger_retriever.info("Gradient checkpointing enabled on query encoder.")
        self.doc_encoder = SentenceTransformer(model_name).to(device) # Used for on-the-fly encoding

        self._freeze_encoder(self.doc_encoder) # Document encoder typically frozen for RAG retriever tuning

        # Initialize tokenizer for passages; tokenization will be on-demand in `search()`
        if passage_tokenizer_name_or_path:
            self.passage_tokenizer = AutoTokenizer.from_pretrained(passage_tokenizer_name_or_path)
        else:
            self.passage_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

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


    def _forward_pass(self, texts: List[str], encoder: SentenceTransformer, batch_size: int = 64, training_mode_for_encoder: bool = False):
        if not texts: # Handle empty input list
            embedding_dim = encoder.get_sentence_embedding_dimension()
            return torch.empty((0, embedding_dim if embedding_dim else 768), device=self.device) # Default dim if not found

        embeddings_list = []
        autocast_active = self.use_fp16 and self.device.type == 'cuda'
        original_module_training_state = encoder.training
        
        if training_mode_for_encoder: encoder.train()
        else: encoder.eval()

        with torch.set_grad_enabled(training_mode_for_encoder):
            with torch.amp.autocast(device_type=self.device.type, enabled=autocast_active):
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    features = encoder.tokenize(batch_texts)
                    features = batch_to_device(features, self.device)
                    model_output = encoder.forward(features)
                    
                    token_embeddings = model_output['token_embeddings']
                    attention_mask = features['attention_mask']
                    
                    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                    sum_mask = input_mask_expanded.sum(1).clamp(min=1e-9)
                    pooled_embeddings = sum_embeddings / sum_mask
                    embeddings_list.append(pooled_embeddings)
        
        encoder.train(original_module_training_state)
        return torch.cat(embeddings_list, dim=0) if embeddings_list else torch.empty((0, encoder.get_sentence_embedding_dimension()), device=self.device)



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
            if idx == -1:
                continue
            meta = self.metadata.get(str(idx), {})
            text = meta.get("text", "")
            if not (isinstance(text, str) and text.strip()):
                continue
            passage_max_len = self.passage_input_ids.shape[1] if self.passage_input_ids is not None else 512
            if self.passage_input_ids is not None and self.passage_attention_mask is not None:
                # load from memmap
                row = self.id2row.get(str(idx))
                if row is None:
                    continue
                input_ids_np = self.passage_input_ids[row]
                attention_mask_np = self.passage_attention_mask[row]
                input_ids = torch.from_numpy(input_ids_np).long().to(self.device)
                attention_mask = torch.from_numpy(attention_mask_np).long().to(self.device)
            else:
                # on-demand tokenize
                text = self._apply_model_specific_prefixes([text], "passage")[0]
                tok = self.passage_tokenizer(
                    text,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=passage_max_len,
                )
                input_ids = tok.input_ids.squeeze(0).to(self.device)
                attention_mask = tok.attention_mask.squeeze(0).to(self.device)
            results.append({
                "id": idx,
                "text": text,
                "score": score,
                "title": meta.get("title", ""),
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            })
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
        
        with torch.amp.autocast(device_type=self.device.type, enabled=self.scaler.is_enabled()):
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

    def batch_search(self, queries: List[str], k: int = 5, batch_size_embed: int = 128):
        """
        Batch process multiple queries, returning a list of results for each query.
        :param queries: A list of query strings.
        :param k: The number of top documents to retrieve for each query.
        :param batch_size_embed: Batch size for the embedding model's forward pass.
        :return: A list of lists, where each inner list contains result dictionaries for a query.
        """
        if not queries:
            return []

        self.query_encoder.eval() # Ensure encoder is in evaluation mode
        prefixed_queries = self._apply_model_specific_prefixes(queries, "query")

        # Batch encode using _forward_pass, which handles autocast for FP16
        query_embeddings = self._forward_pass(
            prefixed_queries,
            self.query_encoder,
            batch_size=batch_size_embed, # Use the provided batch_size_embed
            training_mode_for_encoder=False
        )

        if query_embeddings.numel() == 0: # Should not happen if queries is not empty
            return [[] for _ in queries]


        query_embeddings_normalized = F.normalize(query_embeddings, p=2, dim=1)
        query_embeddings_np = query_embeddings_normalized.detach().cpu().numpy()

        try:
            # FAISS batch search
            distances, indices = self.index.search(query_embeddings_np, k)
        except Exception as e:
            logger_retriever.error(f"Faiss batch search failed: {e}")
            # Return empty results for all queries in case of Faiss error
            return [[] for _ in queries]

        batch_results = []
        for i in range(len(queries)): # Iterate through each query's results from FAISS
            single_query_results = []
            if i >= indices.shape[0]: # Safety check, should not happen
                logger_retriever.warning(f"Index out of bounds for query {i} in FAISS results. Skipping.")
                batch_results.append(single_query_results)
                continue

            for j in range(k): # Iterate through top-k documents for the current query
                if j >= indices.shape[1]: # Safety check for k
                    break
                idx = indices[i][j]
                dist = distances[i][j]

                if idx == -1: # FAISS uses -1 for no result or padded results
                    continue

                metadata_entry = self.metadata.get(str(idx))
                if not metadata_entry:
                    logger_retriever.warning(f"No metadata found for Faiss index {idx} (query {i}). Skipping.")
                    continue

                text = metadata_entry.get("text", "")
                if isinstance(text, str) and len(text.strip()) > 0:
                    single_query_results.append({
                        "id": int(idx),       # FAISS index usually int
                        "text": text,
                        "score": float(dist), # Ensure score is standard float
                        "title": metadata_entry.get("title", "")
                    })
            batch_results.append(single_query_results)
        if torch.cuda.is_available():
            logger_retriever.info(f"batch_search: CUDA memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        return batch_results

    @classmethod
    def load_from_paths(
        cls,
        model_load_path: str,
        index_path: str,
        metadata_path: str,
        device,
        passage_tokenizer_name_or_path: str = None,
        base_model_name_for_doc_encoder: str = "intfloat/e5-large-v2",
    ):
        instance = cls(
            index_path,
            metadata_path,
            device,
            model_name=base_model_name_for_doc_encoder,
            passage_tokenizer_name_or_path=passage_tokenizer_name_or_path,
            fine_tune=False
        )
        logger_retriever.info(f"Loading query_encoder from {model_load_path} into DenseRetriever.")
        instance.query_encoder = SentenceTransformer(model_load_path, device=str(device))
        instance.query_encoder.to(device)
        return instance
    

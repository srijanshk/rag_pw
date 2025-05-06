import os
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device
from torch.cuda.amp import autocast, GradScaler
import logging # Added for logging

logger_retriever = logging.getLogger(__name__) # Added for logging

class DenseRetriever:
    def __init__(self, index_path, metadata_path, device, model_name="intfloat/e5-large-v2", fine_tune_flag=False): # Renamed fine_tune to fine_tune_flag
        self.model_name = model_name
        self.fine_tune_enabled = fine_tune_flag # Internal flag for enabling fine-tuning mode
        self.index = faiss.read_index(index_path)
        self.metadata = self._load_metadata(metadata_path)
        self.device = device

        # Load separate encoders
        self.query_encoder = SentenceTransformer(model_name).to(device)
        self.doc_encoder = SentenceTransformer(model_name).to(device) # Used for on-the-fly encoding in fine_tune_on_batch
        self._freeze_encoder(self.doc_encoder) # Document encoder is always frozen

        if self.fine_tune_enabled: # If instantiated with fine-tuning capabilities
            self._init_optimizer_and_scaler() # Renamed for clarity
            self.query_encoder.train() # Start in train mode if fine-tuning is intended
        else:
            self.query_encoder.eval() # Start in eval mode otherwise

    def _freeze_encoder(self, model):
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

    def _init_optimizer_and_scaler(self):
        """Initializes optimizer and GradScaler for the query_encoder."""
        if not hasattr(self, "optimizer"): # Initialize only if not already present
            self.optimizer = torch.optim.AdamW(self.query_encoder.parameters(), lr=2e-5) # Standard LR for ST
            self.scaler = GradScaler()
            logger_retriever.info(f"Optimizer and GradScaler initialized for DenseRetriever's query_encoder ({self.model_name}).")

    def enable_fine_tuning_mode(self):
        """Enables fine-tuning mode for the query_encoder."""
        self.fine_tune_enabled = True
        if not hasattr(self, "optimizer"): # Ensure optimizer is there
            self._init_optimizer_and_scaler()
        self.query_encoder.train()
        logger_retriever.info("DenseRetriever fine-tuning mode enabled. Query encoder set to train().")

    def disable_fine_tuning_mode(self):
        """Disables fine-tuning mode and sets query_encoder to eval."""
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
                    logger_retriever.warning(f"Skipping malformed JSON line in metadata: {line.strip()} - Error: {e}")
        return meta

    def embed_query(self, text_or_texts, for_training_loss_computation=False):
        """
        Embeds queries.
        :param text_or_texts: A single query string or a list of query strings.
        :param for_training_loss_computation: If True, returns tensors on device with grads enabled (if model is in train mode).
                                     If False (default for search), returns numpy arrays, normalized, with no_grad.
        """
        is_single_query = isinstance(text_or_texts, str)
        queries = [text_or_texts] if is_single_query else text_or_texts

        # Add "query: " prefix for e5 models
        # This prefixing should ideally be tied to the model type more robustly
        if "e5" in self.query_encoder.tokenizer.name_or_path.lower():
             prefixed_queries = [f"query: {q}" for q in queries]
        else:
             prefixed_queries = queries


        current_model_mode_is_train = self.query_encoder.training

        # Determine context for gradients
        if for_training_loss_computation and current_model_mode_is_train :
            context = torch.enable_grad()
        else: # For search or if model is in eval mode
            context = torch.no_grad()

        with context:
            embeddings = self.query_encoder.encode(
                prefixed_queries,
                convert_to_tensor=for_training_loss_computation, # True if for loss, False for numpy search
                convert_to_numpy=not for_training_loss_computation, # True for numpy search
                normalize_embeddings=not for_training_loss_computation, # Normalize for Faiss search, not for loss (loss does it)
                device=self.device,
                show_progress_bar=False
            )
        return embeddings


    def _forward_pass(self, texts, encoder, batch_size=64, training_mode_for_encoder=False):
        """
        Helper to pass texts through an encoder and get pooled embeddings.
        training_mode_for_encoder: Determines if gradients should be enabled for this specific pass.
        """
        embeddings_list = []
        original_encoder_training_state = encoder.training

        if training_mode_for_encoder:
            encoder.train() # Set to train mode for this pass if specified
            grad_context = torch.enable_grad()
        else:
            encoder.eval() # Set to eval mode for this pass
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
                pooled_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1)
                embeddings_list.append(pooled_embeddings)

        # Restore original training state of the encoder
        encoder.train(original_encoder_training_state)

        return torch.cat(embeddings_list, dim=0)


    def search(self, query: str, k=5):
        self.query_encoder.eval() # Ensure query encoder is in eval mode for search
        query_embedding_np = self.embed_query(query, for_training_loss_computation=False)
        # embed_query already normalizes and converts to numpy when for_training_loss_computation=False

        # Faiss search (assuming index uses inner product and expects normalized vectors)
        # If index uses L2, normalization is still good.
        # self.index.nprobe can be set here if it's an IVF-type index
        # faiss.normalize_L2(query_embedding_np) # Already done by embed_query if normalize_embeddings=True

        distances, indices = self.index.search(query_embedding_np, k)
        
        results = []
        if indices.size == 0 or distances.size == 0: # Check for empty results
            return results

        for batch_idx in range(indices.shape[0]): # Iterate if query was a batch (though usually single for search)
            for i in range(len(indices[batch_idx])):
                idx = indices[batch_idx][i]
                score = distances[batch_idx][i]
                if idx == -1: # Faiss can return -1 for invalid indices
                    continue
                metadata = self.metadata.get(str(idx), {})
                text = metadata.get("text", "")
                if isinstance(text, str) and len(text.strip()) > 0: # Check if text is valid
                    results.append({"id": idx, "text": text, "score": score, "title": metadata.get("title", "")})
                # else:
                #     logger_retriever.debug(f"[SKIP] Invalid or too-short text for doc #{idx} in metadata.")
        return results


    def fine_tune_on_batch(self, batch_of_triplets):
        """
        Fine-tunes the query_encoder on a batch of (query, positive_doc, negative_docs) triplets.
        This method is for training the retriever itself, separately from a RAG pipeline.
        """
        if not self.fine_tune_enabled:
            logger_retriever.warning("fine_tune_on_batch called, but fine-tuning mode is not enabled. Skipping.")
            return 0.0
        if not hasattr(self, "optimizer"):
            logger_retriever.error("Optimizer not initialized. Call enable_fine_tuning_mode() or ensure DenseRetriever was initialized with fine_tune_flag=True.")
            return 0.0

        self.query_encoder.train() # Ensure query encoder is in train mode
        # self.doc_encoder is already frozen and in eval mode

        queries = []
        docs_for_encoding = [] # Will store positive and negative docs for batch encoding
        doc_groups_lengths = [] # Stores the number of docs (1 positive + N negatives) for each query

        for item in batch_of_triplets:
            query_text = item["query"]
            positive_doc_text = item["positive_doc"]
            negative_doc_texts = item.get("negative_docs", [])

            # Apply prefixes based on model type (example for e5)
            if "e5" in self.model_name.lower():
                prefixed_query = f"query: {query_text}"
                prefixed_positive_doc = f"passage: {positive_doc_text}"
                prefixed_negative_docs = [f"passage: {neg_doc}" for neg_doc in negative_doc_texts]
            else: # Fallback or other model types might not need prefixes or have different ones
                prefixed_query = query_text
                prefixed_positive_doc = positive_doc_text
                prefixed_negative_docs = negative_doc_texts

            queries.append(prefixed_query)
            current_docs_in_group = [prefixed_positive_doc] + prefixed_negative_docs
            docs_for_encoding.extend(current_docs_in_group)
            doc_groups_lengths.append(len(current_docs_in_group))

        # Encode queries (gradients needed)
        # Using _forward_pass to get PyTorch tensors directly without SentenceTransformer's .encode() overhead for training
        query_embeddings = self._forward_pass(queries, self.query_encoder, training_mode_for_encoder=True)

        # Encode documents (no gradients needed for doc_encoder as it's frozen)
        doc_embeddings_flat = self._forward_pass(docs_for_encoding, self.doc_encoder, training_mode_for_encoder=False)

        # Normalize embeddings for cosine similarity based loss
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        doc_embeddings_flat = F.normalize(doc_embeddings_flat, p=2, dim=1)

        # Reconstruct document groups
        grouped_doc_embeddings = []
        current_pos = 0
        for length in doc_groups_lengths:
            grouped_doc_embeddings.append(doc_embeddings_flat[current_pos : current_pos + length])
            current_pos += length
        
        # Calculate loss for each query
        total_loss = 0.0
        for i in range(len(query_embeddings)):
            q_emb = query_embeddings[i].unsqueeze(0)  # [1, D]
            doc_group_embs = grouped_doc_embeddings[i] # [num_docs_in_group, D]

            # Scores: dot product of query with all its positive/negative docs
            scores = torch.mm(q_emb, doc_group_embs.transpose(0, 1)).squeeze(0)  # [num_docs_in_group]
            
            # Contrastive loss: positive is at index 0
            # Targets for cross_entropy: tensor of zeros, as positive is always first.
            targets = torch.zeros(1, dtype=torch.long, device=self.device) # Only one query processed at a time here for simplicity in score shaping
            
            # Reshape scores for CrossEntropyLoss: [N, C] where N=1 (batch_size for this single query)
            # and C=num_docs_in_group
            loss = F.cross_entropy(scores.unsqueeze(0), targets)
            total_loss += loss
        
        if not batch_of_triplets: # Avoid division by zero if batch is empty
             return 0.0
        
        average_loss = total_loss / len(batch_of_triplets)

        self.optimizer.zero_grad()
        self.scaler.scale(average_loss).backward()
        self.scaler.unscale_(self.optimizer) # Unscale before clipping
        torch.nn.utils.clip_grad_norm_(self.query_encoder.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return average_loss.item()

    def save_query_encoder(self, path):
        logger_retriever.info(f"Saving DenseRetriever's query_encoder to {path}")
        self.query_encoder.save(path)

    @classmethod
    def load_from_paths(cls, model_load_path, index_path, metadata_path, device, base_model_name_for_doc_encoder="intfloat/e5-large-v2"):
        """
        Loads a DenseRetriever. The query_encoder is loaded from model_load_path.
        The doc_encoder is initialized from base_model_name_for_doc_encoder and frozen.
        """
        # Instantiate with fine_tune_flag=False initially, as optimizer state isn't saved by ST.
        # The base_model_name is used for the doc_encoder.
        instance = cls(index_path, metadata_path, device, model_name=base_model_name_for_doc_encoder, fine_tune_flag=False)
        
        logger_retriever.info(f"Loading fine-tuned query_encoder from {model_load_path} into DenseRetriever.")
        instance.query_encoder = SentenceTransformer(model_load_path, device=str(device)) # device mapping
        instance.query_encoder.to(device) # Ensure it's on the correct device.
        
        # If you intend to continue fine-tuning after loading, you might want to re-enable fine-tuning mode
        # and re-initialize the optimizer for the newly loaded query_encoder.
        # instance.enable_fine_tuning_mode() # Optional: if you want to continue training the loaded model
        
        return instance

# class DenseRetriever:
#     def __init__(self, index_path, metadata_path, device, model_name="intfloat/e5-large-v2", fine_tune=False):
#         self.fine_tune = fine_tune
#         self.index = faiss.read_index(index_path)
#         self.metadata = self._load_metadata(metadata_path)

#         self.device = device

#         # Load separate encoders
#         self.query_encoder = SentenceTransformer(model_name).to(device)
#         self.doc_encoder = SentenceTransformer(model_name).to(device)
#         self._freeze_encoder(self.doc_encoder)

#         if self.fine_tune:
#             self._init_optimizer()

    
#     def _freeze_encoder(self, model):
#         for param in model.parameters():
#             param.requires_grad = False
#         model.eval()

#     def _init_optimizer(self):
#         self.query_encoder.train()
#         self.optimizer = torch.optim.AdamW(self.query_encoder.parameters(), lr=2e-5)
#         self.scaler = GradScaler()
    
#     def enable_fine_tuning(self):
#         self.fine_tune = True
#         if not hasattr(self, "optimizer"):
#             self._init_optimizer()
#         self.query_encoder.train()
    
#     def disable_fine_tuning(self):
#         """Disables fine-tuning mode and sets model to eval."""
#         self.fine_tune = False
#         self.query_encoder.eval()

#     def _load_metadata(self, path):
#         meta = {}
#         with open(path) as f:
#             for line in f:
#                 entry = json.loads(line)
#                 meta[str(entry["id"])] = entry
#         return meta


#     def embed_query(self, text, train_mode=False):
#         if isinstance(text, str):
#             text = [f"query: {text}"]
#         else:
#             text = [f"query: {t}" for t in text]
#         # if train_mode:
#         #     embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device, show_progress_bar=False )
#         # else:
#         #     embeddings = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False )
#         # return embeddings
#         encoder = self.query_encoder if train_mode else self.query_encoder.eval()
#         with torch.no_grad() if not train_mode else torch.enable_grad():
#             embeddings = encoder.encode(
#                 text,
#                 convert_to_tensor=train_mode,
#                 convert_to_numpy=not train_mode,
#                 normalize_embeddings=True,
#                 device=self.device
#             )
#         return embeddings
    
#     def mean_pooling(model_output, attention_mask):
#         token_embeddings = model_output['token_embeddings']
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#         return torch.sum(token_embeddings * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1)

#     def _forward_pass(self, texts, encoder, batch_size=64):
#         embeddings = []
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i: i + batch_size]
#             batch = encoder.tokenize(batch)
#             batch = batch_to_device(batch, self.device)
#             with torch.no_grad():
#                 output = encoder.forward(batch)
#             token_embeddings = output["token_embeddings"]
#             attention_mask = batch["attention_mask"]
#             input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1)
#             embeddings.append(pooled)
#         return torch.cat(embeddings, dim=0)


#     def search(self, query, k=5):
#         self.index.nprobe = 128
#         query_embedding = self.embed_query(query, train_mode=False)
#         if isinstance(query_embedding, torch.Tensor):
#             query_embedding = query_embedding.cpu().numpy()
#         faiss.normalize_L2(query_embedding)

#         distances, indices = self.index.search(query_embedding, k)
#         results = []
#         for idx, score in zip(indices[0], distances[0]):
#             metadata = self.metadata.get(str(idx), {})
#             text = metadata.get("text", "")
#             if isinstance(text, str) and len(text.strip()) > 30:
#                 results.append((metadata, score))
#         return results

#     def fine_tune_on_batch(self, batch):
#         self.query_encoder.train()

#         queries = []
#         docs = []
#         doc_per_query = []

#         for item in batch:
#             query = item["query"]
#             negs = item.get("negatives", [])
#             queries.append(query)
#             docs.append(item["positive"])
#             docs.extend(negs)
#             doc_per_query.append(1 + len(negs))

#         query_embeddings = self._forward_pass(queries, self.query_encoder)
#         doc_embeddings = self._forward_pass(docs, self.doc_encoder)

#         query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
#         doc_embeddings = F.normalize(doc_embeddings, p=2, dim=1)

#         padded_doc_embeddings = []
#         start = 0
#         max_docs = max(doc_per_query)

#         for count in doc_per_query:
#             group = doc_embeddings[start:start + count]
#             if count < max_docs:
#                 pad = torch.zeros((max_docs - count, doc_embeddings.size(1)), device=self.device)
#                 group = torch.cat([group, pad], dim=0)
#             padded_doc_embeddings.append(group)
#             start += count

#         doc_tensor = torch.stack(padded_doc_embeddings)         # [B, max_docs, D]
#         query_tensor = query_embeddings.unsqueeze(1)            # [B, 1, D]

#         with autocast():
#             scores = torch.bmm(doc_tensor, query_tensor.transpose(1, 2)).squeeze(-1)  # [B, max_docs]
#             targets = torch.zeros(len(batch), dtype=torch.long, device=self.device)
#             loss = F.cross_entropy(scores, targets)

#         self.optimizer.zero_grad()
#         self.scaler.scale(loss).backward()
#         self.scaler.unscale_(self.optimizer)
#         torch.nn.utils.clip_grad_norm_(self.query_encoder.parameters(), 1.0)
#         self.scaler.step(self.optimizer)
#         self.scaler.update()

#         return loss.item()
    
#     def save(self, path):
#         self.query_encoder.save(path)

#     @classmethod
#     def load(cls, path, index_path, metadata_path, device, fine_tune=False):
#         instance = cls(index_path, metadata_path, device)
#         instance.query_encoder = SentenceTransformer(path, device=str(device))
#         return instance
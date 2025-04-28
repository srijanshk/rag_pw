import os
import torch
import torch.nn.functional as F
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import batch_to_device

class DenseRetriever:
    def __init__(self, index_path, metadata_path, device, model_name="intfloat/e5-large-v2", fine_tune=False):
        self.fine_tune = fine_tune
        self.index = faiss.read_index(index_path)
        self.metadata = self._load_metadata(metadata_path)

        self.model = SentenceTransformer(model_name, device=str(device))
        self.device = device

        if self.fine_tune:
            self._init_optimizer()  # ✅ initialize optimizer if fine_tune is True

    def _init_optimizer(self):
        self.model.train()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5)
    
    def enable_fine_tuning(self):
        self.fine_tune = True
        if not hasattr(self, "optimizer"):  # 🔐 Only initialize if not already set
            self._init_optimizer()

    def _load_metadata(self, path):
        if path.endswith(".json"):
            with open(path) as f:
                return json.load(f)
        elif path.endswith(".jsonl"):
            with open(path) as f:
                return {str(i): json.loads(line) for i, line in enumerate(f)}
        else:
            raise ValueError("Unsupported metadata file format")

    def embed_query(self, text, train_mode=False):
        if isinstance(text, str):
            text = [f"query: {text}"]
        else:
            text = [f"query: {t}" for t in text]
        if train_mode:
            embeddings = self.model.encode(text, convert_to_tensor=True, device=self.device, show_progress_bar=False )
        else:
            embeddings = self.model.encode(text, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False )
        return embeddings
    
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output['token_embeddings']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1)


    def _forward_pass(self, texts):
        """Tokenizes and forwards inputs through the encoder with gradient tracking."""
        batch = self.model.tokenize(texts)
        batch = batch_to_device(batch, self.device)

        model_output = self.model.forward(batch)

        # Apply mean pooling manually
        token_embeddings = model_output['token_embeddings']
        attention_mask = batch['attention_mask']
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1) / input_mask_expanded.sum(dim=1)

        return embeddings


    def search(self, query, k=5):
        query_embedding = self.embed_query(query)
        if isinstance(query_embedding, torch.Tensor):
            query_embedding = query_embedding.cpu().numpy()
        faiss.normalize_L2(query_embedding)

        distances, indices = self.index.search(query_embedding, k)
        results = []
        for idx, score in zip(indices[0], distances[0]):
            metadata = self.metadata.get(str(idx), {})
            text = metadata.get("text", "")
            if isinstance(text, str) and len(text.strip()) > 30:
                results.append((metadata, score))
            else:
                print(f"[SKIP] Invalid or too-short text for doc #{idx}")
        return results

    def fine_tune_on_batch(self, batch):
        self.model.train()
        queries = []
        docs = []
        doc_per_query = []

        for item in batch:
            query = item["query"]
            pos = item["positive"]
            negs = item.get("negatives", [])

            queries.append(query)
            docs.append(pos)
            docs.extend(negs)
            doc_per_query.append(1 + len(negs))

        query_embeddings = self._forward_pass(queries)  # [B, D]
        doc_embeddings = self._forward_pass(docs)       # [sum(docs), D]

        padded_doc_embeddings = []
        start = 0
        max_docs = max(doc_per_query)

        for count in doc_per_query:
            group = doc_embeddings[start:start + count]
            if count < max_docs:
                padding = torch.zeros((max_docs - count, doc_embeddings.size(1)), device=self.device)
                group = torch.cat([group, padding], dim=0)
            padded_doc_embeddings.append(group)
            start += count

        doc_tensor = torch.stack(padded_doc_embeddings)            # [B, max_docs, D]
        query_tensor = query_embeddings.unsqueeze(1)               # [B, 1, D]
        scores = torch.bmm(doc_tensor, query_tensor.transpose(1, 2)).squeeze(-1)  # [B, max_docs]

        targets = torch.zeros(len(batch), dtype=torch.long, device=self.device)  # always first doc is positive

        loss = F.cross_entropy(scores, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()


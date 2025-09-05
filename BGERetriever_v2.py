from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


class BGERetriever:
    def __init__(
        self,
        embedding_model: BGEM3FlagModel,
        index_path: str | Path,
        metadata_path: str | Path,
        device: str = "cuda",
        ef_search: int = 300,
    ) -> None:
        """
        Initialize BGE-based retriever.
        
        Args:
            embedding_model: Pre-loaded BGEM3FlagModel instance
            index_path: Path to FAISS index file
            metadata_path: Path to metadata JSONL file
            device: Device to use for embeddings
            ef_search: HNSW search parameter
        """
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = embedding_model

        # Load FAISS index
        print(f"Loading FAISS index from {index_path}")
        self.index = faiss.read_index(str(index_path))
        
        # Set HNSW search parameters if applicable
        inner = self.index
        while hasattr(inner, "index"):
            inner = inner.index
        inner = faiss.downcast_index(inner)
        if hasattr(inner, "hnsw"):
            inner.hnsw.efSearch = ef_search
            print(f"Set efSearch to {ef_search}")
        
        self.dim = self.index.d
        print(f"Index dimension: {self.dim}")

        # Load metadata
        print(f"Loading metadata from {metadata_path}")
        self.metadata: Dict[int, Dict[str, Any]] = {}
        with Path(metadata_path).open() as f:
            for line_num, line in enumerate(f):
                try:
                    d = json.loads(line)
                    doc_id = int(d["id"])
                    self.metadata[doc_id] = d
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    print(f"Warning: Skipping malformed metadata line {line_num + 1}: {e}")
        
        print(f"Loaded {len(self.metadata)} metadata entries")

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query using BGE-M3."""
        try:
            result = self.model.encode(
                [query], 
                return_dense=True, 
                return_sparse=False,
                return_colbert_vecs=False
            )
            vec = result["dense_vecs"].astype(np.float32)
            
            # Normalize for cosine similarity
            faiss.normalize_L2(vec)
            
            return vec
        except Exception as e:
            print(f"Error embedding query '{query}': {e}")
            # Return zero vector as fallback
            return np.zeros((1, self.dim), dtype=np.float32)

    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of documents with scores and metadata
        """
        if not query or not query.strip():
            return []
        
        try:
            # Embed query
            query_vec = self._embed_query(query.strip())
            
            # Search index
            scores, indices = self.index.search(query_vec, k)
            
            # Collect results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # FAISS returns -1 for missing results
                    continue
                    
                # Get metadata
                metadata = self.metadata.get(int(idx), {})
                
                # Create result document
                doc = {
                    "id": int(idx),
                    "dense_score": float(score),
                    "problem": metadata.get("problem", ""),
                    "solution_chunk": metadata.get("solution_chunk", ""),
                    "text": metadata.get("text", ""),
                    "row_id": metadata.get("row_id", ""),
                    "chunk_id": metadata.get("chunk_id", ""),
                    "expected_answer": metadata.get("expected_answer", ""),
                    "problem_from": metadata.get("problem_from", ""),
                }
                
                # Add combined text field if not present
                if not doc["text"]:
                    doc["text"] = f"Problem: {doc['problem']}\nSolution: {doc['solution_chunk']}"
                
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error during search for query '{query}': {e}")
            return []

    def search_batch(self, queries: List[str], k: int = 20) -> List[List[Dict[str, Any]]]:
        """
        Batch search for multiple queries.
        
        Args:
            queries: List of query strings
            k: Number of results per query
            
        Returns:
            List of result lists (one per query)
        """
        if not queries:
            return []
        
        try:
            # Embed all queries
            query_vecs = []
            valid_queries = []
            
            for query in queries:
                if query and query.strip():
                    vec = self._embed_query(query.strip())
                    query_vecs.append(vec[0])  # Remove batch dimension
                    valid_queries.append(query)
                else:
                    query_vecs.append(np.zeros(self.dim, dtype=np.float32))
                    valid_queries.append("")
            
            if not query_vecs:
                return [[] for _ in queries]
            
            # Stack into batch
            query_batch = np.stack(query_vecs)
            
            # Batch search
            scores_batch, indices_batch = self.index.search(query_batch, k)
            
            # Process results for each query
            all_results = []
            for i, (query, scores, indices) in enumerate(zip(valid_queries, scores_batch, indices_batch)):
                if not query:
                    all_results.append([])
                    continue
                
                results = []
                for score, idx in zip(scores, indices):
                    if idx == -1:
                        continue
                    
                    metadata = self.metadata.get(int(idx), {})
                    doc = {
                        "id": int(idx),
                        "dense_score": float(score),
                        "problem": metadata.get("problem", ""),
                        "solution_chunk": metadata.get("solution_chunk", ""),
                        "text": metadata.get("text", ""),
                        "row_id": metadata.get("row_id", ""),
                        "chunk_id": metadata.get("chunk_id", ""),
                        "expected_answer": metadata.get("expected_answer", ""),
                        "problem_from": metadata.get("problem_from", ""),
                    }
                    
                    if not doc["text"]:
                        doc["text"] = f"Problem: {doc['problem']}\nSolution: {doc['solution_chunk']}"
                    
                    results.append(doc)
                
                all_results.append(results)
            
            return all_results
            
        except Exception as e:
            print(f"Error during batch search: {e}")
            return [[] for _ in queries]
    
    @staticmethod
    def _colbert_scores_safe(
        query: str,
        docs: List[Dict[str, Any]],
        model: BGEM3FlagModel,
        batch_pairs: int = 16,
    ) -> List[float]:
        """
        Robust ColBERT scorer.

        • Handles OOM / shape errors by falling back to each doc’s dense score.
        • Splits work into chunks of ≤ `batch_pairs` to reduce peak memory.
        """
        pairs = [[query, d.get("solution_chunk") or d.get("text", "")] for d in docs]
        scores: List[float] = []

        for i in range(0, len(pairs), batch_pairs):
            chunk = pairs[i : i + batch_pairs]
            try:
                s = model.compute_score_single_device(
                    chunk, batch_size=batch_pairs, max_query_length=128
                )["colbert"]
            except (RuntimeError, ValueError):
                # fall back to dense score if ColBERT fails
                s = [d["dense_score"] for d in docs[i : i + batch_pairs]]
            scores.extend(float(x) for x in s)
        return scores
    
    def search_and_rerank(
        self,
        query: str,
        top_k: int = 50,
        top_k_final: int = 10,
        batch_pairs: int = 16,
    ) -> List[Dict[str, Any]]:
        """
        1. Retrieve `top_k` candidates with dense BGE vectors.
        2. Rerank them with ColBERT-style late-interaction scores.
        3. Return the `top_k_final` highest-scoring docs.
        """
        # ── Stage 1: dense retrieval ────────────────────────────────
        docs = self.search(query, k=top_k)
        if not docs:
            return []

        # ── Stage 2: ColBERT reranking ─────────────────────────────
        scores = self._colbert_scores_safe(
            query=query,
            docs=docs,
            model=self.model,
            batch_pairs=batch_pairs,
        )
        for doc, s in zip(docs, scores):
            doc["rerank_score"] = s

        # ── Stage 3: sort & truncate ───────────────────────────────
        docs.sort(key=lambda d: d["rerank_score"], reverse=True)
        return docs[:top_k_final]

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "num_documents": len(self.metadata),
            "index_dimension": self.dim,
            "index_total_vectors": self.index.ntotal,
            "device": self.device,
        }
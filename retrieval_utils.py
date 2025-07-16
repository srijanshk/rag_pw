from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import faiss  # type: ignore
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel


class BGERetriever:
    def __init__(
        self,
        index_path: str | Path,
        meta_path: str | Path,
        device: str = "cuda",
        ef_search: int = 300,
    ) -> None:
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"

        # --- Faiss index ---
        self.index = faiss.read_index(str(index_path))
        inner = self.index
        while hasattr(inner, "index"):
            inner = inner.index
        inner = faiss.downcast_index(inner)
        if hasattr(inner, "hnsw"):
            inner.hnsw.efSearch = ef_search
        self.dim = self.index.d

        # --- metadata ---
        self.meta: Dict[int, Dict[str, Any]] = {}
        with Path(meta_path).open() as f:
            for line in f:
                d = json.loads(line)
                self.meta[int(d["id"])] = d

        # --- BGE‑m3 embedder ---
        self.embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device=self.device)

    # ---------- internal helpers ----------
    def _embed(self, query: str) -> np.ndarray:
        vec = self.embedder.encode([query], return_dense=True, return_sparse=False)["dense_vecs"]
        vec = np.ascontiguousarray(vec, dtype=np.float32)
        faiss.normalize_L2(vec)
        return vec

    @torch.no_grad()
    def _rerank_scores(self, query: str, docs: List[str]) -> List[float]:
        if self.rr_mdl is None:
            return [0.0] * len(docs)
        tok = self.rr_tok([(query, d) for d in docs], padding=True, truncation=True, return_tensors="pt").to(self.device)
        scores = self.rr_mdl(**tok).logits.squeeze(-1).float()
        return scores.cpu().tolist()

    # ---------- public search ----------
    def search(self, query: str, *, k: int = 20, rerank: int | None = None) -> List[Dict[str, Any]]:
        vec = self._embed(query)
        D, I = self.index.search(vec, k)
        docs = []
        for score, idx in zip(D[0], I[0]):
            meta = self.meta.get(idx, {})
            text = meta.get("text") or f"Q: {meta.get('problem','')} \n A: {meta.get('solution_chunk','')}"
            docs.append({
                **meta,
                "id": idx,
                "dense_score": float(score),
                "text": text,
            })

        if rerank and self.rr_mdl is not None:
            top_dense = docs[:rerank]
            scores = self._rerank_scores(query, [d["text"] for d in top_dense])
            for d, s in zip(top_dense, scores):
                d["rerank_score"] = s
            ordered = sorted(top_dense, key=lambda d: d["rerank_score"], reverse=True) + docs[rerank:]
        else:
            for d in docs:
                d["rerank_score"] = d["dense_score"]  # fallback
            ordered = docs

        # deduplicate identical text
        seen, uniq = set(), []
        for d in ordered:
            if d["text"] not in seen:
                seen.add(d["text"])
                uniq.append(d)
        for r, d in enumerate(uniq, 1):
            d["rank"] = r
        return uniq
    
    # -- batch version ----------------------------------------------------
    def search_batch(self, queries: list[str], k: int = 20,
                    rerank: int | None = None) -> list[list[dict]]:
        """Vectorise `search()` over many queries."""
        if not queries:
            return []

        q_vecs = self.embedder.encode(
            queries, return_dense=True, return_sparse=False)["dense_vecs"]
        q_vecs = np.ascontiguousarray(q_vecs, dtype=np.float32)
        faiss.normalize_L2(q_vecs)

        Ds, Is = self.index.search(q_vecs, k)
        all_results: list[list[dict]] = []
        for q, D, I in zip(queries, Ds, Is):
            docs = [self.meta[i] | {
                        "id": int(i),
                        "dense_score": float(d),
                        "text": self.meta[i].get("text") or
                                f"Q: {self.meta[i].get('problem','')}  "
                                f"A: {self.meta[i].get('solution_chunk','')}"
                    }
                    for d, i in zip(D, I)]
            # optional ColBERT rerank (same logic as single‑query search)
            if rerank:
                top = docs[:rerank]
                scores = self._colbert_scores(q, [d["text"] for d in top])
                for d, s in zip(top, scores):
                    d["rerank_score"] = s
                docs = sorted(top, key=lambda d: d["rerank_score"],
                            reverse=True) + docs[rerank:]
            all_results.append(docs)
        return all_results


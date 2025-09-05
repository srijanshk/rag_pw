from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import re
from collections import defaultdict

import faiss  # type: ignore
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel

class BGERetriever:
    def __init__(
        self,
        embedding_model: BGEM3FlagModel,
        index_path: str | Path,
        metadata_path: str | Path,          # can be JSONL (per-line objects) OR JSON list of chunk_ids
        example_map_path: str | Path,       # JSON: example_id -> {problem, solution}
        device: str = "cuda",
        ef_search: int = 300,               # for HNSW if present
        nprobe: int = 64,                   # for IVF/IVFPQ if present
        use_all_gpus: bool = True,          # move FAISS index to all GPUs if available
        chunk_texts_path: Optional[str | Path] = None,  # optional: JSON list of texts aligned to chunk_ids
    ) -> None:
        """
        Initialize BGE-based retriever with example map for full Q+A.

        metadata_path can be:
        - JSONL with one object per line:  {"id": <faiss_id>, "chunk_id": "EXAMPLEID_CHUNKIDX", ...}
        - JSON list of chunk_ids (index i -> chunk_ids[i] corresponds to FAISS vector i)
        """
        self.device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
        self.model = embedding_model

        # ---- FAISS index
        print(f"Loading FAISS index from {index_path}")
        cpu_index = faiss.read_index(str(index_path))
        if use_all_gpus and hasattr(faiss, "get_num_gpus") and faiss.get_num_gpus() > 0:
            ng = faiss.get_num_gpus()
            print(f"🚀 Using {ng} GPU(s) for FAISS search")
            self.index = faiss.index_cpu_to_all_gpus(cpu_index)
        else:
            self.index = cpu_index

        # IVF search tuning
        if hasattr(self.index, "nprobe"):
            try:
                self.index.nprobe = int(nprobe)
                print(f"Set IVF nprobe = {self.index.nprobe}")
            except Exception:
                pass

        # HNSW tuning if applicable
        inner = self.index
        while hasattr(inner, "index"):
            inner = inner.index
        inner = faiss.downcast_index(inner)
        if hasattr(inner, "hnsw"):
            try:
                inner.hnsw.efSearch = int(ef_search)
                print(f"Set HNSW efSearch = {ef_search}")
            except Exception:
                pass

        self.dim = self.index.d
        print(f"Index dimension: {self.dim}")
        print(f"Index total vectors: {self.index.ntotal}")

        # ---- Load metadata (auto-detect format)
        print(f"Loading chunk metadata from {metadata_path}")
        self.metadata: Dict[int, Dict[str, Any]] = {}
        meta_path = Path(metadata_path)
        with meta_path.open("r", encoding="utf-8") as f:
            # peek first non-whitespace char to detect JSON array vs JSONL
            first_char = f.read(1)
            while first_char and first_char.isspace():
                first_char = f.read(1)
            f.seek(0)

            if first_char == "[":
                # JSON array: assume it's a list of chunk_ids aligned to FAISS vector ids
                chunk_ids: List[str] = json.load(f)
                for i, cid in enumerate(chunk_ids):
                    self.metadata[i] = {"id": i, "chunk_id": cid}
                print(f"Detected JSON list of chunk_ids; loaded {len(self.metadata)} entries")
            else:
                # JSONL: each line is a JSON object with "id" and "chunk_id"
                count = 0
                for line_num, line in enumerate(f, start=1):
                    try:
                        d = json.loads(line)
                        vid = int(d["id"])
                        self.metadata[vid] = d
                        count += 1
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        if line_num <= 5:
                            print(f"Warning: Skipping malformed metadata line {line_num}: {e}")
                        continue
                print(f"Detected JSONL; loaded {count} entries")

        # Optional: load chunk_texts aligned to FAISS vector ids (improves reranker text)
        self._chunk_texts: Optional[List[str]] = None
        if chunk_texts_path:
            try:
                with Path(chunk_texts_path).open("r", encoding="utf-8") as ftxt:
                    self._chunk_texts = json.load(ftxt)  # list aligned with FAISS ids
                print(f"Loaded {len(self._chunk_texts)} chunk texts from {chunk_texts_path}")
            except Exception as e:
                print(f"Warning: failed to load chunk_texts from {chunk_texts_path}: {e}")

        # ---- Load example map (full Q+A)
        print(f"Loading example map from {example_map_path}")
        with Path(example_map_path).open("r", encoding="utf-8") as fex:
            raw_map: Dict[str, Dict[str, Any]] = json.load(fex)
        self.example_map: Dict[int, Dict[str, Any]] = {int(k): v for k, v in raw_map.items()}
        print(f"Loaded {len(self.example_map)} full examples")

    # -------------------------
    # Embedding helper
    # -------------------------
    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a single query using BGE-M3 (L2-normalized for cosine)."""
        try:
            out = self.model.encode(
                [query],
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
                max_length=8192
            )
            vec = out["dense_vecs"].astype(np.float32)
            faiss.normalize_L2(vec)
            return vec
        except Exception as e:
            print(f"Error embedding query '{query}': {e}")
            return np.zeros((1, self.dim), dtype=np.float32)

    # -------------------------
    # Utilities
    # -------------------------
    def _attach_full_example(self, chunk_meta: Dict[str, Any]) -> Dict[str, Any]:
        """Attach full problem and solution using example_id from chunk metadata."""
        example_id_str = chunk_meta.get("row_id")
        if example_id_str is None:
            cid = chunk_meta.get("chunk_id", "")
            example_id_str = cid.split("_")[0] if "_" in cid else None

        full_problem, full_solution = "", ""
        if example_id_str is not None:
            try:
                eid = int(example_id_str)
                ex = self.example_map.get(eid, {})
                full_problem = ex.get("problem", "")
                full_solution = ex.get("solution", "")
            except ValueError:
                pass

        out = dict(chunk_meta)
        out["full_problem"] = full_problem
        out["full_solution"] = full_solution
        return out

    def _doc_text_for_rerank(self, d: Dict[str, Any]) -> str:
        """
        Prefer more general text; truncate and mask numbers/answers to reduce instance leakage
        during re-ranking.
        """
        base = (
            (d.get("text") or "").strip()
            or (d.get("solution_chunk") or "").strip()
            or (d.get("full_solution") or "").strip()
            or (d.get("full_problem") or d.get("problem") or "").strip()
        )
        base = base[:1500]
        return self._mask_instance_bits(base)

    # === Helper utilities for safer rerank & diversity ===
    _NUM_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

    def _mask_instance_bits(self, s: str) -> str:
        """Mask numbers and common answer markers to reduce instance leakage."""
        if not s:
            return s
        s = self._NUM_RE.sub('#', s)
        s = s.replace('\\boxed', '').replace('The final answer is', '')
        return s

    def _group_key(self, d: Dict[str, Any]) -> str:
        """Grouping key per example to cap duplicates."""
        rid = d.get("row_id")
        if rid:
            return str(rid)
        cid = d.get("chunk_id", "")
        return cid.split("_")[0] if "_" in cid else cid or str(d.get("id", ""))

    def _dedup_and_diversify(self, docs: List[Dict[str, Any]], max_per_example: int = 1) -> List[Dict[str, Any]]:
        """Limit number of results per example for diversity."""
        counts = defaultdict(int)
        out: List[Dict[str, Any]] = []
        for d in docs:
            g = self._group_key(d)
            if counts[g] < max_per_example:
                out.append(d)
                counts[g] += 1
        return out

    def _mmr_lite(self, docs: List[Dict[str, Any]], k_final: int, sim_key: str = "rerank_score", novelty_weight: float = 0.2) -> List[Dict[str, Any]]:
        """
        Simple MMR-like selection to avoid near-duplicates.
        score = (1 - novelty_weight)*sim - novelty_weight*max_overlap
        Overlap is token-Jaccard on chosen text field. Keeps k_final items.
        """
        def text_of(d: Dict[str, Any]) -> str:
            return (d.get("text") or d.get("solution_chunk") or d.get("full_solution")
                    or d.get("full_problem") or d.get("problem") or "")

        def tokens(s: str) -> set:
            return set(re.findall(r'\w+', s.lower()))

        selected: List[Dict[str, Any]] = []
        selected_tok: List[set] = []
        pool = sorted(docs, key=lambda x: x.get(sim_key, x.get("dense_score", 0.0)), reverse=True)

        for d in pool:
            if len(selected) >= k_final:
                break
            t = tokens(text_of(d))
            if not selected:
                selected.append(d)
                selected_tok.append(t)
                continue
            overlaps = [(len(t & st) / max(1, len(t | st))) for st in selected_tok]
            max_ov = max(overlaps) if overlaps else 0.0
            sim = d.get(sim_key, d.get("dense_score", 0.0))
            score = (1 - novelty_weight) * sim - novelty_weight * max_ov
            if score >= 0 or max_ov < 0.5:
                selected.append(d)
                selected_tok.append(t)

        if len(selected) < k_final:
            for d in pool:
                if d in selected:
                    continue
                selected.append(d)
                if len(selected) >= k_final:
                    break
        return selected[:k_final]

    def _merge_dense_runs(self, runs: List[List[Dict[str, Any]]], qnames: List[str]) -> List[Dict[str, Any]]:
        """
        Merge multiple dense runs (main + alt queries) by id.
        Keep the doc with the BEST dense_score and tag the source query.
        """
        by_id: Dict[int, Dict[str, Any]] = {}
        for run, qn in zip(runs, qnames):
            for d in run:
                did = int(d["id"])
                if (did not in by_id) or (d.get("dense_score", -1e9) > by_id[did].get("dense_score", -1e9)):
                    d["_src_query"] = qn
                    by_id[did] = d
        return list(by_id.values())

    def _low_confidence(self, docs: List[Dict[str, Any]], min_score: float = -0.1) -> bool:
        """
        Flag weak retrieval to trigger query rewrite upstream (tune threshold).
        Uses rerank_score if available, else dense_score.
        """
        if not docs:
            return True
        top = docs[0]
        top_score = top.get("rerank_score", top.get("dense_score", -1e9))
        return top_score < min_score

    # -------------------------
    # Search (no rerank)
    # -------------------------
    def search(self, query: str, k: int = 20) -> List[Dict[str, Any]]:
        """FAISS ANN search → top-k chunks with metadata and full example text."""
        if not query or not query.strip():
            return []
        try:
            qvec = self._embed_query(query.strip())
            scores, indices = self.index.search(qvec, k)

            results: List[Dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:
                    continue
                meta = self.metadata.get(int(idx), {})
                doc = {
                    "id": int(idx),
                    "dense_score": float(score),
                    "chunk_id": meta.get("chunk_id", ""),
                    "problem": meta.get("problem", ""),
                    "solution_chunk": meta.get("solution_chunk", ""),
                    "text": meta.get("text", ""),
                    "row_id": meta.get("row_id", ""),
                    "expected_answer": meta.get("expected_answer", ""),
                    "problem_from": meta.get("problem_from", ""),
                }
                if not doc["text"]:
                    if doc["problem"] or doc["solution_chunk"]:
                        doc["text"] = f"Problem: {doc['problem']}\nSolution: {doc['solution_chunk']}"
                doc = self._attach_full_example(doc)
                results.append(doc)
            return results
        except Exception as e:
            print(f"Error during search: {e}")
            return []

    def search_batch(self, queries: List[str], k: int = 20) -> List[List[Dict[str, Any]]]:
        """Batched FAISS ANN search."""
        if not queries:
            return []
        qvecs = [
            self._embed_query(q.strip()) if q and q.strip() else np.zeros((1, self.dim), dtype=np.float32)
            for q in queries
        ]
        qmat = np.vstack(qvecs)
        scores_batch, indices_batch = self.index.search(qmat, k)

        all_res: List[List[Dict[str, Any]]] = []
        for scores, indices in zip(scores_batch, indices_batch):
            res: List[Dict[str, Any]] = []
            for score, idx in zip(scores, indices):
                if idx == -1:
                    continue
                meta = self.metadata.get(int(idx), {})
                doc = {
                    "id": int(idx),
                    "dense_score": float(score),
                    "chunk_id": meta.get("chunk_id", ""),
                    "problem": meta.get("problem", ""),
                    "solution_chunk": meta.get("solution_chunk", ""),
                    "text": meta.get("text", ""),
                    "row_id": meta.get("row_id", ""),
                    "expected_answer": meta.get("expected_answer", ""),
                    "problem_from": meta.get("problem_from", ""),
                }
                if not doc["text"]:
                    if doc["problem"] or doc["solution_chunk"]:
                        doc["text"] = f"Problem: {doc['problem']}\nSolution: {doc['solution_chunk']}"
                doc = self._attach_full_example(doc)
                res.append(doc)
            all_res.append(res)
        return all_res

    # -------------------------
    # ColBERT reranker (BGE-M3)
    # -------------------------
    def _colbert_scores_safe(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        batch_pairs: int = 16
    ) -> List[float]:
        """
        Compute ColBERT scores with BGE-M3, batched; back off batch size on OOM.
        """
        pairs = [[query, self._doc_text_for_rerank(d)] for d in docs]
        scores: List[float] = []
        i = 0
        while i < len(pairs):
            b = min(batch_pairs, len(pairs) - i)
            while True:
                try:
                    chunk_scores = self.model.compute_score_single_device(
                        pairs[i:i+b],
                        batch_size=b
                    )["colbert"]
                    break
                except RuntimeError:
                    if b <= 4:
                        # last-resort fallback: use dense_score for the remaining pairs
                        chunk_scores = [docs[i+j].get("dense_score", 0.0) for j in range(b)]
                        break
                    b = max(4, b // 2)  # back off
            scores.extend(float(s) for s in chunk_scores)
            i += b
        return scores

    def _rerank_colbert(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: int,
        batch_pairs: int = 16
    ) -> List[Dict[str, Any]]:
        if not docs:
            return []
        scores = self._colbert_scores_safe(query, docs, batch_pairs=batch_pairs)
        for d, s in zip(docs, scores):
            d["rerank_score"] = s
        return sorted(docs, key=lambda d: d.get("rerank_score", d.get("dense_score", 0.0)), reverse=True)[:top_k]

    # -------------------------
    # One-call retrieval + rerank
    # -------------------------
    def search_and_rerank(
        self,
        query: str,
        top_k: int = 200,
        top_k_final: int = 10,
        batch_pairs: int = 16,
        alt_queries: Optional[List[str]] = None,
        max_per_example: int = 1,
        use_mmr: bool = True,
        problem_text: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Dense-only fusion: run ANN search for main + alt queries, merge by id (keep best dense score),
        ColBERT re-rank, per-example cap, optional MMR-lite diversity, slice to k_final.
        """
        alt_queries = [q for q in (alt_queries or []) if q and q.strip()]
        qnames = ["q0"] + [f"q{i+1}" for i in range(len(alt_queries))]

        # 1) Dense searches
        runs = [self.search(query, k=top_k)] + [self.search(q, k=top_k) for q in alt_queries]

        # 2) Merge by id (best dense score wins)
        merged = self._merge_dense_runs(runs, qnames=qnames)
        if not merged:
            self._last_search_info = {
                "pool_merged": 0,
                "low_confidence": True,
                "top_score": None,
                "used_alt_queries": len(alt_queries) > 0
            }
            return []

        # 3) Re-rank with ColBERT over a manageable pool
        pool = sorted(merged, key=lambda x: x.get("dense_score", 0.0), reverse=True)[:max(top_k, top_k_final)]
        reranked = self._rerank_colbert(query, pool, top_k=len(pool), batch_pairs=batch_pairs)

        # 4) Cap per example and apply optional MMR-lite
        capped = self._dedup_and_diversify(reranked, max_per_example=max_per_example)
        final_docs = self._mmr_lite(capped, k_final=top_k_final) if use_mmr else capped[:top_k_final]

        # 5) Telemetry for upstream agent
        top_score = (final_docs[0].get("rerank_score", final_docs[0].get("dense_score", None)) if final_docs else None)
        self._last_search_info = {
            "pool_merged": len(merged),
            "pool_reranked": len(reranked),
            "pool_after_cap": len(capped),
            "top_score": top_score,
            "low_confidence": self._low_confidence(final_docs),
            "used_alt_queries": len(alt_queries) > 0,
        }
        return final_docs

    # -------------------------
    # Introspection
    # -------------------------
    def get_last_search_info(self) -> Dict[str, Any]:
        """Telemetry from the most recent search; safe to call after search_and_rerank."""
        return getattr(self, "_last_search_info", {})

    def get_stats(self) -> Dict[str, Any]:
        return {
            "num_documents": len(self.metadata),
            "num_examples": len(self.example_map),
            "index_dimension": self.dim,
            "index_total_vectors": self.index.ntotal,
            "device": self.device,
        }

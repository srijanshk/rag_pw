#!/usr/bin/env python
"""
openmath_retrieval_demo.py ──────────────────────────────────────────
Console demo that
1. Retrieves top‑K chunks from your **Faiss HNSW + BGE‑m3** index.
2. Re‑ranks them with **BAAI/bge‑reranker‑base** (cross‑encoder) and shows the
   final K‑final results.

Usage
─────
```bash
python openmath_retrieval_demo.py \
  --index /path/openmath_hnsw.index \
  --meta  /path/openmath_meta.jsonl \
  --k 20           # first‑stage Faiss hits (default 20) \
  --k-final 10     # how many after rerank (default 10)
```
The reranker runs on GPU if available, otherwise CPU (fp32).  It’s quick for
10‑20 docs so interactive latency stays <200 ms on a V100.
"""
from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" 

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict

import faiss  # type: ignore
import numpy as np
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#######################################################################
# CLI
#######################################################################
parser = argparse.ArgumentParser(description="Faiss + BGE‑reranker demo")
parser.add_argument("--index", type=Path, required=True, help="Faiss .index file")
parser.add_argument("--meta", type=Path, required=True, help="Metadata JSONL")
parser.add_argument("--k", type=int, default=100, help="First‑stage Faiss top‑K")
parser.add_argument("--k-final", type=int, default=10, help="Post‑rerank K to show")
args = parser.parse_args()

#######################################################################
# Load Faiss & metadata
#######################################################################
print("⏳  Loading Faiss index …", file=sys.stderr)
index = faiss.read_index(str(args.index))
print(f"✅  {index.ntotal:,} vectors, dim={index.d}")

print("⏳  Loading metadata …", file=sys.stderr)
id2meta: Dict[int, Dict] = {}
with args.meta.open() as f:
    for line in f:
        obj = json.loads(line)
        id2meta[int(obj["id"])] = obj
print(f"✅  {len(id2meta):,} meta rows loaded")

#######################################################################
# Load embedder (retriever) & reranker
#######################################################################
print("⏳  Loading BGEM3FlagModel …", file=sys.stderr)
embedder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True,
                          device="cuda" if torch.cuda.is_available() else "cpu")

def embed(text: str) -> np.ndarray:
    vec = embedder.encode([text], return_dense=True, return_sparse=False)["dense_vecs"]
    vec = np.ascontiguousarray(vec, dtype=np.float32)
    faiss.normalize_L2(vec)
    return vec

print("⏳  Loading bge‑reranker (cross‑encoder) …", file=sys.stderr)
rr_tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rr_mdl = AutoModelForSequenceClassification.from_pretrained(
    "BAAI/bge-reranker-base", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to("cuda" if torch.cuda.is_available() else "cpu").eval()

@torch.no_grad()
def rerank(query: str, docs: List[str]) -> List[float]:
    pairs = [(query, d) for d in docs]
    enc = rr_tok(pairs, padding=True, truncation=True, return_tensors="pt").to(rr_mdl.device)
    scores = rr_mdl(**enc).logits.squeeze(-1).float()
    return scores.cpu().tolist()

#######################################################################
# REPL loop
#######################################################################
print("\nReady!  (Enter on empty line to quit)\n")
while True:
    try:
        query = input("🔍  Query: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        break
    if not query:
        break

    # 1) dense search
    q_vec = embed(query)
    D, I = index.search(q_vec, args.k)
    cand_ids = I[0].tolist()
    cand_metas = [id2meta[i] for i in cand_ids]

    # 2) gather doc texts for rerank (use robust preview builder)
    cand_texts = []
    for m in cand_metas:
        if "text" in m:
            cand_texts.append(m["text"])
        else:
            if "problem" in m and "solution_chunk" in m:
                cand_texts.append(f"Q: {m['problem']}  A: {m['solution_chunk']}")
            else:
                cand_texts.append(m.get("problem", m.get("solution_chunk", "")))

# 3) rerank
rr_scores = rerank(query, cand_texts)
for meta, s in zip(cand_metas, rr_scores):
    meta["rerank_score"] = s

# 4) deduplicate by canonical text before final cut
seen_keys = set(); uniq: List[Dict] = []
for m in cand_metas:
    if "text" in m:
        key = m["text"]
    else:
        key = f"{m.get('problem', '')}//{m.get('solution_chunk', '')}"
    if key not in seen_keys:
        seen_keys.add(key)
        uniq.append(m)

# 5) sort & slice
final_docs = sorted(uniq, key=lambda d: d["rerank_score"], reverse=True)[: args.k_final]

# 4) display
print("\n🟢  Top hits (reranked)")
for rank, m in enumerate(final_docs, 1):
    preview_src = m.get("source", "?")
    preview_ans = m.get("expected_answer", "")
    if "text" in m:
        ptxt = m["text"]
    else:
        ptxt = f"Q: {m.get('problem', '')}  A: {m.get('solution_chunk', '')}"
    print(f"{rank:2d}. s_retr={m['rerank_score']:.4f} src={preview_src} ans={preview_ans}")
    print(f"    {ptxt}\n")

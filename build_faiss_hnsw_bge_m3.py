from __future__ import annotations

import argparse
import csv
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import tqdm

try:
    import wandb  # optional
except ModuleNotFoundError:
    wandb = None  # type: ignore

############################################################
# CLI                                                       
############################################################
parser = argparse.ArgumentParser(description="Build Faiss, BM25, ColBERT indexes + wandb logging")
parser.add_argument("--tsv", type=Path, required=True, help="Chunk TSV file")

# Faiss options
parser.add_argument("--faiss-index", type=Path, help="Output Faiss .index")
parser.add_argument("--faiss-meta", type=Path, help="Metadata JSONL for Faiss IDs")
parser.add_argument("--faiss-batch", type=int, default=256)
parser.add_argument("--faiss-gpu", action="store_true")
parser.add_argument("--faiss-m", type=int, default=64)
parser.add_argument("--faiss-ef", type=int, default=200)

# wandb options
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb-project", default="mathpile-indexes")
parser.add_argument("--wandb-entity", default=None)
parser.add_argument("--wandb-run-name", default=None)
args = parser.parse_args()

############################################################
# Weights & Biases setup                                   
############################################################
if args.wandb:
    if wandb is None:
        sys.exit("❌  pip install wandb to enable logging")
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name or "build-indexes",
        config={k: getattr(args, k) for k in vars(args) if k not in {"wandb", "wandb_project", "wandb_entity", "wandb_run_name"}},
    )

def wb_log(d: dict):
    if args.wandb:
        wandb.log(d)

script_start = time.time()

############################################################
# FAISS HNSW section                                       
############################################################
if args.faiss_index:
    try:
        from FlagEmbedding import BGEM3FlagModel
        import faiss  # type: ignore
    except ModuleNotFoundError as e:
        sys.exit(f"❌  Faiss dependencies missing: {e}")

    print("🔄  Loading BGE‑m3 for Faiss …", file=sys.stderr)
    model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True, device_map="auto")
    DIM = model.model.config.hidden_size

    if args.faiss_gpu:
        res = faiss.StandardGpuResources()
        base = faiss.IndexHNSWFlat(DIM, args.faiss_m, faiss.METRIC_INNER_PRODUCT)
        base.hnsw.efConstruction = args.faiss_ef
        idx = faiss.index_cpu_to_gpu(res, 0, base)
    else:
        idx = faiss.IndexHNSWFlat(DIM, args.faiss_m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = args.faiss_ef
    id_map = faiss.IndexIDMap(idx)

    if args.faiss_meta is None:
        args.faiss_meta = args.faiss_index.with_suffix(".jsonl")
    meta_fp = args.faiss_meta.open("w", encoding="utf-8")

    def embed(txts: List[str]) -> np.ndarray:
        vecs = model.encode(txts, batch_size=args.faiss_batch, max_length=600)["dense_vecs"].astype(np.float32)
        faiss.normalize_L2(vecs)
        return vecs

    processed, start = 0, time.time()
    batch_txt, batch_ids, batch_meta = [], [], []
    next_id = 0

    with args.tsv.open("r", encoding="utf-8", newline="") as fp:
        rdr = csv.DictReader(fp, delimiter="\t")
        for row in tqdm.tqdm(rdr, desc="Faiss embed+add"):
            batch_txt.append(row["text"].strip())
            batch_ids.append(next_id)
            row["id"] = next_id
            batch_meta.append(row)
            next_id += 1; processed += 1
            if len(batch_txt) >= 100_000:
                vecs = embed(batch_txt)
                id_map.add_with_ids(vecs, np.asarray(batch_ids, dtype=np.int64))
                for m in batch_meta: meta_fp.write(json.dumps(m) + "\n")
                batch_txt.clear(); batch_ids.clear(); batch_meta.clear()
                wb_log({"faiss_vectors": processed, "faiss_throughput": processed/(time.time()-start)})
        if batch_txt:
            vecs = embed(batch_txt)
            id_map.add_with_ids(vecs, np.asarray(batch_ids, dtype=np.int64))
            for m in batch_meta: meta_fp.write(json.dumps(m) + "\n")
    meta_fp.close()

    if args.faiss_gpu:
        id_map = faiss.index_gpu_to_cpu(id_map)
    faiss.write_index(id_map, str(args.faiss_index))
    wb_log({"faiss_total": processed, "faiss_time_sec": time.time()-start})
    print(f"✅  Faiss index → {args.faiss_index}")

wb_log({"total_elapsed_sec": time.time()-script_start})
if args.wandb:
    wandb.finish()
print("🎉  All requested indexes built.")

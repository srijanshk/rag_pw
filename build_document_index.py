import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import faiss
import csv
import json
import glob
import logging
from typing import List
from tqdm import tqdm
import torch


# Logging setup
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    "tsv_file": "downloads/data/wikipedia_split/psgs_w100.tsv",
    "index_path": "wikipedia_faiss_index",
    "metadata_path": "wikipedia_metadata.jsonl",
    "model_name": "intfloat/e5-large-v2",
    "batch_size": 256,
    "chunk_size": 4000,
    "min_token_count": 50,
    "faiss_use_ivf": True,
    "faiss_use_hnsw": False,
    "faiss_nlist": 4096,
    "train_sample_size": 500_000,
    "batch_embed_size": 1_000_000,
}

def yield_wikipedia_chunks(path, chunk_size):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=["id", "text", "title"])
        next(reader)  # skip header
        chunk = []
        for row in reader:
            if row["text"].strip():
                chunk.append({
                    "id": int(row["id"]),
                    "title": row["title"],
                    "text": row["text"]
                })
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk

def load_wikipedia_tsv(path: str) -> List[dict]:
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=["id", "text", "title"])
        next(reader)  # skip header
        for row in reader:
            if row["text"].strip():
                entries.append({
                    "id": row["id"],
                    "title": row["title"],
                    "text": row["text"]
                })
    return entries

# def save_metadata(metadata: List[dict], path: str):
#     with open(path, "w", encoding="utf-8") as f:
#         for entry in metadata:
#             f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def save_metadata(metadata: list, path: str):
    with open(path, "a", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
#     faiss.normalize_L2(embeddings)
#     dim = embeddings.shape[1]

#     if CONFIG["faiss_use_ivf"]:
#         quantizer = faiss.IndexFlatIP(dim)
#         index = faiss.IndexIVFFlat(quantizer, dim, CONFIG["faiss_nlist"], faiss.METRIC_INNER_PRODUCT)
#         index.train(embeddings)
#         index.add(embeddings)
#         logger.info(f"⚙️  Using FAISS IndexIVFFlat with {CONFIG['faiss_nlist']} clusters")
#     elif CONFIG["faiss_use_hnsw"]:
#         index = faiss.IndexHNSWFlat(dim, 32)
#         index.hnsw.efConstruction = 40
#         index.add(embeddings)
#         logger.info("⚙️  Using FAISS IndexHNSWFlat")
#     else:
#         index = faiss.IndexFlatIP(dim)
#         index.add(embeddings)
#         logger.info("⚙️  Using FAISS IndexFlatIP")

#     return index

def build_ivf_index(dim, train_embeddings):
    logger.info("🔍 Normalizing and training FAISS IVF index...")
    faiss.normalize_L2(train_embeddings)
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, CONFIG["faiss_nlist"], faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(train_embeddings)
    return faiss.IndexIDMap(index_ivf)

# def main():
#     logger.info(f"🖥️  Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

#     logger.info(f"📂 Loading Wikipedia TSV file from: {CONFIG['tsv_file']}")
#     data = load_wikipedia_tsv(CONFIG["tsv_file"])
#     logger.info(f"✅ Loaded {len(data)} passages")

#     texts = [f"passage: {entry['text']}" for entry in data]

#     logger.info("🧠 Encoding passages using SentenceTransformer.encode_multi_process()")
#     model = SentenceTransformer(CONFIG["model_name"])
#     pool = model.start_multi_process_pool(
#         target_devices=["cuda:0"]
#     )

#     embeddings = model.encode_multi_process(
#         texts,
#         pool,
#         batch_size=CONFIG["batch_size"],
#         chunk_size=CONFIG["chunk_size"],
#         show_progress_bar=True,
#         normalize_embeddings=True
#     )

#     model.stop_multi_process_pool(pool)
#     embeddings = np.array(embeddings)
#     logger.info(f"✅ Embeddings shape: {embeddings.shape}")

#     logger.info("📦 Building FAISS index...")
#     index = build_faiss_index(embeddings)
#     faiss.write_index(index, CONFIG["index_path"])
#     logger.info(f"💾 Saved FAISS index to {CONFIG['index_path']}")

#     logger.info("📝 Saving metadata...")
#     save_metadata(data, CONFIG["metadata_path"])
#     logger.info(f"💾 Saved metadata to {CONFIG['metadata_path']}")

#     logger.info("🎉 Done.")

def main():
    logger.info(f"🖥️  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model = SentenceTransformer(CONFIG["model_name"])
    pool = model.start_multi_process_pool(target_devices=["cuda:0"])

    all_metadata = []
    train_pool = []

    logger.info("📂 First pass: Collect training samples...")
    for chunk in tqdm(yield_wikipedia_chunks(CONFIG["tsv_file"], 100_000), desc="Sampling"):
        texts = [f"passage: {entry['text']}" for entry in chunk]
        emb = model.encode_multi_process(
            texts, pool, batch_size=CONFIG["batch_size"],
            chunk_size=CONFIG["chunk_size"], normalize_embeddings=True
        )
        train_pool.extend(emb)
        if len(train_pool) >= CONFIG["train_sample_size"]:
            break
    train_embeddings = np.array(train_pool[:CONFIG["train_sample_size"]])
    index = build_ivf_index(train_embeddings.shape[1], train_embeddings)

    logger.info("📦 Second pass: Encode & Add to FAISS...")
    idx_offset = 0
    with open(CONFIG["metadata_path"], "w", encoding="utf-8") as _f: pass  # clear old metadata

    for chunk in tqdm(yield_wikipedia_chunks(CONFIG["tsv_file"], CONFIG["batch_embed_size"]), desc="Indexing"):
        texts = [f"passage: {entry['text']}" for entry in chunk]
        ids = [int(entry["id"]) for entry in chunk]
        embeddings = model.encode_multi_process(
            texts, pool, batch_size=CONFIG["batch_size"],
            chunk_size=CONFIG["chunk_size"], normalize_embeddings=True
        )
        embeddings = np.array(embeddings)
        faiss.normalize_L2(embeddings)

        index.add_with_ids(embeddings, np.array(ids))
        save_metadata(chunk, CONFIG["metadata_path"])
        idx_offset += len(chunk)

    model.stop_multi_process_pool(pool)

    faiss.write_index(index, CONFIG["index_path"])
    logger.info(f"✅ Saved final FAISS index to {CONFIG['index_path']}")
    logger.info(f"✅ Saved metadata to {CONFIG['metadata_path']}")
    logger.info("🎉 All done!")


if __name__ == "__main__":
    main()
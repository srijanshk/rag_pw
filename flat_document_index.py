import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import faiss
from faiss import IndexHNSWFlat, METRIC_INNER_PRODUCT
import csv
import json
import logging
from tqdm import tqdm
import torch
import wandb  # added for logging
import math  # for computing total chunks

# Logging setup
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)
logger = logging.getLogger(__name__)

CONFIG = {
    "tsv_file": "downloads/data/wikipedia_split/psgs_w100.tsv",
    # Paths for both FlatIP and HNSW indexes
    "flatip_index_path": "/local00/student/shakya/wikipedia_flatip_index",
    "hnsw_index_path": "/local00/student/shakya/wikipedia_hnsw_index",
    "metadata_path": "/local00/student/shakya/wikipedia_metadata.jsonl",
    "model_name": "intfloat/e5-large-v2",
    "batch_size": 512,
    "chunk_size": 1_000_000,

    
}


def yield_wikipedia_chunks(path, chunk_size):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=["id", "text", "title"])
        next(reader)  # skip header
        chunk = []
        for row in reader:
            text = row["text"].strip()
            if text:
                chunk.append({
                    "id": int(row["id"]),
                    "title": row["title"],
                    "text": text
                })
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:
            yield chunk


def save_metadata(metadata: list, path: str):
    with open(path, "a", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    wandb.init(
        project="rag_indexing",
        name="build_flatip_hnsw",
        config=CONFIG
    )
    logger.info(f"🖥️  Using devices: cuda:0 and cuda:1")

    model = SentenceTransformer(CONFIG["model_name"])
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])

    dim = model.get_sentence_embedding_dimension()
    logger.info(f"📏 Embedding dimension: {dim}")
    flat_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    hnsw_index = faiss.IndexIDMap(IndexHNSWFlat(dim, 128, METRIC_INNER_PRODUCT))

    open(CONFIG["metadata_path"], "w", encoding="utf-8").close()

    total_lines = sum(1 for _ in open(CONFIG["tsv_file"], "r", encoding="utf-8")) - 1
    total_chunks = math.ceil(total_lines / CONFIG["chunk_size"])
    total_vectors = 0

    logger.info("📦 Embedding & indexing passages with FlatIP & HNSW...")
    for idx, chunk in enumerate(tqdm(
        yield_wikipedia_chunks(CONFIG["tsv_file"], CONFIG["chunk_size"]),
        desc="Indexing chunks", total=total_chunks, ncols=80
    )):
        embed_file = f"embeddings_chunk_{idx}.npy"
        if os.path.exists(embed_file):
            logger.info(f"⏩ Skipping chunk {idx}, found existing {embed_file}")
            embeddings = np.load(embed_file)
            ids = np.array([e['id'] for e in chunk])
        else:
            texts = [f"passage: {e['text']}" for e in chunk]
            ids = np.array([e['id'] for e in chunk])
            embeddings = model.encode_multi_process(
                texts,
                pool,
                batch_size=CONFIG["batch_size"],
                chunk_size=CONFIG["chunk_size"],
                normalize_embeddings=True
            )
            embeddings = np.array(embeddings).astype(np.float32)
            np.save(embed_file, embeddings)

        faiss.normalize_L2(embeddings)
        flat_index.add_with_ids(embeddings, ids)
        hnsw_index.add_with_ids(embeddings, ids)
        save_metadata(chunk, CONFIG["metadata_path"])

        total_vectors += len(ids)
        wandb.log({
            "chunk_index": idx,
            "vectors_indexed": len(ids),
            "total_vectors_indexed": total_vectors
        })

    model.stop_multi_process_pool(pool)
    faiss.write_index(flat_index, CONFIG["flatip_index_path"])
    faiss.write_index(hnsw_index, CONFIG["hnsw_index_path"])
    wandb.log({ "final_total_vectors": total_vectors })
    wandb.finish()

    logger.info(f"✅ Metadata saved to '{CONFIG['metadata_path']}'")
    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()

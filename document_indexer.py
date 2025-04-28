import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import faiss
import csv
import json
import logging
from tqdm import tqdm
import torch

# Logging setup
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO, handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

CONFIG = {
    "tsv_file": "downloads/data/wikipedia_split/psgs_w100.tsv",
    "index_path": "wikipedia_faiss_index",
    "metadata_path": "wikipedia_metadata.jsonl",
    "model_name": "intfloat/e5-large-v2",
    "batch_size": 512,
    "chunk_size": 1_000_000,
    "faiss_nlist": 4096,
    "train_sample_size": 500_000,
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


def save_metadata(metadata: list, path: str):
    with open(path, "a", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def build_ivf_index(dim, train_embeddings):
    logger.info("🔍 Training FAISS IVF index...")
    faiss.normalize_L2(train_embeddings)
    quantizer = faiss.IndexFlatIP(dim)
    index_ivf = faiss.IndexIVFFlat(quantizer, dim, CONFIG["faiss_nlist"], faiss.METRIC_INNER_PRODUCT)
    index_ivf.train(train_embeddings)
    return faiss.IndexIDMap(index_ivf)


def embed_texts(model, texts, batch_size):
    return model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, device="cuda")


def main():
    logger.info(f"🖥️  Using device: {torch.cuda.get_device_name(0)}")

    model = SentenceTransformer(CONFIG["model_name"]).to("cuda")

    logger.info("📂 First pass: Sampling training data...")
    train_pool = []
    for chunk in tqdm(yield_wikipedia_chunks(CONFIG["tsv_file"], 100_000), desc="Sampling"):
        texts = [f"passage: {entry['text']}" for entry in chunk]
        embeddings = embed_texts(model, texts, CONFIG["batch_size"])
        train_pool.extend(embeddings)
        if len(train_pool) >= CONFIG["train_sample_size"]:
            break

    train_embeddings = np.array(train_pool[:CONFIG["train_sample_size"]])
    index = build_ivf_index(train_embeddings.shape[1], train_embeddings)

    logger.info("📦 Second pass: Embedding and indexing passages...")
    with open(CONFIG["metadata_path"], "w", encoding="utf-8") as f:
        pass  # clear metadata file

    for chunk in tqdm(yield_wikipedia_chunks(CONFIG["tsv_file"], CONFIG["chunk_size"]), desc="Indexing"):
        texts = [f"passage: {entry['text']}" for entry in chunk]
        ids = [entry["id"] for entry in chunk]
        embeddings = embed_texts(model, texts, CONFIG["batch_size"])
        embeddings = np.array(embeddings)
        faiss.normalize_L2(embeddings)

        index.add_with_ids(embeddings, np.array(ids))
        save_metadata(chunk, CONFIG["metadata_path"])

    faiss.write_index(index, CONFIG["index_path"])
    logger.info(f"✅ Indexed {index.ntotal} passages. Saved FAISS index to {CONFIG['index_path']}")
    logger.info(f"✅ Metadata saved to {CONFIG['metadata_path']}")
    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()

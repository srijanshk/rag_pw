import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import logging
from typing import List
import numpy as np
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, LoggingHandler
import glob
import torch
import multiprocessing

# Configuration
CONFIG = {
    "chunk_file_list": "processed_chunks",
    "index_path": "wikipedia_faiss_index",
    "metadata_path": "merged_metadata.jsonl",
    "model_name": "intfloat/e5-base-v2",
    "batch_size": 64,
    "chunk_size": 512,
    "min_token_count": 50,
    "faiss_use_ivf": True,             # Enable IndexIVFFlat
    "faiss_use_hnsw": False,           # Alternatively, enable HNSW
    "faiss_nlist": 256                 # Number of clusters for IVF
}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)

def score_chunk(text):
    return round(np.log(len(text.split()) + 1), 4)

def find_all_jsonl_files(directory: str) -> List[str]:
    return glob.glob(os.path.join(directory, "**", "*.jsonl"), recursive=True)

def load_chunks(files: List[str]) -> List[dict]:
    all_chunks = []
    for file_path in files:
        logger.info(f"📂 Loading file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if data.get("token_count", 0) >= CONFIG["min_token_count"]:
                        data["score"] = score_chunk(data["text"])
                        all_chunks.append(data)
                except Exception as e:
                    logger.error(f"Error parsing line in {file_path}: {e}")
    return all_chunks

def save_metadata(metadata: List[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    if CONFIG["faiss_use_ivf"]:
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, CONFIG["faiss_nlist"], faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        logger.info(f"⚙️  Using FAISS IndexIVFFlat with {CONFIG['faiss_nlist']} clusters")
    elif CONFIG["faiss_use_hnsw"]:
        index = faiss.IndexHNSWFlat(dim, 32)
        index.hnsw.efConstruction = 40
        index.add(embeddings)
        logger.info("⚙️  Using FAISS IndexHNSWFlat")
    else:
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        logger.info("⚙️  Using FAISS IndexFlatIP")

    return index

def main():
    logger.info(f"🖥️  Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    logger.info("🔹 Searching for JSONL files...")
    jsonl_files = find_all_jsonl_files(CONFIG["chunk_file_list"])
    if not jsonl_files:
        logger.error("❌ No .jsonl files found. Exiting.")
        return

    logger.info("🔹 Loading chunked JSONL files...")
    chunks = load_chunks(jsonl_files)
    texts = [c["text"] for c in chunks]
    logger.info(f"✅ Loaded {len(texts)} valid chunks")

    logger.info("🔹 Encoding using encode_multi_process (GPU-supported)")
    model = SentenceTransformer(CONFIG["model_name"])
    pool = model.start_multi_process_pool()

    embeddings = model.encode_multi_process(
        texts,
        pool,
        batch_size=CONFIG["batch_size"],
        chunk_size=CONFIG["chunk_size"],
        show_progress_bar=True,
        normalize_embeddings=True
    )

    model.stop_multi_process_pool(pool)
    embeddings = np.array(embeddings)
    logger.info(f"✅ Embeddings shape: {embeddings.shape}")

    logger.info("🔹 Building FAISS index...")
    index = build_faiss_index(embeddings)
    faiss.write_index(index, CONFIG["index_path"])
    logger.info(f"💾 Saved FAISS index to {CONFIG['index_path']}")

    logger.info("🔹 Saving metadata...")
    save_metadata(chunks, CONFIG["metadata_path"])
    logger.info(f"💾 Saved metadata to {CONFIG['metadata_path']}")

    logger.info("✅ Done.")

if __name__ == "__main__":
    main()

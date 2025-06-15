import os
import torch
import faiss
import csv
import json
import numpy as np
import logging
import math
import sys
from tqdm import tqdm
from faiss import IndexHNSWFlat, METRIC_INNER_PRODUCT
from FlagEmbedding import BGEM3FlagModel
import wandb
import pickle
from pathlib import Path

# --- Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(1024 * 1024 * 1024)

# --- Configuration ---
CONFIG = {
    "tsv_file": "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing_bge-m3.tsv",
    "hnsw_index_path": "/local00/student/shakya/openmath_bge-m3_hnsw_index",
    "metadata_path": "/local00/student/shakya/openmath_bge-m3_metadata.jsonl",
    "model_name": "BAAI/bge-m3",
    "batch_size": 256,
    "chunk_size": 500_000,
    "max_length": 2048,
    "hnsw_m": 64,  # Increased M for potentially better recall
    "ef_construction": 200,
}

# --- Helper Functions ---
def yield_openmath_chunks(path, chunk_size):
    """Generator that yields chunks of documents with unique IDs."""
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        chunk = []
        current_id = 0
        
        for row in reader:
            unique_id = int(row['row_id']) * 1000 + int(row['chunk_id'])
            doc = {
                'id': unique_id,
                'row_id': row['row_id'],
                'chunk_id': row['chunk_id'],
                'problem': row['problem'],
                'solution_chunk': row['solution_chunk'],
                'expected_answer': row['expected_answer'],
                'problem_from': row['problem_from']
            }
            chunk.append(doc)
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
                
        if chunk:
            yield chunk

def save_metadata(metadata_chunk: list, path: str):
    """Appends metadata to a JSONL file."""
    with open(path, 'a', encoding='utf-8') as f:
        for doc in metadata_chunk:
            # We are saving the whole row dict, which now includes the integer 'id'
            f.write(json.dumps(doc) + '\n')

def save_colbert_embeddings(colbert_vecs, ids, base_dir, chunk_idx):
    """Saves ColBERT embeddings to a structured directory."""
    chunk_dir = os.path.join(base_dir, f"chunk_{chunk_idx}")
    os.makedirs(chunk_dir, exist_ok=True)
    for vec, doc_id in zip(colbert_vecs, ids):
        np.save(os.path.join(chunk_dir, f"{doc_id}.npy"), vec)

# --- Main Indexing Logic ---
def main():
    wandb.init(project="RAG-Thesis-Indexing", name="bge-m3-full-multi-embed-final", config=CONFIG)
    logger.info("Starting Full RAG Indexing Pipeline with BAAI/bge-m3.")
    
    # Clean up old files for a fresh start
    for path in [CONFIG["hnsw_index_path"], CONFIG["metadata_path"]]:
        if os.path.exists(path):
            logger.warning(f"Removing existing file: {path}")
            if os.path.isdir(path):
                import shutil; shutil.rmtree(path)
            else:
                os.remove(path)
    
    Path(os.path.dirname(CONFIG["hnsw_index_path"])).mkdir(parents=True, exist_ok=True)
    
    # Load BGE-M3 model
    logger.info(f"Loading embedding model: {CONFIG['model_name']}")
    model = BGEM3FlagModel(CONFIG['model_name'], use_fp16=True)
    dim = model.model.config.hidden_size
    
    # Initialize Indexes
    quantizer = faiss.IndexHNSWFlat(dim, CONFIG["hnsw_m"], METRIC_INNER_PRODUCT)
    quantizer.hnsw.efConstruction = CONFIG["ef_construction"]
    hnsw_index = faiss.IndexIDMap(quantizer)
    sparse_index = {} # In-memory dictionary for sparse vectors

    try:
        with open(CONFIG["tsv_file"], 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f) - 1
        total_chunks_to_process = math.ceil(total_lines / CONFIG["chunk_size"]) if CONFIG["chunk_size"] > 0 else 1
    except FileNotFoundError:
        logger.error(f"Input TSV file not found. Aborting."); wandb.finish(exit_code=1); return

    logger.info(f"Processing {total_lines:,} documents in {total_chunks_to_process} chunks...")

    for chunk_idx, doc_chunk in enumerate(tqdm(yield_openmath_chunks(CONFIG["tsv_file"], CONFIG["chunk_size"]), total=total_chunks_to_process, desc="Processing Chunks")):
        logger.info(f"--- Processing Chunk {chunk_idx + 1}/{total_chunks_to_process} ---")
        texts_to_encode = [f"Question: {doc.get('problem', '')}\nAnswer Chunk: {doc.get('solution_chunk', '')}" for doc in doc_chunk]
        ids_np = np.array([doc['id'] for doc in doc_chunk], dtype=np.int64)

        embedding_dict = model.encode(
            texts_to_encode, batch_size=CONFIG["batch_size"], max_length=CONFIG["max_length"],
            return_dense=True, return_sparse=False, return_colbert_vecs=False
        )
        
        # 1. Process dense embeddings
        dense_embeddings = embedding_dict['dense_vecs']
        dense_embeddings_float32 = dense_embeddings.astype(np.float32)
        faiss.normalize_L2(dense_embeddings_float32)
        hnsw_index.add_with_ids(dense_embeddings_float32, ids_np)
        logger.info(f"Chunk {chunk_idx + 1}: Done with dense embeddings. Added to Faiss index.")
        
        # Save metadata and log progress
        save_metadata(doc_chunk, CONFIG["metadata_path"])
        wandb.log({"chunk_processed": chunk_idx + 1, "total_vectors_indexed": hnsw_index.ntotal})

    # Save final indexes
    logger.info(f"Saving HNSW index with {hnsw_index.ntotal} vectors...")
    faiss.write_index(hnsw_index, CONFIG["hnsw_index_path"])
    
    logger.info("✅ Indexing complete.")
    wandb.summary["final_total_vectors_indexed"] = hnsw_index.ntotal
    wandb.finish()

if __name__ == "__main__":
    main()
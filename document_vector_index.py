import os
# This line is critical: it makes physical GPUs 2 and 3 visible to PyTorch as cuda:0 and cuda:1.
# It should ideally be set before any CUDA-related libraries (like torch) are imported.
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"

from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import faiss
from faiss import IndexHNSWFlat, METRIC_INNER_PRODUCT # Using CPU Faiss indexes
import csv
import json
import logging
from tqdm import tqdm
import torch
import wandb  # for logging
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
    "flatip_index_path": "/local00/student/shakya/wikipedia_flatip_index",
    "hnsw_index_path": "/local00/student/shakya/wikipedia_hnsw_index",
    "metadata_path": "/local00/student/shakya/wikipedia_metadata.jsonl",
    "embeddings_save_path_template": "/local00/student/shakya/wikipedia_embeddings/chunk_{idx}.npy",
    "ids_save_path_template": "/local00/student/shakya/wikipedia_embeddings/ids_chunk_{idx}.npy",
    "save_embeddings_and_ids": True,
    "model_name": "models/retriever_finetuned_e5_best",
    "batch_size": 256,  # Per GPU for encoding. Adjust based on P100 16GB VRAM if OOM.
    "chunk_size": 1_000_000, # Number of passages to process in one go (affects CPU RAM)
    "encode_internal_chunk_size": 10000 # Internal chunk_size for model.encode_multi_process
}


def yield_wikipedia_chunks(path, chunk_size):
    """Yields chunks of documents from the TSV file."""
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t", fieldnames=["id", "text", "title"])
        try:
            next(reader)  # skip header row
        except StopIteration:
            logger.warning(f"TSV file '{path}' might be empty or only contain a header.")
            return # Stop iteration if no data rows
        
        chunk = []
        for i, row in enumerate(reader):
            text = row.get("text", "").strip()
            doc_id_str = row.get("id")
            title = row.get("title", "")

            if not doc_id_str:
                logger.warning(f"Skipping row {i+2} due to missing 'id': {row}") # +2 for header and 0-indexing
                continue
            try:
                doc_id = int(doc_id_str)
            except ValueError:
                logger.warning(f"Skipping row {i+2} due to non-integer 'id' ('{doc_id_str}'): {row}")
                continue

            if text:  # Only include if text is not empty
                chunk.append({
                    "id": doc_id,
                    "title": title,
                    "text": text
                })
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        if chunk:  # Yield the last remaining chunk
            yield chunk


def save_metadata(metadata: list, path: str):
    """Appends metadata entries to a JSONL file."""
    with open(path, "a", encoding="utf-8") as f:
        for entry in metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    wandb.init(
        project="rag_indexing",
        name="build_faiss_flatip_hnsw_2xp100_encoding", # Descriptive name
        config=CONFIG
    )

    logger.info(f"Script initiated with CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}'")

    # Verify GPU availability as seen by PyTorch
    num_gpus_torch = torch.cuda.device_count()
    logger.info(f"PyTorch detects {num_gpus_torch} available CUDA device(s).")

    if num_gpus_torch == 0:
        logger.error("No GPUs detected by PyTorch. Ensure CUDA drivers and PyTorch with CUDA support are correctly installed and CUDA_VISIBLE_DEVICES is set if needed.")
        wandb.finish(exit_code=1)
        return
    
    # Define target devices for sentence_transformers based on PyTorch's view.
    # If CUDA_VISIBLE_DEVICES="2,3", PyTorch maps them to "cuda:0" and "cuda:1".
    # We aim to use up to 2 available GPUs.
    num_gpus_to_use = min(num_gpus_torch, 4)
    target_devices_for_pool = [f"cuda:{i}" for i in range(num_gpus_to_use)]

    # if num_gpus_to_use < 2:
    #     logger.warning(f"Only {num_gpus_to_use} GPU(s) will be used for encoding ({target_devices_for_pool}). Expected 2.")
    # else:
    #     logger.info(f"SentenceTransformer will use target devices: {target_devices_for_pool} for encoding (mapping to physical GPUs 2 & 3).")

    model = SentenceTransformer(CONFIG["model_name"])
    
    # Start multi-process pool for encoding on the specified GPUs
    pool = model.start_multi_process_pool(target_devices=target_devices_for_pool)

    dim = model.get_sentence_embedding_dimension()
    logger.info(f"Embedding dimension for model '{CONFIG['model_name']}': {dim}")

    # Initialize CPU-based Faiss indexes
    logger.info("Initializing CPU-based Faiss indexes (IndexFlatIP and IndexHNSWFlat).")
    flat_index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    # For HNSW, M is the number of bi-directional links created for every new element during construction.
    # Higher M means better recall but more memory/build time. 128 is a strong value.
    # METRIC_INNER_PRODUCT is used because e5 embeddings are typically normalized and compared with dot product.
    hnsw_index = faiss.IndexIDMap(IndexHNSWFlat(dim, 128, METRIC_INNER_PRODUCT))
    # HNSWFlat does not require a separate training step.

    # Ensure metadata file is cleared at the start of a new indexing run
    if os.path.exists(CONFIG["metadata_path"]):
        logger.info(f"Clearing existing metadata file: '{CONFIG['metadata_path']}'")
        open(CONFIG["metadata_path"], "w").close()


    try:
        # Count total lines for progress bar, excluding header
        with open(CONFIG["tsv_file"], "r", encoding="utf-8") as f_count:
            total_lines = sum(1 for _ in f_count) -1 
        if total_lines < 0: total_lines = 0 # handles empty file or file with only header
    except FileNotFoundError:
        logger.error(f"Input TSV file not found: '{CONFIG['tsv_file']}'. Exiting.")
        wandb.finish(exit_code=1)
        return
    
    total_chunks = math.ceil(total_lines / CONFIG["chunk_size"]) if CONFIG["chunk_size"] > 0 and total_lines > 0 else (1 if total_lines > 0 else 0)
    logger.info(f"Estimated total passages (excluding header): {total_lines}, to be processed in {total_chunks} chunks of size {CONFIG['chunk_size']}.")

    total_vectors_processed = 0

    logger.info(f"Starting embedding generation using {target_devices_for_pool} and CPU-based Faiss indexing...")
    for chunk_idx, doc_chunk in enumerate(tqdm(
        yield_wikipedia_chunks(CONFIG["tsv_file"], CONFIG["chunk_size"]),
        desc="Processing chunks", total=total_chunks, ncols=100
    )):
        if not doc_chunk: # Should not happen if yield_wikipedia_chunks is correct, but good check
            logger.warning(f"Skipping empty document chunk at index {chunk_idx}.")
            continue

        logger.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} with {len(doc_chunk)} passages.")

        # Prepare texts and IDs for the current chunk
        texts_to_encode = [f"passage: {doc['text']}" for doc in doc_chunk] # e5 models expect "passage: " or "query: "
        ids_np = np.array([doc['id'] for doc in doc_chunk], dtype=np.int64) # Faiss requires int64 for IDs

        # Encode passages using the multi-GPU process pool
        embeddings = model.encode_multi_process(
            texts_to_encode,
            pool=pool,
            batch_size=CONFIG["batch_size"], # Batch size per process/GPU
            chunk_size=CONFIG["encode_internal_chunk_size"], # Internal chunking for encode_multi_process for memory efficiency
            normalize_embeddings=True # Crucial for e5 models and METRIC_INNER_PRODUCT
        )
        # Convert to float32 numpy array, which Faiss prefers
        embeddings_np = np.array(embeddings, dtype=np.float32)

        if CONFIG["save_embeddings_and_ids"]:
            # Create directory if it doesn't exist
            embeddings_dir = os.path.dirname(CONFIG["embeddings_save_path_template"])
            if embeddings_dir: # Ensure there's a directory part in the template
                 os.makedirs(embeddings_dir, exist_ok=True)

            chunk_embedding_path = CONFIG["embeddings_save_path_template"].format(idx=chunk_idx)
            chunk_ids_path = CONFIG["ids_save_path_template"].format(idx=chunk_idx)
            
            logger.info(f"Saving embeddings for chunk {chunk_idx + 1} to '{chunk_embedding_path}'")
            np.save(chunk_embedding_path, embeddings_np)
            
            logger.info(f"Saving IDs for chunk {chunk_idx + 1} to '{chunk_ids_path}'")
            np.save(chunk_ids_path, ids_np) # ids_np is already created in your loop


        # Add embeddings and their IDs to the CPU Faiss indexes
        # Normalization is already handled by `normalize_embeddings=True` in encode_multi_process
        flat_index.add_with_ids(embeddings_np, ids_np)
        hnsw_index.add_with_ids(embeddings_np, ids_np) # This step can be time-consuming for HNSW on CPU

        # Save metadata for the current chunk
        save_metadata(doc_chunk, CONFIG["metadata_path"])

        total_vectors_processed += len(ids_np)
        wandb.log({
            "chunk_index_processed": chunk_idx,
            "vectors_in_chunk": len(ids_np),
            "total_vectors_indexed": total_vectors_processed
        })
        logger.info(f"Chunk {chunk_idx + 1} indexed. Total vectors so far: {total_vectors_processed}.")

    logger.info("All chunks processed. Stopping multi-process pool...")
    model.stop_multi_process_pool(pool)

    logger.info(f"Writing FlatIP index with {flat_index.ntotal} vectors to '{CONFIG['flatip_index_path']}'...")
    faiss.write_index(flat_index, CONFIG["flatip_index_path"])
    logger.info("FlatIP index saved.")

    logger.info(f"Writing HNSW index with {hnsw_index.ntotal} vectors to '{CONFIG['hnsw_index_path']}'...")
    faiss.write_index(hnsw_index, CONFIG["hnsw_index_path"])
    logger.info("HNSW index saved.")
    
    wandb.log({ "final_total_vectors_indexed": total_vectors_processed })
    wandb.finish()

    logger.info(f"Metadata for all passages saved to '{CONFIG['metadata_path']}'")
    logger.info(f"🎉 Indexing complete. A total of {total_vectors_processed} vectors were indexed.")


if __name__ == "__main__":
    main()
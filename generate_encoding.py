# generate_encoding.py

import os
import json
import logging
from typing import List
import numpy as np
import faiss
import torch
import multiprocessing as mp
from transformers import AutoTokenizer, AutoModel

# Configuration
CONFIG = {
    "input_dir": "wiki_extracted",
    "output_dir": "processed_chunks",
    "index_path": "wikipedia_faiss_index",
    "xapian_db_path": "wikipedia_xapian_db",
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "max_token_length": 384,
    "target_chunk_tokens": 384,
    "min_chunk_tokens": 0,
    "overlap_sentences": 2,
    "batch_size": 16,
    "faiss_metric": 2,
    "metadata_path": "merged_metadata.jsonl",
    "num_processes": 10,
    "partial_embeddings_dir": "partial_embeddings",
}

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_worker(proc_id: int, files_subset: List[str], config: dict) -> int:
    logger.info(f"Process {proc_id}: Starting worker with {len(files_subset)} file(s).")
    
    device = torch.device("cpu")
    logger.info(f"Process {proc_id}: Using device {device}.")
    
    # Load model and tokenizer.
    model = AutoModel.from_pretrained(config["model_name"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    logger.info(f"Process {proc_id}: Model and tokenizer loaded.")
    
    # Limit CPU threads to help avoid contention issues.
    torch.set_num_threads(1)
    
    embeddings_list = []
    metadata_list = []

    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_batch(batch_texts: List[str]):
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=config["max_token_length"],
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            model_output = model(**inputs)
        embeddings = mean_pooling(model_output, inputs["attention_mask"])
        return embeddings.cpu().numpy()

    # Process each file in the subset.
    for ind,file_path in enumerate(files_subset):
        logger.info(f"Process {proc_id}: Processing file {ind+1} of {len(files_subset)}")
        batch_texts = []
        temp_metadata = []
        with open(file_path, "r") as f:
            for i, line in enumerate(f):
                try:
                    data = json.loads(line)
                except Exception as e:
                    logger.error(f"Process {proc_id}: Error parsing line {i} in {file_path}: {e}")
                    continue

                
                if data.get("token_count", 0) < config["min_chunk_tokens"]:
                    continue

                temp_metadata.append(data)
                batch_texts.append(data.get("text", ""))

                # When batch is full, embed and clear the accumulators.
                if len(batch_texts) >= config["batch_size"]:
                    outputs = embed_batch(batch_texts)
                    embeddings_list.append(outputs)
                    metadata_list.extend(temp_metadata)
                    batch_texts = []
                    temp_metadata = []
            # Process any remaining texts.
            if batch_texts:
                outputs = embed_batch(batch_texts)
                embeddings_list.append(outputs)
                metadata_list.extend(temp_metadata)

    logger.info(f"Process {proc_id}: Finished embedding. Total batches: {len(embeddings_list)}")
    if embeddings_list:
        worker_embeddings = np.concatenate(embeddings_list, axis=0)
    else:
        worker_embeddings = np.array([]).reshape(0, model.config.hidden_size)

    os.makedirs(config["partial_embeddings_dir"], exist_ok=True)
    partial_embeddings_path = os.path.join(config["partial_embeddings_dir"], f"embeddings_{proc_id}.npy")
    np.save(partial_embeddings_path, worker_embeddings)
    partial_metadata_path = os.path.join(config["partial_embeddings_dir"], f"metadata_{proc_id}.jsonl")
    with open(partial_metadata_path, "w") as meta_f:
        for m in metadata_list:
            meta_f.write(json.dumps(m) + "\n")

    logger.info(f"Process {proc_id}: Finished processing. {worker_embeddings.shape[0]} embeddings saved.")
    return proc_id


def parallel_embedding_main(chunk_files: List[str], config: dict):
    """
    Splits chunk_files among worker processes, then merges the resulting embeddings and metadata.
    """
    n_processes = config["num_processes"]
    chunk_size = len(chunk_files) // n_processes
    file_sublists = []
    start_idx = 0
    for i in range(n_processes):
        end_idx = start_idx + chunk_size
        if i == n_processes - 1:  # Last process takes the remainder.
            end_idx = len(chunk_files)
        file_sublists.append(chunk_files[start_idx:end_idx])
        start_idx = end_idx

    os.makedirs(config["partial_embeddings_dir"], exist_ok=True)
    
    with mp.Pool(processes=n_processes) as pool:
        results = [
            pool.apply_async(embed_worker, args=(i, file_sublists[i], config))
            for i in range(n_processes)
        ]
        for r in results:
            r.get()

    all_embeddings = []
    all_metadata = []
    for i in range(n_processes):
        partial_embeddings_path = os.path.join(config["partial_embeddings_dir"], f"embeddings_{i}.npy")
        partial_metadata_path = os.path.join(config["partial_embeddings_dir"], f"metadata_{i}.jsonl")
        if os.path.exists(partial_embeddings_path):
            emb = np.load(partial_embeddings_path)
            if emb.shape[0] > 0:
                all_embeddings.append(emb)
        if os.path.exists(partial_metadata_path):
            with open(partial_metadata_path, "r") as fm:
                for line in fm:
                    try:
                        data = json.loads(line)
                        all_metadata.append(data)
                    except Exception as e:
                        logger.error(f"Error parsing merged metadata: {e}")

    if all_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
    else:
        all_embeddings = np.array([]).reshape(0, 768)

    logger.info(f"Total embeddings shape: {all_embeddings.shape}")
    logger.info(f"Total metadata count: {len(all_metadata)}")

    if all_embeddings.shape[0] > 0:
        dimension = all_embeddings.shape[1]
        faiss.normalize_L2(all_embeddings)
        index = faiss.IndexFlatIP(dimension)
        index.add(all_embeddings)
        faiss.write_index(index, config["index_path"])
        logger.info(f"FAISS index with {index.ntotal} vectors saved to: {config['index_path']}")

        with open(config["metadata_path"], "w") as f:
            for item in all_metadata:
                f.write(json.dumps(item) + "\n")
        logger.info(f"Merged metadata written to: {config['metadata_path']}")
    else:
        logger.info("No embeddings found. The index was not created.")


def main():
    # Collect JSON/JSONL files from the output directory.
    chunk_files = []
    for root, dirs, files in os.walk(CONFIG["output_dir"]):
        for file in files:
            if file == ".DS_Store":
                continue
            if file == ".DS_Store.json" or file == ".DS_Store.jsonl":
                continue
            if file.endswith(".json") or file.endswith(".jsonl"):
                chunk_files.append(os.path.join(root, file))
    if not chunk_files:
        print(f"No JSON/JSONL files found in {CONFIG['output_dir']}. Exiting.")
        return

    parallel_embedding_main(chunk_files, CONFIG)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()

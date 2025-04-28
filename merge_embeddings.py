#!/usr/bin/env python3
"""
merge_embeddings.py

This script merges partial embeddings (.npy files) and metadata (.jsonl files)
stored in a designated directory. It then normalizes the embeddings, creates a FAISS index,
and writes the merged metadata to disk.
"""

import os
import json
import logging
import numpy as np
import faiss

# Configuration
CONFIG = {
    "partial_embeddings_dir": "partial_embeddings",
    "index_path": "wikipedia_faiss_index",
    "metadata_path": "merged_metadata.jsonl",
    # Expected hidden dimension of the embeddings.
    "hidden_size": 768,
    "faiss_metric": faiss.METRIC_INNER_PRODUCT,
}

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_partial_embeddings(config: dict):
    """
    Merge all partial embeddings and metadata files.
    Looks for files named embeddings_*.npy and metadata_*.jsonl in the specified directory.
    """
    partial_dir = config["partial_embeddings_dir"]
    embeddings_list = []
    metadata_list = []

    if not os.path.isdir(partial_dir):
        logger.error(f"Partial embeddings directory '{partial_dir}' does not exist.")
        return np.array([]).reshape(0, config["hidden_size"]), metadata_list

    # Sort files for consistent ordering.
    for filename in sorted(os.listdir(partial_dir)):
        file_path = os.path.join(partial_dir, filename)
        if filename.startswith("embeddings_") and filename.endswith(".npy"):
            try:
                emb = np.load(file_path)
                if emb.shape[0] > 0:
                    embeddings_list.append(emb)
                    logger.info(f"Loaded {emb.shape[0]} embeddings from {filename}")
                else:
                    logger.info(f"No embeddings found in {filename}")
            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
        elif filename.startswith("metadata_") and filename.endswith(".jsonl"):
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            metadata_list.append(data)
                        except Exception as parse_err:
                            logger.error(f"Error parsing a line in {filename}: {parse_err}")
                logger.info(f"Loaded metadata from {filename}")
            except Exception as e:
                logger.error(f"Error reading {filename}: {e}")

    if embeddings_list:
        all_embeddings = np.concatenate(embeddings_list, axis=0)
    else:
        all_embeddings = np.array([]).reshape(0, config["hidden_size"])
    
    return all_embeddings, metadata_list


def create_faiss_index(embeddings: np.ndarray, config: dict):
    """
    Normalize embeddings, create a FAISS index, and save it to disk.
    """
    if embeddings.shape[0] == 0:
        logger.warning("No embeddings to index.")
        return

    dimension = embeddings.shape[1]
    # Normalize embeddings to unit length for inner product search.
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, config["index_path"])
    logger.info(f"FAISS index with {index.ntotal} vectors saved to {config['index_path']}")


def save_merged_metadata(metadata_list: list, config: dict):
    """
    Save the merged metadata to a JSONL file.
    """
    if not metadata_list:
        logger.warning("No metadata to save.")
        return

    with open(config["metadata_path"], "w") as f:
        for item in metadata_list:
            f.write(json.dumps(item) + "\n")
    logger.info(f"Merged metadata written to {config['metadata_path']}")


def main():
    logger.info("Starting merge of partial embeddings and metadata...")
    all_embeddings, all_metadata = merge_partial_embeddings(CONFIG)
    logger.info(f"Total merged embeddings shape: {all_embeddings.shape}")
    logger.info(f"Total merged metadata count: {len(all_metadata)}")
    
    create_faiss_index(all_embeddings, CONFIG)
    save_merged_metadata(all_metadata, CONFIG)
    logger.info("Merging complete.")


if __name__ == "__main__":
    main()

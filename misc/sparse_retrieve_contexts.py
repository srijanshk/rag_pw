import os
import json
import logging
import hashlib
from typing import Any, Dict, List # Keep for ID hashing if needed, though Xapian doc IDs are usually integers
import xapian
from multiprocessing import Process, Pool, cpu_count, Manager # Ensure Manager is imported
from tqdm import tqdm

from xapian_retriever import XapianRetriever

# --- Configuration ---
CONFIG = {
    "nq_train_file": "downloads/data/gold_passages_info/nq_train.json",
    "nq_dev_file": "downloads/data/gold_passages_info/nq_dev.json",
    "xapian_db_path": "/local00/student/shakya/wikipedia_xapian_db",
    "output_sparse_train_file": "downloads/data/nq_train_sparse_retrieval.jsonl",
    "output_sparse_dev_file": "downloads/data/nq_dev_sparse_retrieval.jsonl",
    "k_sparse": 50,
    "num_workers": max(1, cpu_count() - 1)
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
)
logger = logging.getLogger(__name__)

def load_local_nq_json(file_path: str, limit: int = None) -> List[Dict[str, Any]]:
    logger.info(f"Loading local NQ data from: {file_path}")
    data_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, dict) and "data" in content and isinstance(content["data"], list):
                data_list = content["data"]
            elif isinstance(content, list):
                logger.warning("JSON file appears to be a root list, not a dict with a 'data' key.")
                data_list = content
            else:
                raise ValueError(f"JSON file {file_path} does not contain a 'data' key with a list of entries, or is not a root list of entries.")
        
        total_loaded = len(data_list)
        if limit is not None and limit > 0 and total_loaded > limit:
            logger.info(f"Limiting dataset from {total_loaded} to first {limit} entries.")
            return data_list[:limit]
        logger.info(f"Loaded {total_loaded} entries from {file_path}.")
        return data_list
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {file_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Error: File not found at {file_path}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading {file_path}: {e}")
        raise

# --- Worker process for Xapian search ---
worker_xapian_retriever = None
worker_output_queue = None

def worker_init(q, db_path, k_sparse_val, bm25_k1_val, bm25_b_val, use_bm25_val):
    global worker_xapian_retriever, worker_output_queue

    worker_xapian_retriever = XapianRetriever(
        db_path=db_path, 
        use_bm25=use_bm25_val,
        bm25_k1=bm25_k1_val,
        bm25_b=bm25_b_val
    )
    worker_output_queue = q
    # logger.info(f"Worker {os.getpid()} initialized XapianRetriever.")

def process_nq_entry_for_sparse_retrieval(nq_entry_tuple):
    """
    Worker function to process a single NQ entry, perform Xapian search,
    and put the result on the queue.
    """
    global worker_xapian_retriever, worker_output_queue
    
    entry_idx, nq_entry = nq_entry_tuple # Unpack if you pass index for some reason

    question_text = nq_entry.get("question")
    if not question_text:
        return # Skip if no question

    try:
        # Perform Xapian search using the worker's retriever instance
        sparse_results = worker_xapian_retriever.search(question_text, k=CONFIG["k_sparse"])

        processed_sparse_docs = []
        for metadata_dict, sparse_score in sparse_results:
            processed_sparse_docs.append({
                "id": metadata_dict.get("id"), # Ensure this ID matches your FAISS/main metadata IDs
                "title": metadata_dict.get("title"),
                "text": metadata_dict.get("text"), 
                "sparse_score": sparse_score
            })
        
        output_entry = {
            "original_question": question_text,
            "original_answers": nq_entry.get("short_answers", []), 
            "example_id": nq_entry.get("example_id", f"gen_id_{entry_idx}"), 
            "sparse_retrieved_docs": processed_sparse_docs
        }
        worker_output_queue.put(output_entry)
    except Exception as e:
        logger.warning(f"Worker failed to process entry for question '{question_text[:50]}...': {e}")
    return True # Indicate processing attempt


# --- Writer process for JSONL output ---
def jsonl_writer_process(queue, output_jsonl_file_path):
    """
    Dedicated process for writing results from the queue to a JSONL file.
    """
    logger.info(f"Writer process started for {output_jsonl_file_path}.")
    count = 0
    with open(output_jsonl_file_path, 'w', encoding='utf-8') as outfile:
        while True:
            item = queue.get()
            if item is None: # Sentinel value to stop
                break
            try:
                outfile.write(json.dumps(item) + "\n")
                count += 1
                if count % 10000 == 0: # Log progress every 10k writes
                    logger.info(f"Writer (for {os.path.basename(output_jsonl_file_path)}) wrote {count} entries...")
            except Exception as e:
                logger.warning(f"Writer failed to write one entry: {e}")
    logger.info(f"Writer process finished for {output_jsonl_file_path}. Wrote {count} entries.")


def generate_sparse_retrieval_parallel(input_nq_file, output_jsonl_file):
    nq_data = load_local_nq_json(input_nq_file) # Load all questions in main process
    if not nq_data:
        logger.warning(f"No data loaded from {input_nq_file}. Skipping.")
        return

    manager = Manager()
    # Queue for results from workers to writer
    output_queue = manager.Queue(maxsize=CONFIG["num_workers"] * 20) # Adjust maxsize as needed

    # Start the dedicated writer process
    writer = Process(target=jsonl_writer_process, args=(output_queue, output_jsonl_file))
    writer.start()

    # Initialize XapianRetriever once to get its parameters for workers
    # (Workers will create their own instances but can use these params)
    temp_retriever_for_params = XapianRetriever(db_path=CONFIG["xapian_db_path"])

    # Create a pool of worker processes
    # Each worker will initialize its own XapianRetriever
    pool = Pool(
        processes=CONFIG["num_workers"], 
        initializer=worker_init, 
        initargs=(
            output_queue, 
            CONFIG["xapian_db_path"], 
            CONFIG["k_sparse"],
            temp_retriever_for_params.bm25_k1, # Pass BM25 params
            temp_retriever_for_params.bm25_b,
            temp_retriever_for_params.use_bm25
            )
    )
    
    # Add index to nq_data items for potential use in worker (e.g., for generated example_id)
    nq_data_with_indices = list(enumerate(nq_data))

    logger.info(f"Distributing {len(nq_data_with_indices)} NQ entries to {CONFIG['num_workers']} worker processes...")
    try:
        # Use imap_unordered for progress bar and potentially better load balancing
        for _ in tqdm(pool.imap_unordered(process_nq_entry_for_sparse_retrieval, nq_data_with_indices), 
                      total=len(nq_data_with_indices), 
                      desc=f"Processing {os.path.basename(input_nq_file)}"):
            pass
    finally:
        pool.close() # Prevent new tasks from being submitted
        pool.join()  # Wait for all worker processes to finish

    logger.info("All worker processes finished.")
    output_queue.put(None) # Send sentinel to writer process to stop
    writer.join()          # Wait for writer process to finish
    logger.info(f"Sparse retrieval generation for {input_nq_file} complete.")


def main_sparse_generator():
    logger.info("Starting Offline Sparse Retrieval Data Generation...")
    
    # Ensure output directories exist
    os.makedirs(os.path.dirname(CONFIG["output_sparse_train_file"]), exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["output_sparse_dev_file"]), exist_ok=True)

    # Generate for training set
    logger.info(f"--- Generating for Training Set: {CONFIG['nq_train_file']} ---")
    generate_sparse_retrieval_parallel(CONFIG["nq_train_file"], CONFIG["output_sparse_train_file"])
    
    # Generate for dev/test set
    logger.info(f"--- Generating for Dev Set: {CONFIG['nq_dev_file']} ---")
    generate_sparse_retrieval_parallel(CONFIG["nq_dev_file"], CONFIG["output_sparse_dev_file"])
    
    logger.info("Offline Sparse Retrieval Data Generation completed.")

if __name__ == "__main__":
    main_sparse_generator()
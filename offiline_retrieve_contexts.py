import json
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import logging

from xapian_retriever import XapianRetriever

INPUT_DATA_FILES = {
    "train": "downloads/data/gold_passages_info/nq_train.json",
    "dev": "downloads/data/gold_passages_info/nq_dev.json"
}
OUTPUT_DATA_FILES_WITH_CONTEXTS = {
    "train": "downloads/data/xapian/nq-train_xapian_contexts.jsonl",
    "dev": "downloads/data/xapian/nq-dev_xapian_contexts.jsonl"
}
XAPIAN_DB_PATH = "/local00/student/shakya/wikipedia_xapian_db"

# BM25 parameters for retrieval
BM25_K1 = 1.2
BM25_B  = 0.75

K_RETRIEVAL_CONFIG = {
    "train": 10, 
    "dev": 50  
}
NUM_WORKERS = max(1, cpu_count() // 2)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] (%(processName)s) %(message)s',
                    handlers=[logging.StreamHandler()]) # Add FileHandler if you want to log to file
logger = logging.getLogger(__name__)

worker_retriever_instance = None

def initialize_worker_retriever(db_path, use_bm25_flag, bm25_k1, bm25_b):
    """Initializer for each worker process."""
    global worker_retriever_instance
    process_name = os.getpid()
    logger.info(f"Worker {process_name}: Initializing XapianRetriever with DB: {db_path}")
    try:
        worker_retriever_instance = XapianRetriever(
            db_path=db_path,
            use_bm25=use_bm25_flag,
            bm25_k1=bm25_k1,
            bm25_b=bm25_b
        )
        logger.info(f"Worker {process_name}: XapianRetriever initialized successfully.")
    except Exception as e:
        logger.error(f"Worker {process_name}: Failed to initialize XapianRetriever: {e}", exc_info=True)

def process_query_task(task_args):
    """
    Worker function to retrieve documents for a single query item.
    task_args: (query_item_dict, k_to_retrieve_for_item)
    """
    global worker_retriever_instance
    query_item, k_retrieve = task_args
    query_text = query_item["query"]
    process_name = os.getpid()

    if worker_retriever_instance is None:
        logger.error(f"Worker {process_name}: XapianRetriever not available for query '{query_text[:50]}...'. Skipping.")
        return {
            "query": query_text,
            "answers": query_item["answers"],
            "retrieved_contexts": []
        }

    try:
        retrieved_docs_with_scores = worker_retriever_instance.search(query_text, k=k_retrieve)
        # Attach score into each metadata dict
        contexts_with_scores = []
        for meta, score in retrieved_docs_with_scores:
            entry = dict(meta)
            entry["score"] = score
            contexts_with_scores.append(entry)
        return {
            "query": query_text,
            "answers": query_item["answers"],
            "retrieved_contexts": contexts_with_scores
        }
    except Exception as e:
        logger.error(f"Worker {process_name}: Error retrieving for query '{query_text[:50]}...': {e}", exc_info=True)
        return {
            "query": query_text,
            "answers": query_item["answers"],
            "retrieved_contexts": []
        }

def load_original_nq_data(file_path):
    """Loads queries and answers from the NQ JSON file."""
    data_items = []
    logger.info(f"Attempting to load data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
            original_data = payload.get("data", [])
        for idx, item in enumerate(original_data):
            query = item.get("question")
            answers = item.get("short_answers", [])
            if query and answers:
                data_items.append({
                    "query": query,
                    "answers": answers
                })
            else:
                logger.warning(f"Skipping item at index {idx} in {file_path} due to missing query or answers.")
    except FileNotFoundError:
        logger.error(f"FATAL: Input file not found: {file_path}")
    except json.JSONDecodeError:
        logger.error(f"FATAL: Error decoding JSON from {file_path}")
    logger.info(f"Loaded {len(data_items)} valid query-answer pairs from {file_path}")
    return data_items

def main_offline_retrieval_all_sets():
    logger.info("Starting offline Xapian context retrieval for all specified datasets...")
    logger.info(f"Using up to {NUM_WORKERS} worker processes.")

    for set_name in INPUT_DATA_FILES.keys():
        input_file = INPUT_DATA_FILES[set_name]
        output_file = OUTPUT_DATA_FILES_WITH_CONTEXTS[set_name]
        k_for_set = K_RETRIEVAL_CONFIG[set_name]

        logger.info(f"\n--- Processing set: {set_name} ---")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"K for retrieval: {k_for_set}")

        queries_to_process_list = load_original_nq_data(input_file)
        if not queries_to_process_list:
            logger.warning(f"No queries loaded for set '{set_name}'. Skipping.")
            continue

        tasks_for_pool = [(q_item, k_for_set) for q_item in queries_to_process_list]

        output_directory = os.path.dirname(output_file)
        if output_directory: # Ensure directory exists if it's not the current one
            os.makedirs(output_directory, exist_ok=True)

        with Pool(processes=NUM_WORKERS,
                  initializer=initialize_worker_retriever,
                  initargs=(XAPIAN_DB_PATH, True, BM25_K1, BM25_B)) as pool:
            with open(output_file, 'w', encoding='utf-8') as outfile_handle:
                for result_item in tqdm(pool.imap_unordered(process_query_task, tasks_for_pool),
                                   total=len(tasks_for_pool),
                                   desc=f"Retrieving contexts for {set_name}"):
                    if result_item: # Ensure result is not None
                        outfile_handle.write(json.dumps(result_item) + "\n")

        logger.info(f"Offline retrieval for set '{set_name}' complete. Results saved to '{output_file}'.")

    logger.info("\nAll offline retrieval processing finished.")

if __name__ == "__main__":
    main_offline_retrieval_all_sets()
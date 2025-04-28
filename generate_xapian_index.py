import os
import json
import logging
import hashlib
import xapian
from multiprocessing import Process, Pool, cpu_count, Manager
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "chunk_dir": "processed_chunks",  # already-chunked .jsonl files
    "xapian_db_path": "wikipedia_xapian_db",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
)
logger = logging.getLogger(__name__)

# Global shared variable for worker processes
global_worker_queue = None

def pool_worker_init(queue):
    global global_worker_queue
    global_worker_queue = queue

def writer_process(queue):
    """
    Dedicated process for writing to Xapian.
    """
    logger.info("Writer process started.")
    os.makedirs(CONFIG["xapian_db_path"], exist_ok=True)
    db = xapian.WritableDatabase(CONFIG["xapian_db_path"], xapian.DB_CREATE_OR_OVERWRITE)
    count = 0

    while True:
        item = queue.get()
        if item is None:
            break
        try:
            term_generator = xapian.TermGenerator()
            term_generator.set_stemmer(xapian.Stem("en"))

            doc = xapian.Document()
            doc.set_data(json.dumps(item))
            term_generator.set_document(doc)
            term_generator.index_text(item["title"], 1, "S")
            term_generator.index_text(item["text"])

            chunk_id = item["id"]
            prefix = "chunk:"
            max_id_length = 245 - len(prefix)
            if len(chunk_id) > max_id_length:
                hashed_id = hashlib.sha256(chunk_id.encode("utf8")).hexdigest()
                term = f"{prefix}{hashed_id}"
            else:
                term = f"{prefix}{chunk_id}"
            doc.add_term(term)

            db.add_document(doc)
            count += 1
        except Exception as e:
            logger.warning(f"Writer failed to process one chunk: {e}")

    db.commit()
    logger.info(f"Writer process finished. Indexed {count} documents.")

def process_file(file_path):
    """
    Worker function to read each chunk in a pre-chunked .jsonl file
    and send it to the writer queue.
    """
    global global_worker_queue
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    chunk = json.loads(line)
                    if isinstance(chunk, dict) and "id" in chunk and "title" in chunk and "text" in chunk:
                        global_worker_queue.put(chunk)
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON in {file_path}")
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
    return file_path

def collect_chunk_files():
    chunk_files = []
    for root, _, files in os.walk(CONFIG["chunk_dir"]):
        for file in files:
            if file.endswith(".jsonl") or file.endswith(".json"):
                chunk_files.append(os.path.join(root, file))
    return chunk_files

def main():
    logger.info("Starting Xapian indexing...")
    chunk_files = collect_chunk_files()
    logger.info(f"Found {len(chunk_files)} pre-chunked files to process.")

    manager = Manager()
    queue = manager.Queue(maxsize=1000)

    writer = Process(target=writer_process, args=(queue,))
    writer.start()

    pool = Pool(processes=cpu_count(), initializer=pool_worker_init, initargs=(queue,))
    try:
        for _ in tqdm(pool.imap_unordered(process_file, chunk_files),
                      total=len(chunk_files), desc="Indexing"):
            pass
    finally:
        pool.close()
        pool.join()

    queue.put(None)
    writer.join()

    logger.info("Xapian indexing completed.")

if __name__ == "__main__":
    main()

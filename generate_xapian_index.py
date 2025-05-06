import os
import json
import logging
import hashlib
import xapian
from multiprocessing import Process, Pool, cpu_count, Manager
from tqdm import tqdm

# --- Configuration ---
CONFIG = {
    "tsv_file": "downloads/data/wikipedia_split/psgs_w100.tsv",
    "xapian_db_path": "/local00/student/shakya/wikipedia_xapian_db",
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] (%(processName)s) %(message)s",
)
logger = logging.getLogger(__name__)

# Global shared variable for worker processes
global_worker_queue = None

def collect_chunk_files():
    return [CONFIG["tsv_file"]]

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
            if count % 10000 == 0:
                print(f"📄 Indexed {count} documents...")
        except Exception as e:
            logger.warning(f"Writer failed to process one chunk: {e}")

    db.commit()
    logger.info(f"Writer process finished. Indexed {count} documents.")

def process_file(file_path):
    global global_worker_queue
    try:
        import csv
        from tqdm import tqdm
        import os
        with open(file_path, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8")) - 1
            f.seek(0)
            reader = csv.DictReader(f, delimiter="\t", fieldnames=["id", "text", "title"])
            next(reader)
            for row in tqdm(reader, total=total_lines, desc=f"Processing {os.path.basename(file_path)}"):
                if row["text"].strip():
                    chunk = {
                        "id": str(row["id"]),
                        "title": row["title"],
                        "text": row["text"]
                    }
                    global_worker_queue.put(chunk)
    except Exception as e:
        logger.error(f"Failed to process file {file_path}: {e}")
    return file_path


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
        for _ in tqdm(pool.imap_unordered(process_file, chunk_files), desc="Indexing"):
            pass
    finally:
        pool.close()
        pool.join()

    queue.put(None)
    writer.join()

    logger.info("Xapian indexing completed.")

if __name__ == "__main__":
    main()

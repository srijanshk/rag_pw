import os
import json
import re
import spacy
import torch
from tqdm import tqdm
import csv
import logging
from multiprocessing import Pool, get_context
from FlagEmbedding import BGEM3FlagModel
from itertools import islice
import time

# --- Setup ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
INPUT_JSONL_FILE = "./thesis_datasets/openmathinstruct2/openmathinstruct2_train_streamed.jsonl"
OUTPUT_TSV_FILE = "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing_bge-m3.tsv"
TEMP_OUTPUT_FILE = OUTPUT_TSV_FILE + ".temp"

# Chunking Parameters
MAX_SENTENCES_BEFORE_CHECK = 6
SIM_THRESHOLD = 0.85 

# Processing Configuration
NUM_CPU_WORKERS = max(1, os.cpu_count() - 2)  # Leave 2 CPUs for main process and GPU operations
PROCESSING_BATCH_SIZE = 50000  # Records per batch
EMBEDDING_BATCH_SIZE = 1024   # Optimal batch size for BGE-M3
CHECKPOINT_INTERVAL = 100000  # Save progress every 100K records
MAX_RECORDS_IN_MEMORY = 50000 # Process in memory chunks

# Regex for hard breaks
STEP_REGEX = re.compile(r"^(?:#+\s|Step\s*\d+|Therefore|Hence|Thus\b|Finally\b)", re.I)

# Global for CPU worker processes
nlp = None

def worker_init_cpu():
    """Initializes spaCy in each CPU worker process."""
    global nlp
    if nlp is None:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner", "lemmatizer"])
        nlp.add_pipe('sentencizer')  # Ensure sentence boundary detection

def process_record_cpu_part(line_and_idx):
    """Worker function that takes (line, line_idx) tuple."""
    line, line_idx = line_and_idx
    line = line.strip()
    if not line: 
        return None
    
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return None

    solution_full = record.get("generated_solution") or record.get("solution", "")
    if not solution_full or not isinstance(solution_full, str): 
        return None
    
    # Fast sentence splitting with minimal processing
    try:
        doc = nlp(solution_full)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception as e:
        logger.warning(f"Sentence splitting failed: {e}")
        sentences = [solution_full]  # Fallback to whole text
    
    if not sentences:
        return None
        
    return {
        "problem": record.get("problem", ""),
        "sentences": sentences,
        "expected_answer": record.get("expected_answer", ""),
        "problem_source": record.get("problem_source", ""),
        "line_idx": line_idx  # Use the provided line index
    }

def sanitize_for_tsv(text: str) -> str:
    """Replaces characters that interfere with TSV formatting."""
    if not isinstance(text, str): 
        return ""
    return ' '.join(text.replace('\t', ' ').replace('\n', ' ').split())

def batch_generator(iterable, batch_size):
    """Yield batches with line numbers."""
    iterator = enumerate(iterable)  # Keep track of line numbers
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        # Separate lines and indices
        indices, lines = zip(*batch)
        yield (lines, indices)

def process_batch(embedder, batch_records):
    """Process a batch of records with GPU acceleration."""
    all_sentences = []
    record_indices = []
    
    # Prepare batch data
    for i, record in enumerate(batch_records):
        if record and record["sentences"]:
            all_sentences.extend(record["sentences"])
            record_indices.extend([i] * len(record["sentences"]))
    
    if not all_sentences:
        return batch_records
    
    # Batch process embeddings
    try:
        embedding_dict = embedder.encode(
            all_sentences, 
            batch_size=EMBEDDING_BATCH_SIZE, 
            max_length=512,
            return_dense=True, 
            return_sparse=False, 
            return_colbert_vecs=False
        )
        
        # Normalize embeddings
        sentence_embeddings = torch.nn.functional.normalize(
            torch.from_numpy(embedding_dict['dense_vecs']), p=2, dim=1
        )
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return batch_records
    
    # Reconstruct records with embeddings
    embedding_ptr = 0
    for record in batch_records:
        if not record or not record["sentences"]:
            continue
            
        num_sentences = len(record["sentences"])
        record["embeddings"] = sentence_embeddings[embedding_ptr:embedding_ptr+num_sentences]
        embedding_ptr += num_sentences
        
    return batch_records

def chunk_sentences(record):
    """Chunk sentences based on embeddings and rules."""
    if not record or not record["sentences"]:
        return []
        
    embeddings = record.get("embeddings", None)
    if embeddings is None:
        return [" ".join(record["sentences"])]  # Fallback if no embeddings
        
    chunks, cur_chunk = [], []
    
    for i, (sentence, embedding) in enumerate(zip(record["sentences"], embeddings)):
        if STEP_REGEX.match(sentence) and cur_chunk:
            chunks.append(" ".join(cur_chunk))
            cur_chunk = []
            
        if len(cur_chunk) >= MAX_SENTENCES_BEFORE_CHECK and i + 1 < len(record["sentences"]):
            try:
                sim = torch.dot(embedding, embeddings[i+1]).item()
                if sim < SIM_THRESHOLD:
                    chunks.append(" ".join(cur_chunk))
                    cur_chunk = []
            except Exception:
                pass
                
        cur_chunk.append(sentence)
        
    if cur_chunk:
        chunks.append(" ".join(cur_chunk))
        
    return chunks

def count_lines(filename):
    """Count lines in a file efficiently."""
    with open(filename, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)

def process_batches(embedder, pool, batch_iterator, total_lines, output_file):
    """Process batches with checkpointing and progress tracking."""
    writer = None
    checkpoint_counter = 0
    total_chunks_written = 0
    start_time = time.time()
    
    # Open output file
    f_out = open(output_file, 'w', encoding='utf-8', newline='')
    writer = csv.writer(f_out, delimiter='\t')
    writer.writerow(["row_id", "chunk_id", "problem", "solution_chunk", "expected_answer", "problem_from"])
    
    pbar = tqdm(total=total_lines, desc="Processing Records")
    
    try:
        for batch_idx, (batch_lines, batch_indices) in enumerate(batch_iterator):
            # Create (line, idx) tuples for each record
            batch_with_indices = list(zip(batch_lines, batch_indices))
            
            # Process batch
            processed_batch = list(pool.imap(process_record_cpu_part, batch_with_indices, chunksize=100))
            processed_batch = process_batch(embedder, processed_batch)
            
            # Write results
            batch_chunks = 0
            for record in processed_batch:
                if not record:
                    continue
                    
                chunks = chunk_sentences(record)
                for chunk_id, chunk_text in enumerate(chunks):
                    writer.writerow([
                        record["line_idx"],  # Now using the correct line index
                        chunk_id,
                        sanitize_for_tsv(record["problem"]),
                        sanitize_for_tsv(chunk_text),
                        sanitize_for_tsv(str(record["expected_answer"])),
                        sanitize_for_tsv(str(record["problem_source"]))
                    ])
                    batch_chunks += 1
            
            total_chunks_written += batch_chunks
            checkpoint_counter += len(batch_lines)
            
            # Update progress
            pbar.update(len(batch_lines))
            pbar.set_postfix({
                "Chunks": f"{total_chunks_written:,}",
                "Rec/s": f"{(batch_idx * PROCESSING_BATCH_SIZE)/(time.time()-start_time):.1f}"
            })
            
            # Checkpoint
            if checkpoint_counter >= CHECKPOINT_INTERVAL:
                f_out.flush()
                os.fsync(f_out.fileno())
                checkpoint_counter = 0
                logger.info(
                    f"Checkpoint: Processed {batch_idx * PROCESSING_BATCH_SIZE:,} records, "
                    f"{total_chunks_written:,} chunks, "
                    f"{(batch_idx * PROCESSING_BATCH_SIZE)/(time.time()-start_time):.1f} rec/s"
                )
            
            # Clear memory
            del processed_batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise
    finally:
        pbar.close()
        f_out.close()
    
    return total_chunks_written

if __name__ == "__main__":
    logger.info(f"Reading from JSONL: {INPUT_JSONL_FILE}")
    logger.info(f"Writing TSV to: {OUTPUT_TSV_FILE}")

    # Remove existing files
    for f in [OUTPUT_TSV_FILE, TEMP_OUTPUT_FILE]:
        if os.path.exists(f):
            os.remove(f)

    # Load model
    try:
        logger.info("Loading BGE-M3 model...")
        embedder = BGEM3FlagModel(
            'BAAI/bge-m3', 
            use_fp16=True,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        logger.info("✅ Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        exit()

    # Count lines
    logger.info("Counting lines in input file...")
    try:
        total_lines = count_lines(INPUT_JSONL_FILE)
        logger.info(f"Total records to process: {total_lines:,}")
    except Exception as e:
        logger.error(f"Failed to count lines: {e}")
        exit()

    # Process in chunks
    ctx = get_context("spawn")
    try:
        with ctx.Pool(processes=NUM_CPU_WORKERS, initializer=worker_init_cpu) as pool:
            with open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f_in:
                batch_iterator = batch_generator(f_in, PROCESSING_BATCH_SIZE)
                total_chunks = process_batches(
                    embedder, pool, batch_iterator, total_lines, TEMP_OUTPUT_FILE
                )
        
        # Finalize output
        os.rename(TEMP_OUTPUT_FILE, OUTPUT_TSV_FILE)
        logger.info(f"\n✅ Finished processing {total_lines:,} records")
        logger.info(f"Total chunks written: {total_chunks:,}")
        logger.info(f"Output file: {OUTPUT_TSV_FILE}")
        
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        if os.path.exists(TEMP_OUTPUT_FILE):
            logger.info(f"Partial output saved at: {TEMP_OUTPUT_FILE}")
        exit(1)
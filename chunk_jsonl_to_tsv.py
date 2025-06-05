import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import json
import re
from multiprocessing import Pool, cpu_count
import spacy
import torch
from sentence_transformers import SentenceTransformer, util

# ── CUDA / Device Configuration ───────────────────────────────────────────────────
# Check if a CUDA GPU is available; otherwise fall back to CPU.
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA is available. Using device: {DEVICE}")
else:
    DEVICE = "cpu"
    print("CUDA not available. Falling back to CPU.")

# ── Configuration ────────────────────────────────────────────────────────────────

# Input and output paths
INPUT_JSONL_FILE = "./thesis_datasets/openmathinstruct2/openmathinstruct2_train_streamed.jsonl"
OUTPUT_TSV_FILE  = "./thesis_datasets/openmathinstruct2/openmath_chunked_for_indexing.tsv"

# 1) Regex markers for hard structural breaks in solutions
STEP_REGEX = re.compile(r"^(?:#+\s|Step\s*\d+|Therefore|Hence|Thus\b|Finally\b)", re.I)

# 2) Maximum number of sentences before triggering a semantic‐drift check
MAX_SENTENCES_BEFORE_CHECK = 6

# 3) Cosine‐similarity threshold: if sim < this between adjacent sentences, split
SIM_THRESHOLD = 0.82

# 4) Multiprocessing configuration
NUM_WORKERS = max(1, cpu_count() - 1)
BATCH_SIZE = 100

# Placeholders for per-process models (set in worker_init)
nlp = None
embedder = None


def worker_init():
    """Initialize models inside each worker process."""
    global nlp, embedder
    if nlp is None:
        print("Loading spaCy model (en_core_web_sm) in worker...")
        nlp = spacy.load("en_core_web_sm")
    if embedder is None:
        print(f"Loading SentenceTransformer (intfloat/e5-large-v2) on {DEVICE} in worker...")
        embedder = SentenceTransformer("intfloat/e5-large-v2", device=DEVICE)


# ── Helper Functions ─────────────────────────────────────────────────────────────

def _split_into_sentences(text: str) -> list[str]:
    """
    Splits the input text into a list of sentence strings using spaCy.
    Filters out any empty sentences.
    """
    global nlp
    if nlp is None:
        worker_init()
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def embed_sentences(sent_list: list[str]) -> torch.Tensor:
    """
    Given a list of sentence strings, returns a torch.Tensor of normalized embeddings.
    Each sentence is prefixed with "passage:" before encoding.
    Embeddings live on DEVICE (GPU if available).
    """
    global embedder
    if embedder is None:
        worker_init()
    inputs = [f"passage: {s}" for s in sent_list]
    embeddings: torch.Tensor = embedder.encode(
        inputs,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_tensor=True,
    )
    # embeddings.shape == (len(sent_list), embedding_dim)
    return embeddings  # already on DEVICE


def semantic_chunks(text: str,
                    max_sentences: int = MAX_SENTENCES_BEFORE_CHECK,
                    sim_threshold: float = SIM_THRESHOLD) -> list[str]:
    """
    Splits `text` (a long solution string) into semantically coherent chunks,
    using spaCy for sentences and E5 embeddings (on DEVICE) for drift detection.

    Algorithm:
      1) Sentence‐tokenize the text with spaCy.
      2) Pre‐compute embeddings for every sentence (on GPU if available).
      3) Iterate sentence by sentence, accumulating a current chunk.
         a) If a sentence matches STEP_REGEX and the current chunk is nonempty,
            flush the chunk (start a new one).
         b) Otherwise, add the sentence to the current chunk.
         c) Once the chunk reaches `max_sentences` sentences, compare the embedding
            of the last sentence in the chunk to the embedding of the next sentence
            in the full text. If cosine similarity < sim_threshold, flush the chunk.
      4) At the end, flush any remaining sentences as a final chunk.

    Returns:
      List[str]: Each element is a single chunk (joined sentences) as a string.
    """
    # 1) Split text into sentences
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    # 2) Embed all sentences once (on GPU if available)
    sentence_embeddings: torch.Tensor = embed_sentences(sentences)
    # shape: (num_sentences, hidden_dim)

    chunks: list[str] = []
    cur_chunk_sentences: list[str] = []
    cur_chunk_indices: list[int] = []

    def flush_current_chunk():
        """Joins current chunk sentences into one string, appends to chunks, clears buffers."""
        if not cur_chunk_sentences:
            return
        joined = " ".join(cur_chunk_sentences).replace("\n", " ").strip()
        if joined:
            chunks.append(joined)
        cur_chunk_sentences.clear()
        cur_chunk_indices.clear()

    # 3) Iterate through sentences
    for idx, sentence in enumerate(sentences):
        # (a) Hard marker break
        if STEP_REGEX.match(sentence) and cur_chunk_sentences:
            flush_current_chunk()
            cur_chunk_sentences.append(sentence)
            cur_chunk_indices.append(idx)
            continue

        # (b) Otherwise, accumulate
        cur_chunk_sentences.append(sentence)
        cur_chunk_indices.append(idx)

        # (c) Check semantic drift if max_sentences is reached
        if len(cur_chunk_sentences) >= max_sentences:
            next_idx = idx + 1
            if next_idx < len(sentences):
                emb_last = sentence_embeddings[idx].unsqueeze(0)   # shape: (1, hidden_dim)
                emb_next = sentence_embeddings[next_idx].unsqueeze(0)  # shape: (1, hidden_dim)
                cosine_sim = util.pytorch_cos_sim(emb_last, emb_next).item()
                if cosine_sim < sim_threshold:
                    flush_current_chunk()
                # else: keep adding to same chunk until next check

    # 4) Flush any remainder
    flush_current_chunk()
    return chunks


def process_json_record(args) -> list[str]:
    """Worker function to process a single JSONL line."""
    line_idx, line = args
    line = line.strip()
    if not line:
        return []
    try:
        record = json.loads(line)
    except json.JSONDecodeError:
        return []

    row_id = line_idx
    problem_text = record.get("problem", "")
    solution_full = (
        record.get("generated_solution")
        or record.get("solution")
        or record.get("output")
        or ""
    )
    problem_source = record.get("problem_from", "")

    if not solution_full or not isinstance(solution_full, str):
        return []

    chunks = semantic_chunks(solution_full)
    out_lines = []
    for chunk_id, chunk_text in enumerate(chunks):
        clean_problem = problem_text.replace("\t", " ").replace("\n", " ").strip()
        clean_chunk = chunk_text.replace("\t", " ").replace("\n", " ").strip()
        clean_solution_full = solution_full.replace("\t", " ").replace("\n", " ").strip()
        clean_problem_src = problem_source.replace("\t", " ").replace("\n", " ").strip()
        out_lines.append(
            f"{row_id}\t{chunk_id}\t{clean_problem}\t{clean_chunk}\t{clean_solution_full}\t{clean_problem_src}\n"
        )

    return out_lines


# ── Main Processing ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Reading from JSONL: {INPUT_JSONL_FILE}")
    print(f"Writing TSV to:    {OUTPUT_TSV_FILE}\n")

    total_records = 0
    total_chunks = 0

    with open(INPUT_JSONL_FILE, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_TSV_FILE, "w", encoding="utf-8") as f_out:

        f_out.write("row_id\tchunk_id\tproblem\tsolution_chunk\tsolution\tproblem_from\n")

        pool = Pool(processes=NUM_WORKERS, initializer=worker_init)
        batch = []
        for line_idx, line in enumerate(f_in):
            batch.append((line_idx, line))
            if len(batch) >= BATCH_SIZE:
                for res_lines in pool.imap_unordered(process_json_record, batch):
                    for out_line in res_lines:
                        f_out.write(out_line)
                    total_chunks += len(res_lines)
                total_records += len(batch)
                if total_records % 500 == 0:
                    print(f"Processed {total_records} records, generated ~{total_chunks} chunks...")
                batch = []

        if batch:
            for res_lines in pool.imap_unordered(process_json_record, batch):
                for out_line in res_lines:
                    f_out.write(out_line)
                total_chunks += len(res_lines)
            total_records += len(batch)

        pool.close()
        pool.join()

    print(f"\nFinished! Total records processed: {total_records}")
    print(f"Total chunks written: {total_chunks}")
    print(f"TSV file saved at: {OUTPUT_TSV_FILE}")

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List

import nltk
import tqdm
from transformers import AutoTokenizer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3,4" # Use 2 GPUs
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

############################################################
#                       CONSTANTS                          #
############################################################
MODEL_NAME = "BAAI/bge-m3"
CHUNK_TOKENS = 512        # hard budget
OVERLAP_TOKENS = 20       # last 20 tokens of prev → prefix of next
STRIDE_OVERSIZE = 32      # stride within very long sentence
BUFFER_FLUSH = 2000       # write after this many chunks

# regex helpers for soft cuts inside long math sentences
SOFT_SPLIT_RE = re.compile(r"(?:\\\\|[;,:]|\band\b|\bwhere\b|\bsuch that\b)")

############################################################
#                 TOKENIZER & SENTENCE SPLIT               #
############################################################
print("🔄  Loading tokenizer…", file=sys.stderr)
TOKENIZER = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)
# Disable HuggingFace's hard cap (8 192) *and* raise our manual cap high
TOKENIZER.model_max_length = 2_147_483_647  # effectively "no limit"

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

############################################################
#                 CHUNKING UTILITY FUNCTIONS               #
############################################################

def _slide_token_window(ids: List[int], max_len: int = CHUNK_TOKENS, stride: int = STRIDE_OVERSIZE) -> List[List[int]]:
    """Produce overlapping windows that cover *all* token ids."""
    if stride >= max_len:
        stride = max_len // 2  # always leave at least half overlap
    return [ids[i : i + max_len] for i in range(0, len(ids), max_len - stride)]


def _split_oversize_sentence(text: str, max_len: int = CHUNK_TOKENS) -> List[str]:
    """Break a single >max_len sentence into smaller pieces (no truncation)."""
    try:
        ids = TOKENIZER(text, add_special_tokens=False).input_ids
    except Exception as e:
        # Extremely pathological input (e.g. >2 B tokens) — fall back to crude char split
        if "sequence length" in str(e):
            approx_tokens = len(text) // 2  # rough guess
            slice_size = max_len * 6  # generous slice so we don't iterate forever
            parts = [text[i : i + slice_size] for i in range(0, len(text), slice_size)]
            out = []
            for p in parts:
                out.extend(_split_oversize_sentence(p, max_len))
            return out
        raise

    if len(ids) <= max_len:
        return [text]

    # Try softer linguistic splits first.
    parts = [p.strip() for p in SOFT_SPLIT_RE.split(text) if p.strip()]
    if len(parts) > 1:
        out, cur = [], ""
        for p in parts:
            tentative = f"{cur} {p}".strip()
            if len(TOKENIZER(tentative, add_special_tokens=False).input_ids) <= max_len:
                cur = tentative
            else:
                out.append(cur)
                cur = p
        if cur:
            out.append(cur)
        if all(len(TOKENIZER(s, add_special_tokens=False).input_ids) <= max_len for s in out):
            return out

    # Fallback: raw token sliding window.
    return [TOKENIZER.decode(w) for w in _slide_token_window(ids, max_len)]


def chunk_text(doc: str) -> List[str]:
    """Sentence-aware packing with 20-token overlap that never exceeds 512."""
    sentences = nltk.tokenize.sent_tokenize(doc)

    chunks: List[str] = []
    cur_ids: List[int] = []

    for sent in sentences:
        for piece in _split_oversize_sentence(sent):
            piece_ids = TOKENIZER(piece, add_special_tokens=False).input_ids

            # If adding this piece would overflow → flush
            if cur_ids and len(cur_ids) + len(piece_ids) > CHUNK_TOKENS:
                chunks.append(TOKENIZER.decode(cur_ids))
                cur_ids = cur_ids[-OVERLAP_TOKENS:] if OVERLAP_TOKENS else []

            # Safety: piece still longer than budget (shouldn’t happen)
            if len(piece_ids) > CHUNK_TOKENS:
                for win in _slide_token_window(piece_ids, CHUNK_TOKENS):
                    if cur_ids:
                        chunks.append(TOKENIZER.decode(cur_ids))
                        cur_ids = []
                    chunks.append(TOKENIZER.decode(win))
                continue

            cur_ids.extend(piece_ids)

    if cur_ids:
        chunks.append(TOKENIZER.decode(cur_ids))
    return chunks

############################################################
#                         MAIN CLI                         #
############################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk MathPile → single TSV (no embedding yet)")
    parser.add_argument("--root", type=Path, required=True, help="Folder containing MathPile JSONL / TXT files")
    parser.add_argument("--out", type=Path, required=True, help="Output .tsv path")
    parser.add_argument("--resume", action="store_true", help="Skip files already processed (based on initial row count)")
    args = parser.parse_args()

    # Ensure output directory exists
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Gather input files
    files = list(args.root.rglob("*.jsonl")) + list(args.root.rglob("*.json")) + list(args.root.rglob("*.txt"))
    print(f"🗂  Found {len(files)} input files", file=sys.stderr)

    # If --resume, capture how many rows already exist
    processed_rows = 0
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8", newline="") as tsv_fp:
            processed_rows = sum(1 for _ in tsv_fp) - 1  # header
        print(f"⏩  Resume mode: {processed_rows:,} rows already in TSV", file=sys.stderr)

    # Open TSV writer
    mode = "a" if args.resume else "w"
    with args.out.open(mode, encoding="utf-8", newline="") as tsv_fp:
        writer = csv.writer(tsv_fp, delimiter="\t", quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
        if mode == "w":
            writer.writerow(["text", "source_file", "line_idx", "chunk_id"])  # header

        buffer_rows = []
        row_counter = processed_rows

        for fp in tqdm.tqdm(files, desc="Chunking"):
            is_jsonl = fp.suffix.lower() in {".jsonl", ".json"}
            with fp.open("r", encoding="utf-8", errors="ignore") as f:
                for line_idx, line in enumerate(f):
                    if is_jsonl:
                        try:
                            text = json.loads(line).get("text", "")
                        except Exception:
                            continue
                    else:
                        text = line.strip()

                    if not text:
                        continue

                    for chunk_id, chunk in enumerate(chunk_text(text)):
                        if row_counter < processed_rows:
                            row_counter += 1  # skip rows already written
                            continue
                        safe_chunk = chunk.replace("\n", " ")  # ensure single‑line TSV cell
                        buffer_rows.append([safe_chunk, str(fp.relative_to(args.root)), line_idx, chunk_id])
                        row_counter += 1

                        if len(buffer_rows) >= BUFFER_FLUSH:
                            writer.writerows(buffer_rows)
                            buffer_rows.clear()

        if buffer_rows:
            writer.writerows(buffer_rows)

    print(f"✅  TSV written to {args.out.resolve()}", file=sys.stderr)
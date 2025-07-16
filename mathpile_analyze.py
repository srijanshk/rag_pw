from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

############################################################
#                     CONFIG / CLI                         #
############################################################
parser = argparse.ArgumentParser(description="Compute token‑length stats from chunk TSV")
parser.add_argument("--tsv", type=Path, required=True, help="Path to mathpile_chunks.tsv")
parser.add_argument(
    "--percentiles",
    type=int,
    nargs="*",
    default=[5, 25, 50, 75, 90, 95, 99, 100],
    help="Percentiles to report (0‑100)",
)
parser.add_argument("--hist", type=Path, default=None, help="Optional PNG file to save histogram")
parser.add_argument("--hist_bins", type=int, default=50, help="Number of histogram bins")
args = parser.parse_args()

percentiles = sorted(set([p for p in args.percentiles if 0 <= p <= 100]))
if not percentiles:
    parser.error("At least one valid percentile (0‑100) required")

############################################################
#                      TOKENIZER                           #
############################################################
print("🔄  Loading tokenizer …", file=sys.stderr)
TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
# Disable the safety cap so we can inspect any length
TOKENIZER.model_max_length = 1_000_000

############################################################
#                SCAN TSV & COLLECT LENGTHS               #
############################################################
lengths: List[int] = []

with args.tsv.open("r", encoding="utf-8", newline="") as tsv_fp:
    rdr = csv.reader(tsv_fp, delimiter="\t")
    header = next(rdr, None)
    if header is None or header[0].lower() != "text":
        raise ValueError("TSV does not appear to have expected header (text …)")

    for row in tqdm(rdr, desc="Counting tokens", unit="chunks"):
        text = row[0]
        tok_len = len(TOKENIZER(text, add_special_tokens=False).input_ids)
        lengths.append(tok_len)

lengths_np = np.asarray(lengths, dtype=np.uint16)

############################################################
#                  CALCULATE STATISTICS                   #
############################################################
print("\nToken‑length statistics (BAAI/bge‑m3 tokenizer):")
print("────────────────────────────────────────────────────")
print(f"chunks analysed : {len(lengths_np):,}")
print(f"min tokens      : {int(lengths_np.min())}")
print(f"max tokens      : {int(lengths_np.max())}")
print(f"mean tokens     : {lengths_np.mean():.2f}")
print(f"median tokens   : {np.median(lengths_np):.0f}")
for p in percentiles:
    val = np.percentile(lengths_np, p, interpolation="nearest")
    print(f"{p:>3d}th percentile : {int(val)}")

############################################################
#                     HISTOGRAM (opt)                     #
############################################################
if args.hist:
    try:
        import matplotlib.pyplot as plt  # type: ignore

        plt.figure()
        plt.hist(lengths_np, bins=args.hist_bins)
        plt.title("Token length distribution")
        plt.xlabel("tokens per chunk")
        plt.ylabel("frequency")
        plt.tight_layout()
        plt.savefig(args.hist, dpi=150)
        print(f"\n📊  Histogram saved → {args.hist.resolve()}")
    except Exception as e:
        print(f"⚠️  Failed to write histogram: {e}", file=sys.stderr)

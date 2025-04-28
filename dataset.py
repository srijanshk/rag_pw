import csv
import json
from pathlib import Path
from collections import defaultdict
import random

# === Config ===
DATA_DIR = Path("msmarco")
MAX_NEGATIVES = 3

# === Load Corpus ===
print("📦 Loading corpus...")
corpus = {}
with open(DATA_DIR / "collection.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        pid, passage = row
        corpus[pid] = passage

# === Load Train Queries ===
print("🔹 Loading train queries...")
queries = {}
with open(DATA_DIR / "queries/queries.train.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        qid, query = row
        queries[qid] = query

# === Load Train Qrels ===
qrels = defaultdict(set)
with open(DATA_DIR / "qrels.train.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        qid, _, pid, _ = row
        qrels[qid].add(pid)

# === Build Triplets ===
print("🛠️  Building training triplets...")
train_data = []
all_pids = list(corpus.keys())

for qid, pos_ids in qrels.items():
    query = queries.get(qid)
    if not query:
        continue
    for pos_id in pos_ids:
        # Sample hard negatives randomly (excluding positives)
        negs = [pid for pid in random.sample(all_pids, 50) if pid not in pos_ids]
        negs_text = [corpus[nid] for nid in negs[:MAX_NEGATIVES]]
        train_data.append({
            "query": query,
            "positive": corpus[pos_id],
            "negatives": negs_text
        })

with open("msmarco_train.json", "w") as f:
    json.dump(train_data, f, indent=2)
print("✅ Saved: msmarco_train.json")

# === Prepare Dev Queries & Qrels ===
dev_queries = {}
with open(DATA_DIR / "queries/queries.dev.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        qid, text = row
        dev_queries[qid] = text

dev_qrels = defaultdict(list)
with open(DATA_DIR / "qrels.dev.tsv", encoding="utf-8") as f:
    for row in csv.reader(f, delimiter="\t"):
        qid, _, pid, _ = row
        dev_qrels[qid].append(pid)

with open("msmarco_dev_queries.jsonl", "w") as f:
    for qid, text in dev_queries.items():
        f.write(json.dumps({"id": qid, "text": text}) + "\n")

with open("msmarco_dev_qrels.json", "w") as f:
    json.dump(dev_qrels, f, indent=2)

print("✅ Saved: msmarco_dev_queries.jsonl + msmarco_dev_qrels.json")

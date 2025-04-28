
import os
import json
import re
import string
import faiss
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict

# === MS MARCO Evaluation Functions ===
def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = ''.join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def compute_mrr(ranks):
    return round(np.mean([1.0 / rank for rank in ranks if rank > 0]), 4) if ranks else 0.0

def recall_at_k(ranks, k):
    return round(sum(1 for r in ranks if r <= k) / len(ranks), 4) if ranks else 0.0

# === Main Evaluation ===
def evaluate_on_msmarco_dev(
    retriever,
    qrels_path="msmarco_dev_qrels.json",
    queries_path="msmarco_dev_queries.jsonl",
    top_k=10,
    use_hybrid=False,
    log_path="retrieval_eval_log.json"
):
    # Load queries
    queries = []
    with open(queries_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            queries.append((data["id"], data["text"]))

    # Load qrels
    with open(qrels_path, encoding="utf-8") as f:
        qrels = json.load(f)  # dict[qid] = [doc_ids]

    ranks = []
    logs = []

    for qid, query in tqdm(queries, desc="Evaluating queries"):
        relevant_docs = qrels.get(qid, [])
        if not relevant_docs:
            continue

        retrieved = retriever.search(query, k=top_k)  # returns (metadata, score)
        retrieved_ids = [m.get("docid", str(i)) for i, (m, _) in enumerate(retrieved)]

        rank = 0
        for i, doc_id in enumerate(retrieved_ids, start=1):
            if doc_id in relevant_docs:
                rank = i
                break

        ranks.append(rank)

        logs.append({
            "qid": qid,
            "query": query,
            "relevant_ids": relevant_docs,
            "retrieved_ids": retrieved_ids,
            "rank_of_first_hit": rank
        })

    mrr = compute_mrr(ranks)
    recall10 = recall_at_k(ranks, 10)
    recall5 = recall_at_k(ranks, 5)

    with open(log_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"\n📊 Evaluation Results")
    print(f"MRR@10: {mrr:.4f}")
    print(f"Recall@10: {recall10:.4f}")
    print(f"Recall@5: {recall5:.4f}")

    return {"MRR@10": mrr, "Recall@10": recall10, "Recall@5": recall5}
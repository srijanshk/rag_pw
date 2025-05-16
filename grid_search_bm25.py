#!/usr/bin/env python3
import json
import itertools
from tqdm import tqdm
import wandb
from xapian_retriever import XapianRetriever

# -- CONFIGURATION ---------------------------------------------------------

DEV_JSON = "downloads/data/gold_passages_info/nq_dev.json"  # your original NQ-dev file
KS       = [1, 5, 10, 50]

# Grid of BM25 parameters to try
GRID_K1 = [0.8, 1.0, 1.2, 1.5, 1.8]
GRID_B  = [0.0, 0.25, 0.5, 0.75, 1.0]

# Path to your Xapian DB
XAPIAN_DB_PATH = "/local00/student/shakya/wikipedia_xapian_db"


# -- HELPERS ---------------------------------------------------------------

def load_dev_data(dev_json_path):
    """Return list of (query, answers) from your NQ-dev JSON."""
    with open(dev_json_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
    examples = []
    for item in payload.get("data", []):
        q = item.get("question")
        ans = item.get("short_answers", [])
        if q and ans:
            examples.append((q, ans))
    return examples

def recall_at_k(retrieved, answers, k):
    """1 if any answer str appears within the top-k retrieved passages."""
    answers = [a.lower() for a in answers]
    # Expecting retrieved to be a list of dicts with 'score' or tuples (meta, score)
    for entry in retrieved[:k]:
        meta = entry if isinstance(entry, dict) else entry[0]
        text = (meta.get("text") or meta.get("context") or "").lower()
        if any(ans in text for ans in answers):
            return 1
    return 0

def evaluate_params(k1, b, dev_data):
    """
    For a given (k1, b), build a retriever, run through dev_data, 
    and return a dict of recall@K.
    """
    retriever = XapianRetriever(
        db_path=XAPIAN_DB_PATH,
        use_bm25=True,
        bm25_k1=k1,
        bm25_b=b
    )
    # Counters
    hits   = {k: 0 for k in KS}
    total  = {k: 0 for k in KS}

    for query, answers in tqdm(dev_data, desc=f"Eval k1={k1}, b={b}"):
        # retrieve top max(KS)
        retrieved = retriever.search(query, k=max(KS))
        # if entries are dicts with 'score', sort them
        if retrieved and isinstance(retrieved[0], dict) and 'score' in retrieved[0]:
            retrieved = sorted(retrieved, key=lambda e: e['score'], reverse=True)
        for k in KS:
            hits[k]  += recall_at_k(retrieved, answers, k)
            total[k] += 1

    # Compute recall rates
    return {k: hits[k]/total[k] for k in KS}


# -- MAIN GRID SEARCH ------------------------------------------------------

def main():
    # Initialize W&B run
    wandb.init(
        project="bm25-grid-search",
        config={
            "GRID_K1": GRID_K1,
            "GRID_B": GRID_B,
            "KS": KS
        }
    )
    dev_data = load_dev_data(DEV_JSON)
    results  = []

    # Sweep over all (k1, b) combinations
    for k1, b in itertools.product(GRID_K1, GRID_B):
        rates = evaluate_params(k1, b, dev_data)
        # Log this configuration and its recall metrics to W&B
        wandb.log({
            "bm25_k1": k1,
            "bm25_b": b,
            **{f"recall@{k}": rates[k] for k in KS}
        })
        results.append({
            "k1": k1,
            "b" : b,
            **{f"recall@{k}": rates[k] for k in KS}
        })

    # Sort by best Recall@10 descending
    results.sort(key=lambda x: x["recall@10"], reverse=True)

    # Print table
    header = ["k1", "b"] + [f"recall@{k}" for k in KS]
    print("\t".join(header))
    for r in results:
        print("\t".join(f"{r[col]:.4f}" if isinstance(r[col], float) else str(r[col])
                        for col in header))
    wandb.finish()

if __name__ == "__main__":
    main()
import json
from tqdm import tqdm

def recall_at_k(retrieved, answers, k):
    """
    Returns 1 if any ground-truth answer string appears in the top-k retrieved passages.
    `retrieved` is a list of (meta_dict, score) tuples.
    """
    answers = [a.lower() for a in answers]
    for entry in retrieved[:k]:
        # Handle dict entries (with 'score') or tuple entries (meta, score)
        if isinstance(entry, dict):
            meta = entry
        else:
            meta, _ = entry
        text = meta.get("text", "") or meta.get("context", "")
        if any(ans in text.lower() for ans in answers):
            return 1
    return 0

def compute_recalls(retrieval_jsonl_path, ks=(1,5,10,50)):
    totals = {k: 0 for k in ks}
    hits   = {k: 0 for k in ks}
    n = 0

    with open(retrieval_jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Evaluating recall"):
            record = json.loads(line)
            answers = record.get("answers", [])
            retrieved = record.get("retrieved_contexts", [])
            # If contexts include a 'score' field, sort descending by it
            if retrieved and isinstance(retrieved[0], dict) and 'score' in retrieved[0]:
                retrieved = sorted(retrieved, key=lambda e: e['score'], reverse=True)
            if not answers:
                continue
            n += 1
            for k in ks:
                hits[k]   += recall_at_k(retrieved, answers, k)
                totals[k] += 1

    print(f"\nEvaluated on {n} examples\n")
    for k in ks:
        rate = hits[k] / totals[k] if totals[k] else 0.0
        print(f"Recall@{k}: {rate:.4f} ({hits[k]}/{totals[k]})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Compute Recall@K from offline retrieval JSONL")
    p.add_argument(
        "-r", "--retrieval", required=True,
        help="Path to your offline retrieval JSONL (must contain 'answers' & 'retrieved_contexts')"
    )
    p.add_argument(
        "-k", "--ks", nargs="+", type=int, default=[1,5,10,50],
        help="List of K values for Recall@K"
    )
    args = p.parse_args()
    compute_recalls(args.retrieval, tuple(args.ks))
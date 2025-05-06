import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import random
from Dense_Retriever import DenseRetriever
import torch
import faiss 

def evaluate_retriever_recall(retriever, data, k=50):
    correct = 0
    total = 0
    for sample in data:
        query = sample["query"]
        positive_id = sample["positive_id"]
        retrieved = retriever.search(query, k)
        retrieved_ids = [ str(md.get("id")) for md, _ in retrieved ]
        if positive_id in retrieved_ids:
            correct += 1
        total += 1
    recall = correct / total if total > 0 else 0
    print(f"🔎 Retriever Recall@{k}: {recall:.3f}")
    return recall

def load_dpr_json(path):
    with open(path) as f:
        data = json.load(f)

    processed = []
    for item in data:
        query   = item["question"]
        answers = item.get("answers", [])
        positives = item.get("positive_ctxs", [])[:2]
        negatives = item.get("negative_ctxs", [])[:2]

        for pos in positives:
            # grab the DPR-provided passage_id
            pid = pos.get("passage_id")
            # fall back to other fields if it ever differs
            if pid is None:
                pid = pos.get("id") or pos.get("doc_id")

            processed.append({
                "query":        query,
                "positive_id":  str(pid),
                "positive_text":pos.get("text", ""),
                # we keep negative *texts* for fine-tuning if you want
                "negatives":    [neg.get("text","") for neg in negatives],
                "answers":      answers
            })
    return processed

if __name__ == "__main__":
    # 1) init retriever
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever = DenseRetriever(
        model_name="retriever_finetuned_e5_best",
        index_path="wikipedia_faiss_index",
        metadata_path="wikipedia_metadata.jsonl",
        device=device,
        fine_tune=False
    )
    # bump probe for better recall if using IVF
    retriever.index.nprobe = 128

    # 2) load & sample test set
    test_data = load_dpr_json("downloads/data/retriever/nq-dev.json")
    random.seed(42)
    small_test = random.sample(test_data, k=1000)

    # 3) run recall@K
    for k in [1, 5, 20, 50]:
        _ = evaluate_retriever_recall(retriever, small_test, k=k)

    # 4) debug one failure
    for sample in small_test:
        retrieved = retriever.search(sample["query"], k=10)
        retrieved_ids = [ str(md.get("id")) for md, _ in retrieved ]
        if sample["positive_id"] not in retrieved_ids:
            print("\n🔍 Debugging one failure:")
            print(" Query      :", sample["query"])
            print(" Gold ID    :", sample["positive_id"])
            print(" Gold text  :", sample["positive_text"][:100], "…\n")
            print(" Retrieved (ID → score → snippet):")
            for md, score in retrieved:
                snippet = md.get("text", "").replace("\n"," ")[:80]
                print(f"  → {md.get('id')}  {score:.4f}  {snippet}…")
            break

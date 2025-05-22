import os
import torch
import json
from pipeline import RAGPipeline
from retriever import DenseRetriever
from _evaluate import evaluate_pipeline
from tqdm import tqdm
import wandb 


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FAISS_INDEX_PATH = "/local00/student/shakya/wikipedia_hnsw_index"
METADATA_PATH   = "/local00/student/shakya/wikipedia_metadata.jsonl"

wandb.init(
    project="retriever_evaluation",
    name="retriever_evaluation",
    config={
        "faiss_index_path": FAISS_INDEX_PATH,
        "metadata_path": METADATA_PATH,
        "device": str(device),
        "model_name": "models/retriever_finetuned_e5_best",
        "test_data_path": "downloads/data/retriever/nq-dev.json"
    }
)

# --- 1. Load retriever ---
retriever = DenseRetriever(
    model_name=     "models/retriever_finetuned_e5_best",
    index_path=     FAISS_INDEX_PATH,
    metadata_path=  METADATA_PATH,
    device=         device,
    fine_tune=      False,
    ef_search=1500,
    ef_construction=200,
)

# --- 2. Load test data ---
with open("downloads/data/retriever/nq-dev.json") as f:
    test_data = json.load(f)

# --- 3. Retriever evaluation helper ---
def extract_snippets(retrieved):
    if not retrieved:
        return []
    first = retrieved[0]
    # tuple format:  (metadata_dict, score_float)
    if isinstance(first, tuple) and len(first) == 2:
        return [meta["text"] for meta, _ in retrieved]
    # dict format:    {"id":…, "text":…, "score":…}
    elif isinstance(first, dict):
        return [doc["text"] for doc in retrieved]
    else:
        raise ValueError(f"Unrecognized retrieved format: {type(first)}")

def evaluate_retriever(retriever, data, k_values=[10,20,50,100], sample_size=None):
    if sample_size:
        data = data[:sample_size]
    total = len(data)
    print(f"Evaluating retriever on {total} samples")
    recalls = {}
    for k in k_values:
        hits = 0
        for sample in tqdm(data, desc=f"Evaluating for k={k}"):
            query   = sample["question"]
            answers = sample["answers"]
            retrieved = retriever.search(query, k=k)
            snippets = extract_snippets(retrieved)
            if any(
                any(ans.lower() in snippet.lower() for snippet in snippets)
                for ans in answers
            ):
                hits += 1
        recall = hits / total
        print(f"  Recall@{k}: {recall:.3f}")
        recalls[k] = recall
    return recalls

# --- 4. Run retriever evaluation ---
# evaluate_retriever(retriever, test_data, k_values=[1,5,20,50], sample_size=1000)


# ef_search_vals       = [500, 1000, 1500, 2000, 2500]
# ef_construction_vals = [100]                                        

# results = []
# for ef_s in ef_search_vals:
#     for ef_c in ef_construction_vals:
#         # 3. Tweak the live index
#         retriever.update_hnsw_params(ef_search=ef_s, ef_construction=ef_c)

#         # 4. Re-run your recall eval
#         metrics = evaluate_retriever(
#             retriever,
#             test_data,
#             k_values=[1,5,20,50],
#             sample_size=1000
#         )

#         # 5. Log & collect
#         wandb.log({
#             **{f"Recall@{k}": v for k, v in metrics.items()},
#             "ef_search":      ef_s,
#             "ef_construction": ef_c,
#         })
#         results.append({
#             "ef_search":       ef_s,
#             "ef_construction": ef_c,
#             **{f"Recall@{k}": v for k, v in metrics.items()}
#         })

# # 6. Print a summary
# print("Grid Search Results:")
# for r in results:
#     print(r)

# # --- 5. Load RAG pipeline for end-to-end testing ---
rag = RAGPipeline(
    dense_retriever=retriever,
    device=device,
    train_generator=False,
    train_retriever_end_to_end=False,
    k_retrieval_for_inference=10,
    model_name="models/generator_best"
)
rag.generator.eval()
rag.dense_retriever.query_encoder.eval()

# # # --- 6. Retrieval + Generation loop (first 100) ---
outputs = []
for sample in test_data:
    query      = sample["question"]
    references = sample["answers"]

    outputs.append({
        "query":         query,
        "references":    references,
    })

# # # --- 7. Save retrieval+generation outputs ---
# os.makedirs("outputs", exist_ok=True)
# with open("outputs/test_outputs.json", "w") as f:
#     json.dump(outputs, f, indent=2)
# print("✅ Retrieval+Generation outputs saved to outputs/test_outputs.json")

# # # --- 8. Evaluate end‐to‐end with your existing evaluator ---
eval_data = [
    {"query": out["query"], "answers": out["references"]}
    for out in outputs
]
metrics = evaluate_pipeline(
    pipeline=rag,
    test_set=eval_data,
    verbose=True,
    log_path="outputs/evaluation_log.json",
    strategy="thorough",
    top_k=50,
)
print("\n✅ End‐to‐End Evaluation Metrics:")
print(metrics)

wandb.log({
    "metrics": metrics,
})
wandb.finish()

import os
import torch
import json
from rag_pipeline import RAGPipeline
from Dense_Retriever import DenseRetriever
from evaluate import evaluate_pipeline  # Import your evaluation code

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load retriever
retriever = DenseRetriever(
    index_path="wikipedia_faiss_index",
    metadata_path="wikipedia_metadata.jsonl",
    device=device,
    fine_tune=False
)

# Load RAG pipeline
rag = RAGPipeline(dense_retriever=retriever, device=device, fine_tune=False, k=50)

rag.generator.eval()
rag.dense_retriever.model.eval()

# Load test data
with open("downloads/data/retriever/nq-dev.json") as f:
    test_data = json.load(f)

# Storage for results
outputs = []

# Process first 20 samples
for i, sample in enumerate(test_data[:50]):
    query = sample["question"]
    references = sample["answers"]

    # Retrieve top-k docs
    retrieved_docs = []
    retrieved = rag._retrieve_topk_docs(query, k=10)
    for doc, score in retrieved:
        snippet = doc
        retrieved_docs.append({
            "score": float(score),
            "snippet": snippet
        })

    # Generate answer (thorough strategy)
    prediction = rag.generate_answer(query, strategy="thorough", top_k=10)

    outputs.append({
        "query": query,
        "references": references,
        "retrieved_docs": retrieved_docs,
        "prediction": prediction
    })

# Save the results to JSON
os.makedirs("outputs", exist_ok=True)
with open("outputs/test_outputs.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("✅ Results saved to outputs/test_outputs.json")

# Now evaluate
# Prepare data in format expected by `evaluate_pipeline`
eval_data = [
    {
        "query": out["query"],
        "answers": out["references"]
    }
    for out in outputs
]

# Use your evaluate_pipeline function
metrics = evaluate_pipeline(rag, eval_data, verbose=True, log_path="outputs/evaluation_log.json", strategy="thorough", top_k=10)

print("\n✅ Evaluation Metrics:")
print(metrics)

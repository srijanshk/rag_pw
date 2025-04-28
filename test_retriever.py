from Dense_Retriever import DenseRetriever
import torch

# Set device
device = torch.device("cpu")

# Initialize retriever
retriever = DenseRetriever(
    index_path="wikipedia_faiss_index",
    metadata_path="merged_metadata.jsonl",
    device=device,
    model_name="retriever_finetuned_st",  # fine-tuned SentenceTransformer model
    fine_tune=False
)

# Sample queries to test
test_queries = [
    "when did the thrill of it all come out",
    "who is the current president of france",
    "how tall is mount everest",
    "what is the capital of japan",
]

# Run retrieval
for query in test_queries:
    print(f"\n🔎 Query: {query}")
    results = retriever.search(query, k=10)  # top-3
    for i, (meta, score) in enumerate(results):
        snippet = meta.get("text", "")[:200].replace("\n", " ")
        print(f"  {i+1}. [Score: {score:.4f}] {snippet}")

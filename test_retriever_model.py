import faiss
import numpy as np
import json

# Load index
index = faiss.read_index("wikipedia_faiss_index")

# Load metadata
metadata = []
with open("merged_metadata.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        metadata.append(json.loads(line))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer("intfloat/e5-large-v2")
query = "What happens afterlife?"

# E5 uses 'query:' prefix
query_emb = model.encode("query: " + query, normalize_embeddings=True)

top_k = 5
D, I = index.search(np.array([query_emb]), top_k)

print("🔍 Top Results:")
for rank, idx in enumerate(I[0]):
    print(f"\n#{rank+1} (Score: {D[0][rank]:.4f})")
    print(metadata[idx]['text'][:300])  # print the first 300 chars

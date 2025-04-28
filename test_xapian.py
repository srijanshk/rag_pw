from xapian_retriever import XapianRetriever

retriever = XapianRetriever("wikipedia_xapian_db")
query = "how many tires does a 18 wheeler have"

print(f"🔍 Testing Xapian search for query: '{query}'")
results = retriever.search(query, k=5)

for i, (meta, score) in enumerate(results):
    print(f"[{i+1}] Score: {score:.4f}")
    print(f"→ Title: {meta.get('title', 'N/A')}")
    print(f"→ Text: {meta.get('text', '')[:200]}...\n")

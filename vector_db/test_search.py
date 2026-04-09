from vector_db.faiss_index import load_index, load_chunks, semantic_search

index = load_index()
chunks = load_chunks()

print("\n TEST SEARCH")

query = "neural networks for language processing"

results = semantic_search(query, index, chunks)

for i, r in enumerate(results):
    print(f"\nResult {i+1}:")
    print(r[:200])

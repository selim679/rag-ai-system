from vector_db.faiss_index import *
import numpy as np

data = load_data()

# utiliser les résumés
texts = [paper["summary"] for paper in data]

print("Generating embeddings...")
embeddings = create_embeddings(texts)

print("Building FAISS index...")
index = build_faiss_index(np.array(embeddings))

save_index(index)

print("\n🔍 TEST SEARCH")
query = "neural networks for language processing"

results = semantic_search(query, index, texts)

for i, r in enumerate(results):
    print(f"\nResult {i+1}:")
    print(r[:200])

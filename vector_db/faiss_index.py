import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

def load_data(path="data_pipeline/arxiv_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_embeddings(texts):
    return MODEL.encode(texts)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def save_index(index, path="vector_db/faiss.index"):
    faiss.write_index(index, path)

def load_index(path="vector_db/faiss.index"):
    return faiss.read_index(path)

def semantic_search(query, index, texts, top_k=3):
    query_vector = MODEL.encode([query])
    distances, indices = index.search(np.array(query_vector), top_k)

    results = []
    for idx in indices[0]:
        results.append(texts[idx])

    return results

import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from utils.chunking import chunk_text

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TEXT_CHUNKS = []

# -------------------------
# CHUNK DATA
# -------------------------
def prepare_chunks(data):
    global TEXT_CHUNKS

    TEXT_CHUNKS = []

    for paper in data:
        text = paper["title"] + " " + paper["summary"]
        chunks = chunk_text(text)
        TEXT_CHUNKS.extend(chunks)

    print("TOTAL CHUNKS:", len(TEXT_CHUNKS))
    return TEXT_CHUNKS

# -------------------------
# EMBEDDINGS
# -------------------------
def create_embeddings(texts):
    embeddings = MODEL.encode(texts)
    embeddings = np.array(embeddings).astype("float32")
    faiss.normalize_L2(embeddings)
    return embeddings

# -------------------------
# FAISS INDEX
# -------------------------
def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index

# -------------------------
# SAVE INDEX
# -------------------------
def save_index(index, path="index/faiss.index"):
    faiss.write_index(index, path)

def load_index(path="index/faiss.index"):
    return faiss.read_index(path)

# -------------------------
# SAVE CHUNKS (CRITICAL FIX)
# -------------------------
def save_chunks():
    with open("index/chunks.pkl", "wb") as f:
        pickle.dump(TEXT_CHUNKS, f)

def load_chunks():
    with open("index/chunks.pkl", "rb") as f:
        return pickle.load(f)

# -------------------------
# SEARCH
# -------------------------
def semantic_search(query, index, chunks, top_k=3):

    query_vector = MODEL.encode([query]).astype("float32")
    faiss.normalize_L2(query_vector)

    distances, indices = index.search(query_vector, top_k)

    results = []

    for idx in indices[0]:
        if idx < len(chunks):
            results.append(chunks[idx])

    return results


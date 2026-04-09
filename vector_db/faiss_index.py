import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from utils.chunking import chunk_text
import os

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

TEXT_CHUNKS = []


# -------------------
# CHUNKING
# -------------------
def prepare_chunks(data):
    global TEXT_CHUNKS

    TEXT_CHUNKS = []

    for paper in data:
        text = paper["title"] + ". " + paper["summary"]
        TEXT_CHUNKS.extend(chunk_text(text))

    return TEXT_CHUNKS


# -------------------
# EMBEDDINGS
# -------------------
def create_embeddings(texts):
    emb = MODEL.encode(texts)
    emb = np.array(emb).astype("float32")
    faiss.normalize_L2(emb)
    return emb


# -------------------
# INDEX
# -------------------
def build_faiss_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


# -------------------
# SAVE / LOAD
# -------------------
def save_index(index):
    os.makedirs("vector_db/index", exist_ok=True)
    faiss.write_index(index, "vector_db/index/faiss.index")


def load_index():
    return faiss.read_index("vector_db/index/faiss.index")


def save_chunks():
    os.makedirs("vector_db/index", exist_ok=True)
    with open("vector_db/index/chunks.pkl", "wb") as f:
        pickle.dump(TEXT_CHUNKS, f)


def load_chunks():
    with open("vector_db/index/chunks.pkl", "rb") as f:
        return pickle.load(f)


# -------------------
# SEARCH
# -------------------
def semantic_search(query, index, chunks, top_k=10):

    q = MODEL.encode([query]).astype("float32")
    faiss.normalize_L2(q)

    scores, idx = index.search(q, top_k)

    return [chunks[i] for i in idx[0] if i < len(chunks)]

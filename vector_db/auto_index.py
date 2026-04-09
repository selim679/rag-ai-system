import os
from rag.data_loader import load_data
from vector_db.faiss_index import *
from vector_db.bm25_index import BM25Index

def build_all():

    print("⚡ Building index automatically...")

    data = load_data()

    chunks = prepare_chunks(data)

    embeddings = create_embeddings(chunks)
    index = build_faiss_index(embeddings)

    save_index(index)
    save_chunks()

    bm25 = BM25Index(chunks)

    print("✅ SYSTEM READY")

    return index, chunks, bm25


def load_all():

    if not os.path.exists("index/faiss.index"):
        return build_all()

    index = load_index()
    chunks = load_chunks()
    bm25 = BM25Index(chunks)

    return index, chunks, bm25

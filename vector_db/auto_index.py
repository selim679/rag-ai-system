from data_pipeline.arxiv_fetcher import fetch_arxiv
from vector_db.faiss_index import (
    prepare_chunks,
    create_embeddings,
    build_faiss_index,
    save_index,
    save_chunks
)
from vector_db.bm25_index import BM25Index
import numpy as np

def load_all():

    print("🔄 Loading dataset...")

    papers = fetch_arxiv(
        "transformer NLP attention BERT GPT machine translation",
        100
    )

    chunks = prepare_chunks(papers)

    print("📊 Creating embeddings...")
    embeddings = create_embeddings(chunks)

    print("⚙️ Building FAISS index...")
    faiss_index = build_faiss_index(np.array(embeddings))

    save_index(faiss_index)
    save_chunks()

    print("📚 Building BM25 index...")
    bm25 = BM25Index(chunks)

    return faiss_index, chunks, bm25

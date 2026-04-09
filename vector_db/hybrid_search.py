from vector_db.faiss_index import semantic_search

def hybrid_search(query, faiss_index, bm25, chunks, top_k=5):

    # FAISS
    faiss_results = semantic_search(query, faiss_index, chunks, top_k)

    # BM25
    bm25_results = bm25.search(query, top_k)

    # MERGE
    combined = list(dict.fromkeys(faiss_results + bm25_results))

    return combined[:top_k]

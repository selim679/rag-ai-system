from vector_db.faiss_index import semantic_search

def hybrid_search(queries, faiss_index, bm25, chunks, top_k=10):

    all_results = []

    for q in queries:

        # FAISS
        faiss_results = semantic_search(q, faiss_index, chunks, top_k)

        # BM25
        bm25_results = bm25.search(q, top_k)

        all_results.extend(faiss_results + bm25_results)

    # remove duplicates
    return list(dict.fromkeys(all_results))

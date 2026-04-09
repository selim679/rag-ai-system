from vector_db.faiss_index import MODEL
import numpy as np
import faiss

def hybrid_search(*, queries, faiss_index, bm25, chunks, top_k=20):

    results = []

    for q in queries:

        # -------------------------
        # FAISS SEARCH (FIXED)
        # -------------------------
        q_vec = MODEL.encode([q]).astype("float32")
        faiss.normalize_L2(q_vec)

        scores, idx = faiss_index.search(q_vec, top_k)

        faiss_results = [chunks[i] for i in idx[0] if i < len(chunks)]

        # -------------------------
        # BM25 SEARCH
        # -------------------------
        bm25_results = bm25.search(q, top_k)

        # -------------------------
        # MERGE
        # -------------------------
        results.extend(faiss_results + bm25_results)

    # -------------------------
    # DEDUPLICATION
    # -------------------------
    seen = set()
    unique = []

    for r in results:
        if r not in seen:
            unique.append(r)
            seen.add(r)

    return unique

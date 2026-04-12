from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=5):

    if not documents:
        return []

    # ensure strings only (fix your crash)
    documents = [str(d) for d in documents]

    pairs = [[query, doc] for doc in documents]

    scores = reranker.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]

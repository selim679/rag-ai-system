from sentence_transformers import CrossEncoder

# 🔥 BEST PRODUCTION MODEL
reranker = CrossEncoder("BAAI/bge-reranker-large")


def rerank(query, documents, top_k=5):

    pairs = [(query, doc) for doc in documents]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )

    return [doc for doc, _ in ranked[:top_k]]

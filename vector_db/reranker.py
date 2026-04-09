from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query, documents, top_k=5):

    pairs = []

    for doc in documents:
        if isinstance(doc, dict):
            text = doc.get("text", "")
        else:
            text = str(doc)

        if text.strip():
            pairs.append((query, text))

    scores = reranker.predict(pairs)

    ranked = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)

    return [d for d, _ in ranked[:top_k]]

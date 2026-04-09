from rank_bm25 import BM25Okapi

class BM25Index:

    def __init__(self, texts):
        self.texts = texts
        tokenized = [t.split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

    def search(self, query, top_k=5):

        tokenized_query = query.split()

        scores = self.bm25.get_scores(tokenized_query)

        ranked = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )

        return [self.texts[i] for i in ranked[:top_k]]

from rank_bm25 import BM25Okapi
import numpy as np

class BM25Index:
    def __init__(self, chunks):
        self.chunks = chunks
        self.tokenized_corpus = [c.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=10):
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [self.chunks[i] for i in top_indices]

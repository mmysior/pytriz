import logging

import numpy as np
from rank_bm25 import BM25Okapi

from .embedder import Embedder, get_embedder

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, texts: list[str], embedder: Embedder | None = None):
        self._embedder = embedder or get_embedder()
        logger.info("Building retriever with model: %s", self._embedder.model)
        self._texts = texts
        self._bm25 = BM25Okapi([t.lower().split() for t in texts])
        self._corpus_vecs: np.ndarray = self._embedder.embed(texts)

    def search(self, query: str, top_k: int = 5) -> list[int]:
        bm25_scores = np.array(self._bm25.get_scores(query.lower().split()), dtype=np.float32)
        query_vec = self._embedder.embed([query])[0]
        dense_scores = (self._corpus_vecs @ query_vec).astype(np.float32)
        fused = self._rrf([bm25_scores, dense_scores])
        return np.argsort(fused)[::-1][:top_k].tolist()

    @staticmethod
    def _rrf(rankings: list[np.ndarray], k: int = 60) -> np.ndarray:
        n = len(rankings[0])
        scores = np.zeros(n, dtype=np.float32)
        for ranking in rankings:
            for rank, idx in enumerate(np.argsort(ranking)[::-1]):
                scores[idx] += 1.0 / (k + rank + 1)
        return scores

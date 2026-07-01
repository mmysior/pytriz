import logging

import numpy as np
from pydantic_ai import Embedder
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class Retriever:
    """BM25 lexical search, optionally fused with dense embeddings if an `Embedder` is provided."""

    def __init__(self, texts: list[str], embedder: Embedder | None = None):
        self._embedder = embedder
        self._texts = texts
        self._bm25 = BM25Okapi([t.lower().split() for t in texts])
        self._corpus_vecs: np.ndarray | None = None
        if embedder is not None:
            logger.info("Retriever will use embedder: %s", embedder.model)

    async def ensure_index(self) -> np.ndarray:
        """Compute and cache corpus embeddings if not already cached. Safe to call ahead of time to avoid paying this cost on first search."""
        assert self._embedder is not None
        if self._corpus_vecs is None:
            result = await self._embedder.embed_documents(self._texts)
            self._corpus_vecs = np.array(result.embeddings, dtype=np.float32)
        return self._corpus_vecs

    async def search(self, query: str, top_k: int = 5) -> list[int]:
        bm25_scores = np.array(self._bm25.get_scores(query.lower().split()), dtype=np.float32)
        if self._embedder is None:
            return np.argsort(bm25_scores)[::-1][:top_k].tolist()

        corpus_vecs = await self.ensure_index()
        result = await self._embedder.embed_query(query)
        query_vec = np.array(result.embeddings[0], dtype=np.float32)
        dense_scores = (corpus_vecs @ query_vec).astype(np.float32)
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

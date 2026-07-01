from pytriz.core.retriever import Retriever

TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Segmentation splits an object into independent parts",
    "Weight of a moving object affects its dynamics",
    "Taking away or removing a harmful part of an object",
]


async def test_search_returns_top_k_indices():
    retriever = Retriever(TEXTS)
    results = await retriever.search("object", top_k=2)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(i, int) for i in results)


async def test_search_ranks_best_lexical_match_first():
    retriever = Retriever(TEXTS)
    results = await retriever.search("segmentation independent parts", top_k=1)
    assert results[0] == 1


async def test_search_top_k_larger_than_corpus_returns_all():
    retriever = Retriever(TEXTS)
    results = await retriever.search("object", top_k=100)
    assert len(results) == len(TEXTS)


async def test_search_no_embedder_never_computes_corpus_vecs():
    retriever = Retriever(TEXTS)
    await retriever.search("object")
    assert retriever._corpus_vecs is None

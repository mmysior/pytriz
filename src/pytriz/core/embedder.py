import logging
from functools import wraps
from typing import Any, Callable

import httpx
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedder class
# ---------------------------------------------------------------------------


class Embedder:
    """Wraps an embedding provider with model name and vector size."""

    def __init__(self, model: str, vector_size: int, embed_fn: Callable[[list[str]], np.ndarray]):
        self.model = model
        self.vector_size = vector_size
        self._embed_fn = embed_fn

    def embed(self, texts: list[str]) -> np.ndarray:
        return self._embed_fn(texts)


# ---------------------------------------------------------------------------
# Embedding provider registry
# ---------------------------------------------------------------------------

type EmbedderFactoryFn = Callable[..., Embedder]

embed_providers: dict[str, EmbedderFactoryFn] = {}


def register_embedding_provider(name: str):
    def decorator(func: EmbedderFactoryFn):
        @wraps(func)
        def wrapper(model: str, **kwargs: Any) -> Embedder:
            return func(model, **kwargs)

        embed_providers[name] = wrapper
        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@register_embedding_provider("huggingface")
def get_huggingface_embedder(model: str) -> Embedder:
    encoder = SentenceTransformer(model, device="cpu")
    vector_size = encoder.get_embedding_dimension()
    if not vector_size:
        logger.debug("Probing local embedding model to determine vector size...")
        probe = encoder.encode(["dim_probe"], convert_to_numpy=True)
        vector_size = probe.shape[1]

    logger.debug("Initialized local embedder with model '%s' and vector size %d", model, vector_size)

    def embed(texts: list[str]) -> np.ndarray:
        return encoder.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    return Embedder(model=model, vector_size=vector_size, embed_fn=embed)


@register_embedding_provider("ollama")
def get_ollama_embedder(model: str, *, url: str | None = None) -> Embedder:
    base_url = (url or config.OLLAMA_BASE_URL).rstrip("/")

    def embed(texts: list[str]) -> np.ndarray:
        response = httpx.post(
            f"{base_url}/api/embed",
            json={"model": model, "input": texts},
            timeout=60.0,
        )
        response.raise_for_status()
        return np.array(response.json()["embeddings"])

    logger.debug("Probing Ollama embedding model to determine vector size...")
    probe = embed(["dim_probe"])
    vector_size = probe.shape[1]

    logger.debug("Initialized Ollama embedder with model '%s' and vector size %d", model, vector_size)
    return Embedder(model=model, vector_size=vector_size, embed_fn=embed)


@register_embedding_provider("openai")
def get_openai_embedder(model: str) -> Embedder:
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")

    def embed(texts: list[str]) -> np.ndarray:
        response = httpx.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {config.OPENAI_API_KEY}"},
            json={"input": texts, "model": model},
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()["data"]
        sorted_embeddings = sorted(data, key=lambda x: x["index"])
        return np.array([item["embedding"] for item in sorted_embeddings])

    # Probe once to discover vector size
    logger.debug("Probing OpenAI embedding model to determine vector size...")
    probe = embed(["dim_probe"])
    vector_size = probe.shape[1]

    logger.debug("Initialized OpenAI embedding model vector size: %d", vector_size)
    return Embedder(model=model, vector_size=vector_size, embed_fn=embed)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_embedder(
    provider: str | None = None,
    model: str | None = None,
    *,
    url: str | None = None,
) -> Embedder:
    provider = provider or config.EMBEDDING_PROVIDER
    model = model or config.EMBEDDING_MODEL
    factory = embed_providers.get(provider)
    if factory is None:
        raise ValueError(f"Unsupported embedding provider: {provider}")
    logger.info("Using %s embeddings: %s", provider, model)
    return factory(model, url=url)

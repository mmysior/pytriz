from pydantic_ai import Embedder
from pydantic_ai.embeddings.openai import OpenAIEmbeddingModel
from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_model(model: str, *, base_url: str | None = None, api_key: str | None = None) -> Model:
    """Build a pydantic-ai `Model`.

    - Pass a pydantic-ai model string (e.g. `"provider:model"`)
      to use one of its built-in providers directly — pydantic-ai reads the matching
      API key from the environment.
    - Pass `base_url` (and optionally `api_key`) to target any OpenAI-compatible chat
      endpoint — Ollama, LM Studio, vLLM, Together, etc. — with a bare model name
      (e.g. `get_model("llama3", base_url="http://localhost:11434/v1", api_key="your-api-key")`).

    For anything else pydantic-ai supports, construct a `pydantic_ai.models.Model`
    directly and pass it wherever a model is expected.
    """
    if base_url is not None:
        return OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key))
    return infer_model(model)


def get_embedder(model: str, *, base_url: str | None = None, api_key: str | None = None) -> Embedder:
    """Build a pydantic-ai `Embedder`.

    - Pass a pydantic-ai model string (e.g. `"openai:text-embedding-3-small"`,
      `"cohere:embed-v4.0"`) to use one of its built-in providers directly.
    - Pass `base_url` (and optionally `api_key`) to target any OpenAI-compatible
      embeddings endpoint — Ollama, LM Studio, vLLM, etc. — with a bare model name
      (e.g. `get_embedder("nomic-embed-text", base_url="http://localhost:11434/v1")`).

    For anything else pydantic-ai supports (Bedrock, Google, VoyageAI, local
    sentence-transformers, or a custom `EmbeddingModel`), construct a
    `pydantic_ai.Embedder` directly and pass it wherever an `Embedder` is expected.
    """
    if base_url is not None:
        provider = OpenAIProvider(base_url=base_url, api_key=api_key)
        return Embedder(OpenAIEmbeddingModel(model, provider=provider))
    return Embedder(model)

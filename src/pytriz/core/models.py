from pydantic_ai.models import Model, infer_model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider


def get_model(model: str, *, base_url: str | None = None, api_key: str | None = None) -> Model:
    """Build a pydantic-ai `Model`.

    - Pass a pydantic-ai model string (e.g. `"anthropic:claude-sonnet-4-6"`,
      `"openrouter:qwen/qwen3.6-35b-a3b"`) to use one of its built-in providers directly —
      pydantic-ai reads the matching API key from the environment.
    - Pass `base_url` (and optionally `api_key`) to target any OpenAI-compatible chat
      endpoint — Ollama, LM Studio, vLLM, Together, etc. — with a bare model name
      (e.g. `get_model("llama3", base_url="http://localhost:11434/v1")`).

    For anything else pydantic-ai supports, construct a `pydantic_ai.models.Model`
    directly and pass it wherever a model is expected.
    """
    if base_url is not None:
        return OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key))
    return infer_model(model)

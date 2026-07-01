# PyTRIZ

A Python library for applying TRIZ (Theory of Inventive Problem Solving) — look up parameters, principles, and contradiction matrix results, or use LLM-powered agents to analyze engineering trade-offs.

## Installation

```bash
pip install pytriz
```

Requires Python 3.12+.

## Quick start

```python
import asyncio
from pytriz import TRIZStore

store = TRIZStore()  # lexical (BM25) search only, zero config, no API key

# Find TRIZ principles for a contradiction
principles = store.get_principles_from_matrix(
    improving_parameters=[1, 3],
    preserving_parameters=[17, 23],
)

# Search parameters and principles by description
params = asyncio.run(store.search_parameters("improves durability", top_k=5))
principles = asyncio.run(store.search_principles("segmentation", top_k=5))
```

`TRIZStore` builds the BM25 lexical index on instantiation — create it once and reuse it across your application. Passing an `embed_model` (see below) fuses in dense semantic search on top; embeddings are computed lazily on first search and cached from then on, or you can precompute them upfront with `await store.ensure_index()`.

## Semantic search

PyTRIZ doesn't bundle an embedding runtime — it accepts any [pydantic-ai](https://ai.pydantic.dev/) `Embedder`, so you bring whichever backend you want.

```python
import asyncio
from pytriz import TRIZStore, get_embedder

# Any OpenAI-compatible endpoint — Ollama, LM Studio, vLLM, etc.
embedder = get_embedder("nomic-embed-text", base_url="http://localhost:11434/v1")

# Or one of pydantic-ai's built-in providers directly
# embedder = get_embedder("openai:text-embedding-3-small")

store = TRIZStore(embed_model=embedder)
results = asyncio.run(store.search_principles("segmentation", top_k=5))
```

For providers pydantic-ai supports natively (OpenAI, Cohere, VoyageAI, Bedrock, Google, or local `sentence-transformers`), construct a `pydantic_ai.Embedder` yourself and pass it in the same way:

```python
from pydantic_ai import Embedder

store = TRIZStore(embed_model=Embedder("cohere:embed-v4.0"))
```

## LLM-powered analysis

PyTRIZ doesn't read any environment variables or config of its own — every LLM call needs an explicit `llm`, just like `embed_model`.

```python
import asyncio
from pytriz import TRIZStore, get_model
from pytriz import contradictions

store = TRIZStore()

# Bare model name + base_url -> any OpenAI-compatible endpoint (Ollama, LM Studio, vLLM, Together, etc.)
llm = get_model("llama3", base_url="http://localhost:11434/v1")

# Or one of pydantic-ai's built-in providers directly (reads the matching API key from the environment)
# llm = get_model("anthropic:claude-sonnet-4-6")

result = asyncio.run(
    contradictions.analyze_contradiction(
        "Increasing blade thickness improves durability but increases weight.",
        store=store,
        llm=llm,
    )
)

print(result.contradiction)
print(result.improving_parameter)
print(result.preserving_parameter)
```

For providers pydantic-ai supports natively but `get_model` doesn't wrap (Cohere, Bedrock, Google, etc.), construct a `pydantic_ai.models.Model` yourself and pass it in the same way.

## Explicit configuration

For full control — useful when building FastAPI apps, MCP servers, or any long-running service:

```python
from pytriz import TRIZStore, get_embedder, get_model
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.settings import ModelSettings

store = TRIZStore(
    embed_model=get_embedder("nomic-embed-text", base_url="http://my-server:11434/v1"),
)
await store.ensure_index()  # precompute embeddings at startup instead of on the first search

llm = AnthropicModel(
    "claude-sonnet-4-6",
    provider=AnthropicProvider(api_key="your-key-here"),
    settings=ModelSettings(temperature=0.2),
)

result = await contradictions.analyze_contradiction(
    "Increasing blade thickness improves durability but increases weight.",
    store=store,
    llm=llm,
)
```

`store` and `llm` are independent — configure each separately and pass them where needed.

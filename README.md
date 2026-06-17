# PyTRIZ

A Python library for applying TRIZ (Theory of Inventive Problem Solving) — look up parameters, principles, and contradiction matrix results, or use LLM-powered agents to analyze engineering trade-offs.

## Installation

```bash
pip install pytriz
```

Requires Python 3.12+.

## Quick start

```python
from pytriz import TRIZStore

store = TRIZStore()  # uses default embedding model, reads config from env

# Find TRIZ principles for a contradiction
principles = store.get_principles_from_matrix(
    improving_parameters=[1, 3],
    preserving_parameters=[17, 23],
)

# Search parameters and principles by description
params = store.search_parameters("improves durability", top_k=5)
principles = store.search_principles("segmentation", top_k=5)
```

`TRIZStore` builds the indexed corpus on instantiation — create it once and reuse it across your application. Semantic search uses a local embedding model by default, no API key needed.

## LLM-powered analysis

```python
import asyncio
from pytriz import TRIZStore
from pytriz import contradictions

store = TRIZStore()

result = asyncio.run(
    contradictions.analyze_contradiction(
        "Increasing blade thickness improves durability but increases weight.",
        store=store,
    )
)

print(result.contradiction)
print(result.improving_parameter)
print(result.preserving_parameter)
```

Set your LLM provider in a `.env` file:

```env
DEFAULT_PROVIDER=openrouter    # openai | anthropic | mistral | openrouter | ollama | together
DEFAULT_MODEL=qwen/qwen3.6-35b-a3b
OPENROUTER_API_KEY=your-key-here
```

## Explicit configuration

For full control — useful when building FastAPI apps, MCP servers, or any long-running service:

```python
from pytriz import TRIZStore, get_embedder, get_model, ModelSettings

store = TRIZStore(
    embed_model=get_embedder(provider="ollama", model="nomic-embed-text", url="http://my-server:11434"),
)
llm = get_model(
    provider="anthropic",
    model_name="claude-sonnet-4-6",
    settings=ModelSettings(temperature=0.2),
)

result = await contradictions.analyze_contradiction(
    "Increasing blade thickness improves durability but increases weight.",
    store=store,
    llm=llm,
)
```

`store` and `llm` are independent — configure each separately and pass them where needed.

## Embedding providers

PyTRIZ supports three embedding backends, configured via `EMBEDDING_PROVIDER`:

### HuggingFace (default)

Downloads and runs the model locally using `sentence-transformers`. No API key needed for public models.

```env
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

For gated models (e.g. `google/embeddinggemma-300m`), set your HuggingFace token:

```env
HF_TOKEN=your-token-here
EMBEDDING_MODEL=google/embeddinggemma-300m
```

### Ollama

Runs embeddings locally via a running [Ollama](https://ollama.com) instance.

```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_MODEL=nomic-embed-text
OLLAMA_BASE_URL=http://localhost:11434   # optional, this is the default
```

### OpenAI

```env
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
OPENAI_API_KEY=your-key-here
```

## LLM providers

| Provider | `DEFAULT_PROVIDER` | Required env var |
|---|---|---|
| OpenAI | `openai` | `OPENAI_API_KEY` |
| Anthropic | `anthropic` | `ANTHROPIC_API_KEY` |
| Mistral | `mistral` | `MISTRAL_API_KEY` |
| OpenRouter | `openrouter` | `OPENROUTER_API_KEY` |
| Together | `together` | `TOGETHER_API_KEY` |
| Ollama | `ollama` | — (uses `OLLAMA_BASE_URL`) |

## Environment variables reference

| Variable | Default | Description |
|---|---|---|
| `DEFAULT_PROVIDER` | `openrouter` | LLM provider |
| `DEFAULT_MODEL` | `qwen/qwen3.6-35b-a3b` | LLM model name |
| `EMBEDDING_PROVIDER` | `huggingface` | Embedding backend |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama host (used for both LLM and embeddings) |
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `MISTRAL_API_KEY` | — | Mistral API key |
| `OPENROUTER_API_KEY` | — | OpenRouter API key |
| `TOGETHER_API_KEY` | — | Together AI API key |
| `HF_TOKEN` | — | HuggingFace token (for gated models) |

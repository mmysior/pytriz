# PyTRIZ

A Python library for applying TRIZ (Theory of Inventive Problem Solving) — look up parameters, principles, and contradiction matrix results, or use LLM-powered agents to analyze engineering trade-offs.

## Installation

```bash
pip install pytriz
```

Requires Python 3.12+.

## Quick start

### Data lookups (no API key needed)

```python
from pytriz import contradictions

# Find TRIZ principles for a contradiction
principles = contradictions.get_principles_from_matrix(
    improving_parameters=[1, 3],
    preserving_parameters=[17, 23],
)

# Search parameters and principles by description
params = contradictions.search_parameters("improves durability", top_k=5)
principles = contradictions.search_principles("segmentation", top_k=5)
```

Semantic search uses a local embedding model by default — no API key or internet connection required after the first run (the model is cached automatically).

### LLM-powered analysis

```python
import asyncio
from pytriz import contradictions

result = asyncio.run(
    contradictions.analyze_contradiction(
        "Increasing blade thickness improves durability but increases weight.",
    )
)

print(result.contradiction)
print(result.improving_parameter)
print(result.preserving_parameter)
```

Set your API key and preferred provider in a `.env` file:

```env
DEFAULT_PROVIDER=openrouter    # openai | anthropic | mistral | openrouter | ollama | together
DEFAULT_MODEL=qwen/qwen3.6-35b-a3b
OPENROUTER_API_KEY=your-key-here
```

## Explicit configuration

Pass `llm` and `embed_model` objects directly for full control — useful when building applications or MCP servers on top of PyTRIZ. When not provided, PyTRIZ falls back to environment variables.

### LLM

```python
from pytriz import get_model, ModelSettings

# basic
llm = get_model(provider="anthropic", model_name="claude-sonnet-4-6")

# with model settings
llm = get_model(
    provider="anthropic",
    model_name="claude-sonnet-4-6",
    settings=ModelSettings(temperature=0.2),
)
```

Then pass to any LLM function:

```python
result = await contradictions.analyze_contradiction("my problem", llm=llm)
result = await contradictions.extract_tcs("my description", llm=llm)
```

### Embeddings

```python
from pytriz import get_embedder

embed_model = get_embedder(provider="huggingface", model="sentence-transformers/all-MiniLM-L6-v2")
embed_model = get_embedder(provider="ollama", model="nomic-embed-text")
embed_model = get_embedder(provider="ollama", model="nomic-embed-text", url="http://my-server:11434")
embed_model = get_embedder(provider="openai", model="text-embedding-3-small")
```

Then pass to any search function:

```python
results = contradictions.search_principles("reduce friction", embed_model=embed_model)
results = contradictions.search_parameters("improve durability", embed_model=embed_model)
```

### Fully explicit (no env vars needed)

```python
import asyncio
from pytriz import contradictions, get_embedder, get_model, ModelSettings

llm = get_model(provider="anthropic", model_name="claude-sonnet-4-6", settings=ModelSettings(temperature=0.2))
embed_model = get_embedder(provider="ollama", model="nomic-embed-text", url="http://my-server:11434")

result = asyncio.run(
    contradictions.analyze_contradiction(
        "Increasing blade thickness improves durability but increases weight.",
        llm=llm,
        embed_model=embed_model,
    )
)

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

All providers are configured via environment variables:

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

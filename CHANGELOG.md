# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-06-17

### Added

- GitHub Actions workflows for automatic GitHub release creation and PyPI publishing on `v*` tag push
- `TRIZStore` — the main entry point for indexed TRIZ corpus access. Instantiate once with an optional `embed_model`, reuse across your application (FastAPI, FastMCP, scripts)
- Ollama embedding provider (`EMBEDDING_PROVIDER=ollama`) with configurable `OLLAMA_BASE_URL`
- `OLLAMA_BASE_URL` config variable (default: `http://localhost:11434`), used by both the Ollama LLM and embedding providers
- `llm: LLModel` kwarg on all LLM functions for explicit model configuration
- `settings: ModelSettings` kwarg on `get_model` for controlling temperature, max tokens, etc.
- `url` keyword-only kwarg on `get_embedder` and `get_model` for Ollama host override
- `TRIZStore`, `get_model`, `get_embedder`, `Embedder`, `LLModel`, `ModelSettings` exported from top-level `pytriz`

### Fixed

- `get_embedder` and `get_model` now forward `url` only when `provider == "ollama"` — non-Ollama factories no longer receive an unexpected `url` kwarg
- `url: str | None = None` removed from non-Ollama model factory signatures (`openai`, `anthropic`, `together`, `mistral`, `openrouter`)
- `tests/test_contradictions.py` updated to use `TRIZStore` directly, matching the breaking change introduced in this release

### Changed

- **Breaking:** Retrieval functions (`search_parameters`, `search_principles`, `get_all_parameters`, etc.) moved to `TRIZStore` — no longer available as module-level functions
- **Breaking:** `analyze_contradiction` and `classify_principle` now require `store: TRIZStore` as a keyword argument
- **Breaking:** LLM functions replace `provider`/`model` string kwargs with a single `llm: LLModel | None` kwarg
- **Breaking:** Embedding provider `local` renamed to `huggingface` (`EMBEDDING_PROVIDER=huggingface`)
- Default embedding model changed from `google/embeddinggemma-300m` (gated) to `sentence-transformers/all-MiniLM-L6-v2` (public)
- Default LLM provider changed from `openai` (`gpt-4.1`) to `openrouter` (`qwen/qwen3.6-35b-a3b`)
- `core/providers.py` renamed to `core/models.py`

## [0.1.0] - 2026-05-29

### Added

- TRIZ contradiction matrix lookup via `get_principles_from_matrix`
- Parameter and principle data (39 parameters, 40 principles) with semantic search powered by `sentence-transformers`
- LLM-powered functions: `extract_tcs`, `formulate_tc`, `generate_solution`, `analyze_contradiction`, `classify_principle`
- Multi-provider support: OpenAI, Anthropic, Mistral, Together, Ollama, OpenRouter
- Jinja2-based prompt templating system
- Pydantic schemas for all domain models (`Parameter`, `Principle`, `TCModel`, `Contradictions`, `ContradictionResult`)
- Configuration via environment variables or `.env` file using `pydantic-settings`
- `py.typed` marker for downstream type checker support
- LLM functions accept `provider` and `model` as string arguments
- Modern type annotations throughout (`list`, `set`, `X | None` instead of `typing` generics)

[unreleased]: https://github.com/mmysior/pytriz/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mmysior/pytriz/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/mmysior/pytriz/releases/tag/v0.1.0

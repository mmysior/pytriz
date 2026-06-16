# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] - 2026-06-16

### Added

- Ollama embedding provider (`EMBEDDING_PROVIDER=ollama`) with configurable `OLLAMA_BASE_URL`
- `OLLAMA_BASE_URL` config variable (default: `http://localhost:11434`), now also used by the Ollama LLM provider
- `embed_model` kwarg (`Embedder` instance) on `search_parameters`, `search_principles`, `analyze_contradiction`, and `classify_principle` for explicit embedding config without env vars
- `llm` kwarg (`LLModel` instance) on all LLM functions replacing `provider`/`model` string kwargs
- `get_model`, `get_embedder`, `LLModel`, `Embedder` exported from top-level `pytriz` package

### Changed

- Default LLM provider changed from `openai` (`gpt-4.1`) to `openrouter` (`qwen/qwen3.6-35b-a3b`)
- Embedding provider `local` renamed to `huggingface` (`EMBEDDING_PROVIDER=huggingface`)
- **Breaking:** LLM functions (`extract_tcs`, `formulate_tc`, `generate_solution`, `analyze_contradiction`, `classify_principle`) replace `provider`/`model` string kwargs with a single `llm: LLModel | None` kwarg

### Fixed

- Changed default embedding model from `google/embeddinggemma-300m` (gated, requires HF authentication) to `sentence-transformers/all-MiniLM-L6-v2` (public, no auth required)

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

[unreleased]: https://github.com/mmysior/pytriz/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/mmysior/pytriz/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/mmysior/pytriz/releases/tag/v0.1.0

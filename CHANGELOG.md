# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[unreleased]: https://github.com/mmysior/pytriz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mmysior/pytriz/releases/tag/v0.1.0

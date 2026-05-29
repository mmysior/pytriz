# PyTRIZ

A Python library for applying TRIZ (Theory of Inventive Problem Solving) — look up parameters, principles, and contradiction matrix results, or use LLM-powered agents to analyze engineering trade-offs.

## Installation

```bash
pip install pytriz
```

Requires Python 3.12+. LLM features require an API key for your chosen provider (OpenAI, Anthropic, Mistral, or OpenRouter).

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

### LLM-powered analysis

```python
import asyncio
from pytriz import contradictions

result = asyncio.run(
    contradictions.analyze_contradiction(
        "Increasing blade thickness improves durability but increases weight.",
        provider="openai",   # or "anthropic", "mistral", "openrouter"
        model="gpt-4.1",
    )
)

print(result.contradiction)
print(result.improving_parameter)
print(result.preserving_parameter)
```

Set your API key in a `.env` file or as an environment variable:

```
OPENAI_API_KEY=your-key-here
```

import csv
import importlib.resources as pkg_resources
import json
import logging
import random
from functools import lru_cache
from itertools import product

import numpy as np
from pydantic_ai import Agent

from .core.config import config
from .core.embedder import Embedder
from .core.models import LLModel, get_model
from .core.retriever import Retriever
from .prompts import get_prompt
from .schemas.contradictions import (
    ContradictionResult,
    Contradictions,
    Parameter,
    ParameterPairSelection,
    Principle,
    PrincipleSelection,
    TCModel,
)

logger = logging.getLogger(__name__)


# ======================================================================================================================
# Data Loading
# ======================================================================================================================


@lru_cache(maxsize=1)
def _load_parameters() -> list[Parameter]:
    ref = pkg_resources.files("pytriz.resources") / "parameters.json"
    with pkg_resources.as_file(ref) as path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    return [Parameter(**p) for p in data["parameters"]]


@lru_cache(maxsize=1)
def _load_principles() -> list[Principle]:
    ref = pkg_resources.files("pytriz.resources") / "principles.json"
    with pkg_resources.as_file(ref) as path:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    items = data if isinstance(data, list) else data["principles"]
    return [
        Principle(
            id=int(p["id"]),
            name=str(p["name"]),
            description=str(p.get("description", "")),
            rules=list(p.get("rules", [])),
            hints=list(p.get("hints", [])),
            examples=list(p.get("examples", [])),
        )
        for p in items
    ]


@lru_cache(maxsize=1)
def _load_matrix() -> np.ndarray:
    ref = pkg_resources.files("pytriz.resources") / "matrix_values.csv"
    with pkg_resources.as_file(ref) as path:
        with open(path, "r", encoding="utf-8") as f:
            matrix_data = list(csv.reader(f, delimiter=";"))
    matrix = np.array(matrix_data, dtype=object)
    logger.debug("Loaded TRIZ matrix with shape %dx%d", *matrix.shape)
    return matrix


# ======================================================================================================================
# Parameters & Principles
# ======================================================================================================================


@lru_cache(maxsize=1)
def _get_param_retriever() -> Retriever:
    return Retriever([p.text for p in _load_parameters()])


@lru_cache(maxsize=1)
def _get_principle_retriever() -> Retriever:
    return Retriever([p.text for p in _load_principles()])


def get_all_parameters() -> list[Parameter]:
    return _load_parameters()


def get_parameter_by_id(parameter_id: int) -> Parameter:
    parameter = next((p for p in _load_parameters() if p.id == parameter_id), None)
    if not parameter:
        raise ValueError(f"Parameter with id {parameter_id} not found")
    return parameter


def search_parameters(query: str, top_k: int = 5, *, embed_model: Embedder | None = None) -> list[Parameter]:
    params = _load_parameters()
    retriever = Retriever([p.text for p in params], embed_model) if embed_model is not None else _get_param_retriever()
    indices = retriever.search(query, top_k)
    return [params[i] for i in indices]


def get_all_principles() -> list[Principle]:
    return _load_principles()


def get_principle_by_id(principle_id: int) -> Principle:
    principle = next((p for p in _load_principles() if p.id == principle_id), None)
    if not principle:
        raise ValueError(f"Principle with id {principle_id} not found")
    return principle


def get_principle_by_name(principle_name: str) -> Principle:
    principle = next((p for p in _load_principles() if p.name.lower() == principle_name.lower()), None)
    if not principle:
        raise ValueError(f"Principle with name '{principle_name}' not found")
    return principle


def search_principles(query: str, top_k: int = 5, *, embed_model: Embedder | None = None) -> list[Principle]:
    principles = _load_principles()
    retriever = (
        Retriever([p.text for p in principles], embed_model) if embed_model is not None else _get_principle_retriever()
    )
    indices = retriever.search(query, top_k)
    return [principles[i] for i in indices]


def get_random_principles(count: int = 4) -> list[Principle]:
    all_principles = get_all_principles()
    if count >= len(all_principles):
        return all_principles
    return random.sample(all_principles, count)


# ======================================================================================================================
# Contradiction Matrix
# ======================================================================================================================


def get_principles_from_matrix(improving_parameters: list[int], preserving_parameters: list[int]) -> list[Principle]:
    if not all(isinstance(x, int) for x in improving_parameters + preserving_parameters):
        raise TypeError("All parameter IDs must be integers")
    if not all(x > 0 for x in improving_parameters + preserving_parameters):
        raise ValueError("All parameter IDs must be positive integers")

    matrix = _load_matrix()
    all_principles = get_all_principles()
    row_indices = [i - 1 for i in improving_parameters]
    col_indices = [i - 1 for i in preserving_parameters]

    principle_ids: set[int] = set()
    for row, col in product(row_indices, col_indices):
        if row == col or row >= matrix.shape[0] or col >= matrix.shape[1]:
            continue
        cell_value = matrix[row, col]
        if cell_value and cell_value != "":
            for p in cell_value.split(","):
                try:
                    principle_ids.add(int(p.strip()))
                except (ValueError, AttributeError):
                    continue

    return [p for p in all_principles if p.id in sorted(principle_ids)]


# ======================================================================================================================
# LLM Generators
# ======================================================================================================================


def format_principle(principle: Principle) -> str:
    formatted = f"Principle {principle.id}: {principle.name}\n\n{principle.description}\n\n"
    if principle.rules:
        formatted += "Rules:\n" + "\n".join(f"  - {r}" for r in principle.rules) + "\n"
    if principle.hints:
        formatted += "Hints:\n" + "\n".join(f"  - {h}" for h in principle.hints) + "\n"
    if principle.examples:
        formatted += "Examples:\n" + "\n".join(f"  - {e}" for e in principle.examples) + "\n"
    return formatted


def _resolve_llm(llm: LLModel | None) -> LLModel:
    return llm or get_model(config.DEFAULT_PROVIDER, config.DEFAULT_MODEL)


async def extract_tcs(description: str, *, llm: LLModel | None = None) -> Contradictions:
    agent = Agent(
        model=_resolve_llm(llm),
        output_type=Contradictions,
        system_prompt=get_prompt("extract_tc_from_text").compile(),
    )
    try:
        response = await agent.run(description)
        logger.info(f"Extracted {len(response.output.contradictions)} contradictions")
        return response.output
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise ValueError(f"Error getting a response from the model: {e}")


async def formulate_tc(trade_off: str, *, context: str | None = None, llm: LLModel | None = None) -> TCModel:
    agent = Agent(
        model=_resolve_llm(llm),
        output_type=TCModel,
        system_prompt=get_prompt("formulate_tc").compile(context=context),
    )
    try:
        response = await agent.run(trade_off)
        logger.info(f"Formulated TC: {response.output}")
        return response.output
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise ValueError(f"Error getting a response from the model: {e}")


async def generate_solution(
    trade_off: str,
    principle: Principle,
    *,
    context: str | None = None,
    llm: LLModel | None = None,
) -> str:
    agent = Agent(
        model=_resolve_llm(llm),
        output_type=str,
        system_prompt=get_prompt("generate_solution").compile(
            context=context,
            inventive_principle=format_principle(principle),
        ),
    )
    try:
        response = await agent.run(trade_off)
        logger.info("Generated solution description")
        return response.output
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise ValueError(f"Error getting a response from the model: {e}")


async def analyze_contradiction(
    problem_summary: str,
    *,
    llm: LLModel | None = None,
    embed_model: Embedder | None = None,
    retrieve_k: int = 5,
) -> ContradictionResult:
    tc = await formulate_tc(problem_summary, llm=llm)

    seen: set[int] = set()
    candidates: list[Parameter] = []
    for effect in (tc.positive_effect, tc.negative_effect):
        for p in search_parameters(effect, retrieve_k, embed_model=embed_model):
            if p.id not in seen:
                seen.add(p.id)
                candidates.append(p)

    if len(candidates) < 2:
        raise ValueError(f"Not enough parameter candidates found for contradiction: {tc}")

    agent = Agent(
        model=_resolve_llm(llm),
        output_type=ParameterPairSelection,
        system_prompt=get_prompt("rerank_parameters").compile(
            action=tc.action,
            positive_effect=tc.positive_effect,
            negative_effect=tc.negative_effect,
            candidates="\n\n".join(f"ID {p.id}: {p.text}" for p in candidates),
        ),
    )
    try:
        result = await agent.run(problem_summary)
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise ValueError(f"Error getting a response from the model: {e}")

    improving = next((p for p in candidates if p.id == result.output.improving_id), None)
    if improving is None:
        logger.warning(
            "Model returned unknown improving_id=%d, falling back to candidates[0]", result.output.improving_id
        )
        improving = candidates[0]

    preserving = next((p for p in candidates if p.id == result.output.preserving_id and p.id != improving.id), None)
    if preserving is None:
        fallback = next((p for p in candidates if p.id != improving.id), improving)
        logger.warning(
            "Model returned unknown preserving_id=%d, falling back to parameter id=%d",
            result.output.preserving_id,
            fallback.id,
        )
        preserving = fallback

    return ContradictionResult(contradiction=tc, improving_parameter=improving, preserving_parameter=preserving)


async def classify_principle(
    solution_summary: str,
    *,
    llm: LLModel | None = None,
    embed_model: Embedder | None = None,
    retrieve_k: int = 5,
) -> Principle:
    candidates = search_principles(solution_summary, retrieve_k, embed_model=embed_model)
    if not candidates:
        raise ValueError("No principle candidates found for solution summary.")

    agent = Agent(
        model=_resolve_llm(llm),
        output_type=PrincipleSelection,
        system_prompt=get_prompt("rerank_principle").compile(
            candidates="\n\n".join(f"ID {p.id}: {p.text}" for p in candidates),
        ),
    )
    try:
        result = await agent.run(solution_summary)
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise ValueError(f"Error getting a response from the model: {e}")

    principle = next((p for p in candidates if p.id == result.output.id), None)
    if principle is None:
        logger.warning("Model returned unknown principle id=%d, falling back to candidates[0]", result.output.id)
        principle = candidates[0]
    return principle

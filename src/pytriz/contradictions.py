import logging

from pydantic_ai import Agent
from pydantic_ai.models import Model

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
from .store import TRIZStore

logger = logging.getLogger(__name__)


def format_principle(principle: Principle) -> str:
    formatted = f"Principle {principle.id}: {principle.name}\n\n{principle.description}\n\n"
    if principle.rules:
        formatted += "Rules:\n" + "\n".join(f"  - {r}" for r in principle.rules) + "\n"
    if principle.hints:
        formatted += "Hints:\n" + "\n".join(f"  - {h}" for h in principle.hints) + "\n"
    if principle.examples:
        formatted += "Examples:\n" + "\n".join(f"  - {e}" for e in principle.examples) + "\n"
    return formatted


async def extract_tcs(description: str, *, llm: Model | str) -> Contradictions:
    agent = Agent(
        model=llm,
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


async def formulate_tc(trade_off: str, *, context: str | None = None, llm: Model | str) -> TCModel:
    agent = Agent(
        model=llm,
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
    llm: Model | str,
) -> str:
    agent = Agent(
        model=llm,
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
    store: TRIZStore,
    llm: Model | str,
    retrieve_k: int = 5,
) -> ContradictionResult:
    tc = await formulate_tc(problem_summary, llm=llm)

    seen: set[int] = set()
    candidates: list[Parameter] = []
    for effect in (tc.positive_effect, tc.negative_effect):
        for p in await store.search_parameters(effect, retrieve_k):
            if p.id not in seen:
                seen.add(p.id)
                candidates.append(p)

    if len(candidates) < 2:
        raise ValueError(f"Not enough parameter candidates found for contradiction: {tc}")

    agent = Agent(
        model=llm,
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
    store: TRIZStore,
    llm: Model | str,
    retrieve_k: int = 5,
) -> Principle:
    candidates = await store.search_principles(solution_summary, retrieve_k)
    if not candidates:
        raise ValueError("No principle candidates found for solution summary.")

    agent = Agent(
        model=llm,
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

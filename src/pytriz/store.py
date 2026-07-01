import csv
import importlib.resources as pkg_resources
import json
import logging
import random
from functools import lru_cache
from itertools import product

import numpy as np
from pydantic_ai import Embedder

from .core.retriever import Retriever
from .schemas.contradictions import Parameter, Principle

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------------------
# Data loading (cached — corpus never changes at runtime)
# --------------------------------------------------------------------------------------


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


# --------------------------------------------------------------------------------------
# TRIZStore
# --------------------------------------------------------------------------------------


class TRIZStore:
    """Indexed TRIZ corpus. Instantiate once and reuse across your application."""

    def __init__(self, embed_model: Embedder | None = None) -> None:
        self._embed_model = embed_model
        self._parameters = _load_parameters()
        self._principles = _load_principles()
        self._param_retriever = Retriever([p.text for p in self._parameters], embed_model)
        self._principle_retriever = Retriever([p.text for p in self._principles], embed_model)
        logger.info(
            "TRIZStore initialized (%s)",
            f"embedder: {embed_model.model}" if embed_model else "lexical search only",
        )

    async def ensure_index(self) -> None:
        """Precompute embeddings for parameters and principles, if an embedder is configured."""
        if self._embed_model is not None:
            await self._param_retriever.ensure_index()
            await self._principle_retriever.ensure_index()

    # --- Parameters ---

    def get_all_parameters(self) -> list[Parameter]:
        return self._parameters

    def get_parameter_by_id(self, parameter_id: int) -> Parameter:
        parameter = next((p for p in self._parameters if p.id == parameter_id), None)
        if not parameter:
            raise ValueError(f"Parameter with id {parameter_id} not found")
        return parameter

    async def search_parameters(self, query: str, top_k: int = 5) -> list[Parameter]:
        indices = await self._param_retriever.search(query, top_k)
        return [self._parameters[i] for i in indices]

    # --- Principles ---

    def get_all_principles(self) -> list[Principle]:
        return self._principles

    def get_principle_by_id(self, principle_id: int) -> Principle:
        principle = next((p for p in self._principles if p.id == principle_id), None)
        if not principle:
            raise ValueError(f"Principle with id {principle_id} not found")
        return principle

    def get_principle_by_name(self, principle_name: str) -> Principle:
        principle = next((p for p in self._principles if p.name.lower() == principle_name.lower()), None)
        if not principle:
            raise ValueError(f"Principle with name '{principle_name}' not found")
        return principle

    async def search_principles(self, query: str, top_k: int = 5) -> list[Principle]:
        indices = await self._principle_retriever.search(query, top_k)
        return [self._principles[i] for i in indices]

    def get_random_principles(self, count: int = 4) -> list[Principle]:
        if count >= len(self._principles):
            return self._principles
        return random.sample(self._principles, count)

    # --- Contradiction matrix ---

    def get_principles_from_matrix(
        self,
        improving_parameters: list[int],
        preserving_parameters: list[int],
    ) -> list[Principle]:
        if not all(isinstance(x, int) for x in improving_parameters + preserving_parameters):
            raise TypeError("All parameter IDs must be integers")
        if not all(x > 0 for x in improving_parameters + preserving_parameters):
            raise ValueError("All parameter IDs must be positive integers")

        matrix = _load_matrix()
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

        return [p for p in self._principles if p.id in sorted(principle_ids)]

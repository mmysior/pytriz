import pytest
import json

from pytriz.schemas.contradictions import Parameter, Principle, Separation
from pytriz.store import TRIZStore


@pytest.fixture(scope="module")
def store() -> TRIZStore:
    return TRIZStore()


def test_get_all_parameters(store: TRIZStore):
    parameters = store.get_all_parameters()
    assert isinstance(parameters, list)
    assert all(isinstance(p, Parameter) for p in parameters)
    assert len(parameters) == 39


def test_get_parameter_by_id(store: TRIZStore):
    parameter = store.get_parameter_by_id(1)
    assert isinstance(parameter, Parameter)
    assert parameter.id == 1
    assert parameter.name == "Weight of moving object"

    try:
        store.get_parameter_by_id(999)
        assert False, "Expected ValueError for non-existent parameter ID"
    except ValueError as e:
        assert str(e) == "Parameter with id 999 not found"


def test_get_principles_from_matrix(store: TRIZStore):
    improving_ids = [1, 3]
    preserving_ids = [17, 23]
    principles = store.get_principles_from_matrix(improving_ids, preserving_ids)
    assert isinstance(principles, list)
    assert all(isinstance(p, Principle) for p in principles)
    assert len(principles) == 12
    principle_ids = [p.id for p in principles]
    assert set(principle_ids) == {4, 10, 29, 6, 5, 35, 15, 23, 19, 3, 38, 31}


def test_get_all_principles(store: TRIZStore):
    principles = store.get_all_principles()
    assert isinstance(principles, list)
    assert all(isinstance(p, Principle) for p in principles)
    assert len(principles) == 40


def test_get_principle_by_id(store: TRIZStore):
    principle = store.get_principle_by_id(1)
    assert isinstance(principle, Principle)
    assert principle.id == 1
    assert principle.name == "Segmentation"

    try:
        store.get_principle_by_id(999)
        assert False, "Expected ValueError for non-existent principle ID"
    except ValueError as e:
        assert str(e) == "Principle with id 999 not found"


async def test_search_parameters(store: TRIZStore):
    results = await store.search_parameters("weight", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(p, Parameter) for p in results)
    assert any("weight" in p.name.lower() for p in results)


async def test_search_principles(store: TRIZStore):
    results = await store.search_principles("segmentation", top_k=3)
    assert isinstance(results, list)
    assert len(results) == 3
    assert all(isinstance(p, Principle) for p in results)
    assert any(p.name.lower() == "segmentation" for p in results)


def test_get_principle_by_name(store: TRIZStore):
    principle = store.get_principle_by_name("Segmentation")
    assert isinstance(principle, Principle)
    assert principle.id == 1
    assert principle.name == "Segmentation"

    try:
        store.get_principle_by_name("NonExistentPrinciple")
        assert False, "Expected ValueError for non-existent principle name"
    except ValueError as e:
        assert str(e) == "Principle with name 'NonExistentPrinciple' not found"


def test_store_accepts_valid_custom_corpus(tmp_path):
    parameters_path = tmp_path / "parameters.json"
    principles_path = tmp_path / "principles.json"
    matrix_path = tmp_path / "matrix.csv"
    parameters_path.write_text(
        json.dumps(
            {
                "parameters": [
                    {"id": 1, "name": "Speed", "description": "How fast it moves"},
                    {"id": 2, "name": "Cost", "description": "How much it costs"},
                ]
            }
        ),
        encoding="utf-8",
    )
    principles_path.write_text(
        json.dumps(
            {"principles": [{"id": 1, "name": "Split"}, {"id": 2, "name": "Combine"}]}
        ),
        encoding="utf-8",
    )
    matrix_path.write_text(";1\n2;\n", encoding="utf-8")

    store = TRIZStore(
        parameters_path=parameters_path,
        principles_path=principles_path,
        matrix_path=matrix_path,
    )

    assert [parameter.name for parameter in store.get_all_parameters()] == ["Speed", "Cost"]
    assert [principle.name for principle in store.get_principles_from_matrix([1], [2])] == ["Split"]
    assert store.get_all_separations() == []


def test_store_rejects_invalid_custom_matrix(tmp_path):
    matrix_path = tmp_path / "matrix.csv"
    matrix_path.write_text("1;999\n;\n", encoding="utf-8")

    with pytest.raises(ValueError, match="39x39"):
        TRIZStore(matrix_path=matrix_path)


def test_get_all_separations(store: TRIZStore):
    separations = store.get_all_separations()
    assert isinstance(separations, list)
    assert all(isinstance(s, Separation) for s in separations)
    assert [s.id for s in separations] == [1, 2, 3, 4, 5]


def test_get_separation_resolves_principles(store: TRIZStore):
    separation = store.get_separation_by_id(1)
    assert separation.name == "Separation in space"
    assert all(isinstance(principle, Principle) for principle in separation.principles)
    assert [principle.id for principle in separation.principles] == [1, 2, 3, 7, 4, 17]
    assert separation.principles[0] is store.get_principle_by_id(1)

    with pytest.raises(ValueError, match="Separation with id 999 not found"):
        store.get_separation_by_id(999)


def test_get_separation_by_name(store: TRIZStore):
    separation = store.get_separation_by_name("Separation in time")
    assert separation.id == 2

    with pytest.raises(ValueError, match="Separation with name 'Unknown' not found"):
        store.get_separation_by_name("Unknown")


async def test_search_separations(store: TRIZStore):
    results = await store.search_separations("different moments in time", top_k=1)
    assert [separation.id for separation in results] == [2]

import pytest
from pytriz.store import TRIZStore
from pytriz.schemas.contradictions import Parameter, Principle


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

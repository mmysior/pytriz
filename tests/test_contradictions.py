from pytriz import contradictions


def test_get_all_parameters():
    parameters = contradictions.get_all_parameters()
    assert isinstance(parameters, list)
    assert all(isinstance(p, contradictions.Parameter) for p in parameters)
    assert len(parameters) == 39


def test_search_parameters():
    results = contradictions.search_parameters("improves durability", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, contradictions.Parameter) for r in results)
    assert len(results) == 3


def test_get_parameter_by_id():
    parameter = contradictions.get_parameter_by_id(1)
    assert isinstance(parameter, contradictions.Parameter)
    assert parameter.id == 1
    assert parameter.name == "Weight of moving object"

    try:
        contradictions.get_parameter_by_id(999)
        assert False, "Expected ValueError for non-existent parameter ID"
    except ValueError as e:
        assert str(e) == "Parameter with id 999 not found"


def test_get_principles_from_matrix():
    improving_ids = [1, 3]
    preserving_ids = [17, 23]
    principles = contradictions.get_principles_from_matrix(improving_ids, preserving_ids)
    assert isinstance(principles, list)
    assert all(isinstance(p, contradictions.Principle) for p in principles)
    assert len(principles) == 12
    principle_ids = [p.id for p in principles]
    assert set(principle_ids) == {4, 10, 29, 6, 5, 35, 15, 23, 19, 3, 38, 31}


def test_get_all_principles():
    principles = contradictions.get_all_principles()
    assert isinstance(principles, list)
    assert all(isinstance(p, contradictions.Principle) for p in principles)
    assert len(principles) == 40


def test_search_principles():
    results = contradictions.search_principles("segmentation", top_k=3)
    assert isinstance(results, list)
    assert all(isinstance(r, contradictions.Principle) for r in results)
    assert len(results) == 3


def test_get_principle_by_id():
    principle = contradictions.get_principle_by_id(1)
    assert isinstance(principle, contradictions.Principle)
    assert principle.id == 1
    assert principle.name == "Segmentation"

    try:
        contradictions.get_principle_by_id(999)
        assert False, "Expected ValueError for non-existent principle ID"
    except ValueError as e:
        assert str(e) == "Principle with id 999 not found"


def test_get_principle_by_name():
    principle = contradictions.get_principle_by_name("Segmentation")
    assert isinstance(principle, contradictions.Principle)
    assert principle.id == 1
    assert principle.name == "Segmentation"

    try:
        contradictions.get_principle_by_name("NonExistentPrinciple")
        assert False, "Expected ValueError for non-existent principle name"
    except ValueError as e:
        assert str(e) == "Principle with name 'NonExistentPrinciple' not found"

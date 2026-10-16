from tokamak_analysis.analysis_registry import ANALYSES, resolve_analysis


def test_registry_contains_80_unique_ordered_analysis_ids():
    ids = [entry["id"] for entry in ANALYSES]
    assert len(ANALYSES) == 80
    assert ids == list(range(1, 81))


def test_registry_uses_distinct_entry_points_for_all_80_analyses():
    pairs = [(entry["module"], entry["function"]) for entry in ANALYSES]
    assert len(set(pairs)) == 80


def test_all_registered_analyses_resolve_to_callables():
    for entry in ANALYSES:
        assert callable(resolve_analysis(entry["id"])), entry

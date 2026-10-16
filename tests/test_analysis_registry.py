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


def test_thomson_entry_declares_real_inputs_outputs_and_scope():
    entry = ANALYSES[0]
    assert entry["name"] == "Thomson Scattering"
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete"
    assert contract["raw_data_to_physics"] is False
    inputs = {item["name"]: item for item in contract["input_specs"]}
    assert {"radius_m", "density_m3", "temperature_ev"} <= set(inputs)
    assert inputs["radius_m"]["unit"] == "m"
    assert inputs["density_m3"]["unit"] == "m^-3"
    assert inputs["temperature_ev"]["unit"] == "eV"
    assert inputs["major_radius_m"]["minimum"] > 0
    assert any("raw scattered-light spectra" in item for item in contract["limitations"])
    assert callable(resolve_analysis(1))

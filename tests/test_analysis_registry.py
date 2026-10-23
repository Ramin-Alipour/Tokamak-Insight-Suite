from tokamak_analysis.analysis_registry import ANALYSES, get_analysis, resolve_analysis


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
    assert contract["completeness"] == "incomplete"
    assert contract["raw_data_to_physics"] is True
    inputs = {item["name"]: item for item in contract["input_specs"]}
    assert {"radius_m", "density_m3", "temperature_ev"} <= set(inputs)
    assert inputs["radius_m"]["unit"] == "m"
    assert inputs["density_m3"]["unit"] == "m^-3"
    assert inputs["temperature_ev"]["unit"] == "eV"
    assert inputs["major_radius_m"]["minimum"] > 0
    assert any("Gaussian" in item for item in contract["limitations"])
    assert "tokamak_analysis.thomson_spectrum.fit_thomson_spectrum" in contract["function_links"]
    assert {mode["name"] for mode in contract["workflow_modes"]} == {
        "calibrated_profile",
        "raw_spectrum_temperature_fit",
        "calibrated_power_density",
        "calibrated_spectrum_temperature_density",
        "user_defined_spectral_model",
    }
    assert callable(resolve_analysis(1))


def test_get_analysis_returns_deep_copy_of_nested_contract():
    original = ANALYSES[0]["execution_contract"]["input_specs"][0]["name"]
    entry = get_analysis(1)
    entry["execution_contract"]["input_specs"][0]["name"] = "mutated"
    assert ANALYSES[0]["execution_contract"]["input_specs"][0]["name"] == original


def test_get_analysis_rejects_noninteger_ids_instead_of_truncating():
    import pytest

    with pytest.raises(ValueError, match="analysis_id must be an integer"):
        get_analysis(1.5)
    with pytest.raises(ValueError, match="analysis_id must be an integer"):
        get_analysis(True)


def test_molecular_spectroscopy_entry_declares_scoped_workflow():
    entry = get_analysis(6)
    assert entry["name"] == "Molecular Spectroscopy"
    assert entry["module"] == "molecular_spectroscopy"
    assert entry["function"] == "analyze_molecular_spectrum"
    contract = entry["execution_contract"]
    assert contract["completeness"] == "complete"
    assert contract["raw_data_to_physics"] is False
    assert {"wavelength_nm", "intensity"} <= {item["name"] for item in contract["input_specs"]}
    assert callable(resolve_analysis(6))

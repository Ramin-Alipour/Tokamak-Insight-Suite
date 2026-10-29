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
    assert entry["name"] == "Thomson scattering"
    assert "electron distribution constraints" not in entry["quantities"].lower()
    assert "normalized gradients" in entry["quantities"].lower()
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is True
    inputs = {item["name"]: item for item in contract["input_specs"]}
    assert {"radius_m", "density_m3", "temperature_ev"} <= set(inputs)
    assert inputs["radius_m"]["unit"] == "m"
    assert inputs["density_m3"]["unit"] == "m^-3"
    assert inputs["temperature_ev"]["unit"] == "eV"
    assert "major_radius_m" not in inputs
    assert "geometry-weighted volume averages" in " ".join(contract["limitations"]).lower()
    outputs = {item["name"]: item for item in contract["output_specs"]}
    assert {
        "density_m3", "temperature_eV", "density_gradient_m-4",
        "temperature_gradient_eV_m-1", "density_normalized_gradient_m-1",
        "temperature_normalized_gradient_m-1", "density_scale_length_m",
        "temperature_scale_length_m", "summary",
    } <= set(outputs)
    assert "radius_m" not in outputs or outputs["radius_m"]["unit"] == "m"
    assert any("Gaussian" in item for item in contract["limitations"])
    assert "tokamak_analysis.thomson_spectrum.fit_thomson_spectrum" in contract["function_links"]
    assert {mode["name"] for mode in contract["workflow_modes"]} == {
        "calibrated_profile",
        "raw_spectrum_temperature_fit",
        "spatial_profile_reconstruction",
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


def test_ece_entry_is_analysis_number_three_and_declares_scoped_workflow():
    entry = get_analysis(3)
    assert entry["name"] == "Electron Cyclotron Emission (ECE)"
    assert entry["module"] == "ece_analysis"
    assert entry["function"] == "analyze_ece"
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is True
    assert {item["name"] for item in contract["input_specs"]} >= {
        "frequency_Hz", "harmonic", "mode"
    }
    mode_spec = next(item for item in contract["input_specs"] if item["name"] == "mode")
    assert "brightness_temperature" in mode_spec["description"]
    assert "channel_fluctuations" in mode_spec["description"]
    assert "resonance_mapping" in mode_spec["description"]
    assert "tokamak_analysis.ece_analysis.analyze_ece" in contract["function_links"]
    assert callable(resolve_analysis(3))


def test_cxrs_is_analysis_four_with_explicit_physics_contract():
    entry = get_analysis(4)
    assert entry["name"] == "Charge-exchange recombination spectroscopy (CXRS)"
    assert entry["module"] == "cxrs"
    assert entry["function"] == "analyze_cxrs_line"
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is True
    inputs = {item["name"]: item for item in contract["input_specs"]}
    assert {
        "wavelength_nm", "signal", "rest_wavelength_nm",
        "ion_mass_amu", "instrument_sigma_nm",
    } <= set(inputs)
    assert inputs["instrument_sigma_nm"]["unit"] == "nm"
    assert any("FWHM" in inputs["instrument_sigma_nm"]["description"] for _ in [0])
    assert {item["name"] for item in contract["output_specs"]} >= {
        "ion_temperature_eV", "line_of_sight_velocity_m_s",
        "fitted_signal", "residual", "fit_r_squared",
    }
    assert any("instrumental width" in item for item in contract["limitations"])
    assert "instrumental-width uncertainty" in inputs["instrument_sigma_uncertainty_nm"]["description"]
    ti_uncertainty = next(item for item in contract["output_specs"] if item["name"] == "ion_temperature_uncertainty_eV")
    assert "instrumental-width uncertainty" in ti_uncertainty["description"]
    assert callable(resolve_analysis(4))


def test_analysis_id_81_is_not_part_of_the_canonical_80_analysis_registry():
    import pytest

    with pytest.raises(KeyError, match="Unknown analysis ID: 81"):
        get_analysis(81)

from tokamak_analysis.analysis_registry import ANALYSES, get_analysis, resolve_analysis


def test_registry_contains_80_unique_ordered_analysis_ids_after_retiring_id_7():
    ids = [entry["id"] for entry in ANALYSES]
    assert len(ANALYSES) == 80
    assert ids == [entry_id for entry_id in range(1, 82) if entry_id != 7]


def test_registry_uses_distinct_entry_points_for_all_80_registered_analyses():
    pairs = [(entry["module"], entry["function"]) for entry in ANALYSES]
    assert len(set(pairs)) == 80


def test_all_registered_analyses_resolve_to_callables():
    for entry in ANALYSES:
        assert callable(resolve_analysis(entry["id"])), entry


def test_thomson_entry_declares_measured_spectrum_to_physics_workflow():
    entry = ANALYSES[0]
    assert entry["name"] == "Thomson scattering"
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is False
    assert contract["calibrated_measurements_to_physics"] is True
    inputs = {item["name"]: item for item in contract["input_specs"]}
    assert {"wavelength_nm", "spectra", "radius_m", "laser_wavelength_nm",
            "scattering_angle_rad", "instrument_response"} <= set(inputs)
    assert inputs["wavelength_nm"]["unit"].startswith("nm")
    assert inputs["radius_m"]["unit"].startswith("m")
    assert any("measured" in item.lower() and "spectra" in item.lower()
               for item in contract["assumptions"])
    assert any("not identifiable" in item.lower() or "absolute density" in item.lower()
               for item in contract["limitations"])
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


def test_thomson_spectral_distribution_is_registered_as_entry_81():
    entry = get_analysis(81)
    assert entry["name"] == "Thomson spectral distribution"
    assert (entry["module"], entry["function"]) == (
        "thomson_distribution", "analyze_thomson_spectral_distribution"
    )
    assert callable(resolve_analysis(81))


def test_primary_analysis_family_lookup_rejects_invalid_ids():
    import pytest
    from tokamak_analysis.analysis_registry import get_analysis_family

    with pytest.raises(ValueError, match="family_id must be an integer"):
        get_analysis_family(True)
    with pytest.raises(KeyError, match="Unknown primary-analysis family ID: 21"):
        get_analysis_family(21)


def test_thomson_family_registry_contains_only_nonduplicate_workflows():
    import pytest
    from tokamak_analysis.analysis_registry import get_analysis, get_analysis_family

    cts = get_analysis(30)
    profile = get_analysis(46)
    assert (cts["module"], cts["function"]) == ("cts_analysis", "analyze_cts_spectrum_model")
    assert (profile["module"], profile["function"]) == (
        "thomson_workflow", "reconstruct_thomson_profile_from_spectra"
    )
    assert get_analysis_family(1)["entry_ids"] == [1, 30, 46, 81]
    with pytest.raises(KeyError, match="Unknown analysis ID: 7"):
        get_analysis(7)
    with pytest.raises(KeyError, match="Unknown analysis ID: 7"):
        resolve_analysis(7)
    assert callable(resolve_analysis(30))
    # CTS must require an explicit physical model; no electrostatic/full-EM fallback.
    with pytest.raises(ValueError, match="Select model"):
        resolve_analysis(30)()
    cts_contract = cts["execution_contract"]
    cts_inputs = {item["name"] for item in cts_contract["input_specs"]}
    assert {"frequency_shift_Hz", "spectrum", "scattering_geometry", "model"} <= cts_inputs
    assert not {"incident_direction", "scattered_direction", "mode_index",
                "frequency_resolved_geometry", "scattering_sample_count",
                "ion_temperature_eV"} & cts_inputs
    assert {item["name"] for item in cts_contract["workflow_modes"]} == {
        "unmagnetized_electrostatic_longitudinal_cts",
        "magnetized_full_electromagnetic_cts",
    }
    assert all(
        item["function"] == "tokamak_analysis.cts_analysis.analyze_cts_spectrum_model"
        for item in cts_contract["workflow_modes"]
    )
    assert callable(resolve_analysis(46))
    assert callable(resolve_analysis(81))


def test_thomson_formula_registry_contains_traceable_physical_relations():
    import json
    from pathlib import Path

    registry = json.loads((Path(__file__).resolve().parents[1] / "FORMULA_REGISTRY.json").read_text())
    thomson = [item for item in registry if "thomson" in item["name"].lower()]
    keys = {item["execution_contract"]["formula_key"] for item in thomson}
    assert {
        "thomson_electron_temperature_from_spectral_width",
        "thomson_velocity_projection",
        "thomson_density_from_scattered_power",
    } <= keys
    for item in thomson:
        contract = item["execution_contract"]
        assert contract["implementation_link_status"] == "mapped"
        assert contract["function_link"].startswith("tokamak_analysis.")

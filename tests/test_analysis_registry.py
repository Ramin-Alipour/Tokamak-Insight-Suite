from tokamak_analysis.analysis_registry import ANALYSES, get_analysis, resolve_analysis


def test_registry_contains_81_unique_ordered_analysis_ids_after_retiring_id_7():
    ids = [entry["id"] for entry in ANALYSES]
    assert len(ANALYSES) == 81
    assert ids == [entry_id for entry_id in range(1, 83) if entry_id != 7]


def test_registry_uses_distinct_entry_points_for_all_81_registered_analyses():
    pairs = [(entry["module"], entry["function"]) for entry in ANALYSES]
    assert len(set(pairs)) == 81


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
    assert "maximum_recoil_correction_fraction" in inputs
    assert inputs["include_compton_recoil"]["default"] is False
    assert {"recoil_included", "scattering_cross_section_model"} <= {item["name"] for item in contract["output_specs"]}
    assert "Klein-Nishina" in inputs["include_compton_recoil"]["description"]
    profile_inputs = {item["name"] for item in get_analysis(46)["execution_contract"]["input_specs"]}
    assert {"profile_reconstruction_method", "profile_reconstruction_points",
            "profile_reconstruction_mc_samples", "maximum_recoil_correction_fraction"} <= profile_inputs
    distribution_entry = get_analysis(81)
    distribution_inputs = {item["name"] for item in distribution_entry["execution_contract"]["input_specs"]}
    distribution_specs = {
        item["name"]: item
        for item in distribution_entry["execution_contract"]["input_specs"]
    }
    background_covariance_description = distribution_specs["background_signal_covariance"]["description"].lower()
    assert "cross-covariance" not in background_covariance_description or "may be combined" in background_covariance_description
    assert "positive definite" in background_covariance_description
    assert "signal_background_cross_covariance" in distribution_specs
    assert {"moment_mc_samples", "instrument_response_matrix"} <= distribution_inputs
    assert "instrument_sigma_uncertainty_nm" not in distribution_inputs
    distribution_outputs = {item["name"] for item in distribution_entry["execution_contract"]["output_specs"]}
    assert {"observed_sigma_nm", "moment_uncertainty_method",
            "gaussian_fit_parameter_uncertainties", "fitted_spectrum",
            "reduced_chi_square"} <= distribution_outputs
    assert "instrument_corrected_sigma_nm" not in distribution_outputs
    assert "instrument_width_resolved" not in distribution_outputs
    assert "moment_uncertainty_diagnostics" not in distribution_outputs
    primary_outputs = {item["name"] for item in contract["output_specs"]}
    primary_inputs = {item["name"]: item for item in contract["input_specs"]}
    assert "electron_velocity_distribution" in primary_outputs
    assert primary_inputs["enable_evdf_reconstruction"]["default"] is True
    assert primary_inputs["evdf_velocity_points"]["default"] == 121
    assert callable(resolve_analysis(1))


def test_thomson_relativistic_geometry_uncertainty_contract_matches_implementation():
    inputs = {
        item["name"]: item
        for item in get_analysis(1)["execution_contract"]["input_specs"]
    }
    laser = inputs["laser_wavelength_uncertainty_nm"]["description"].lower()
    angle = inputs["scattering_angle_uncertainty_rad"]
    assert "relativistic maxwell-juttner" in laser
    assert "monte carlo" in laser
    assert "relativistic maxwell-juttner" in angle["unit"].lower()
    assert "monte carlo" in angle["description"].lower()
    assert "mutually exclusive" in angle["description"].lower()


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



def test_thomson_spectral_distribution_registry_is_descriptive_only_for_calibrated_measurements():
    registry_path = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = next(item for item in registry
                 if item["diagnostic"] == "Thomson spectral distribution")
    assert entry["raw_data_to_physics"] is False
    assert entry["calibrated_measurements_to_physics"] is False
    contract = entry["execution_contract"]
    assert contract["raw_data_to_physics"] is False
    assert contract["calibrated_measurements_to_physics"] is True
    assert contract["input_data_level"] == "measured_calibrated_spectrum"
    assert "spectral descriptors only" in contract["note"]



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
    assert get_analysis_family(1)["entry_ids"] == [1, 30, 46, 81, 82]
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
    assert {"instrument_response_uncertainty", "response_mc_samples"} <= cts_inputs
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


def test_joint_thomson_profile_mode_exposes_data_only_identifiability_contract():
    entry = get_analysis(46)
    contract = entry["execution_contract"]
    inputs = {item["name"] for item in contract["input_specs"]}
    assert {"joint_spatial_spectral", "spatial_mixing_matrix",
            "spatial_regularization_strength", "kappa_index", "fit_kappa_index",
            "incident_polarization", "include_compton_recoil",
            "maximum_recoil_correction_fraction"} <= inputs
    assert "relativistic_maxwell_juttner" in contract["algorithm"]
    assert "Klein-Nishina" in contract["algorithm"]
    outputs = {item["name"] for item in contract["output_specs"]}
    assert {"primary_result_status", "primary_outputs_reportable_as_identified",
            "optimizer", "optimizer_candidate"} <= outputs
    assert any("regularization rows do not count" in item.lower()
               for item in contract["limitations"])


def test_thomson_spatial_profile_relativistic_geometry_uncertainty_contract_matches_implementation():
    inputs = {
        item["name"]: item
        for item in get_analysis(46)["execution_contract"]["input_specs"]
    }
    laser = inputs["laser_wavelength_uncertainty_nm"]["description"].lower()
    angle = inputs["scattering_angle_uncertainty_rad"]
    assert "relativistic maxwell-juttner" in laser
    assert "monte carlo" in laser
    assert "relativistic maxwell-juttner" in angle["unit"].lower()
    assert "monte carlo" in angle["description"].lower()
    assert "mutually exclusive" in angle["description"].lower()


def test_thomson_profile_json_registry_copies_match_runtime_uncertainty_contract():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    root_registry = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_registry = json.loads(
        (root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert root_registry == packaged_registry
    entry = next(item for item in root_registry if item.get("diagnostic") == "Thomson spatial profile reconstruction")
    inputs = {
        item["name"]: item
        for item in entry["execution_contract"]["input_specs"]
    }
    assert "monte carlo" in inputs["laser_wavelength_uncertainty_nm"]["description"].lower()
    assert "relativistic maxwell-juttner" in inputs["scattering_angle_uncertainty_rad"]["unit"].lower()
    assert "mutually exclusive" in inputs["scattering_angle_uncertainty_rad"]["description"].lower()


def test_full_em_cts_response_uncertainty_matches_runtime_json_and_callable_contracts():
    import json
    from pathlib import Path

    runtime = get_analysis(30)["execution_contract"]
    runtime_inputs = {item["name"]: item for item in runtime["input_specs"]}
    for name in ("instrument_response_uncertainty", "response_mc_samples"):
        assert name in runtime_inputs
        description = runtime_inputs[name]["description"].lower()
        assert "both" in description
        assert "full-electromagnetic" in description or "both cts branches" in description
        assert "required_when_model" not in runtime_inputs[name]
    runtime_outputs = {item["name"] for item in runtime["output_specs"]}
    assert {
        "instrument_response_uncertainty_supplied",
        "instrument_response_uncertainty_propagated",
        "instrument_response_mc_successes",
        "instrument_response_mc_requested",
        "instrument_response_parameter_covariance",
        "instrument_response_uncertainty_warning",
    } <= runtime_outputs
    full_mode = next(
        item for item in runtime["workflow_modes"]
        if item["name"] == "magnetized_full_electromagnetic_cts"
    )
    assert any("instrument_response_uncertainty" in item for item in full_mode["inputs"])
    assert any("response_mc_samples" in item for item in full_mode["inputs"])
    assert any("numerical_convergence" in item for item in full_mode["inputs"])
    assert any("convergence_harmonic_orders" in item for item in full_mode["inputs"])
    assert any("numerical_convergence_status" in item for item in full_mode["outputs"])
    convergence_inputs = {
        "numerical_convergence", "convergence_harmonic_orders",
        "convergence_quadrature_orders", "convergence_relative_tolerance",
    }
    assert convergence_inputs <= set(runtime_inputs)
    assert runtime_inputs["numerical_convergence"]["default"] is True
    assert runtime_inputs["convergence_relative_tolerance"]["default"] == 1e-3
    assert {
        "numerical_convergence", "numerical_convergence_status",
        "numerical_convergence_is_uncertainty",
    } <= runtime_outputs

    root = Path(__file__).resolve().parents[1]
    root_registry = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_registry = json.loads(
        (root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert root_registry == packaged_registry
    entry = next(
        item for item in root_registry
        if item.get("diagnostic") == "Collective Thomson / coherent scattering (CTS)"
    )
    contract = entry["execution_contract"]
    json_inputs = {item["name"]: item for item in contract["input_specs"]}
    assert convergence_inputs <= set(json_inputs)
    assert json_inputs["numerical_convergence"]["default"] is True
    json_outputs = {item["name"] for item in contract["output_specs"]}
    assert {"numerical_convergence", "numerical_convergence_status",
            "numerical_convergence_is_uncertainty"} <= json_outputs
    json_output_specs = contract["output_specs"]
    assert sum(item["name"] == "numerical_convergence_status" for item in json_output_specs) == 1
    convergence_status_description = next(
        item["description"] for item in json_output_specs
        if item["name"] == "numerical_convergence_status"
    ).lower()
    assert "not_converged_at_final_refinement" in convergence_status_description
    assert "single inversions do not establish" not in convergence_status_description
    assert "applies_when_model" not in json_inputs["instrument_response_uncertainty"]
    assert "applies_when_model" not in json_inputs["response_mc_samples"]
    assert "response-matrix uncertainty is propagated by Monte Carlo refitting" in contract["uncertainty_model"]
    assert not any(
        "measured response-matrix and external-geometry/parameter uncertainties are not propagated" in item
        for item in contract["limitations"]
    )


def test_all_thomson_subanalyses_share_calibrated_measurement_and_identifiability_contract():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    root_registry = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_registry = json.loads(
        (root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert root_registry == packaged_registry

    names = {
        "Thomson scattering",
        "Collective Thomson / coherent scattering (CTS)",
        "Thomson spatial profile reconstruction",
        "Thomson spectral distribution",
    }
    for name in names:
        entry = next(item for item in root_registry if item["diagnostic"] == name)
        contract = entry["execution_contract"]
        assert contract["raw_data_to_physics"] is False, name
        assert "measured" in (contract.get("note", "") + entry.get("method", "")).lower(), name
        assert "calibrat" in (
            contract.get("note", "") + entry.get("method", "") + " ".join(contract.get("assumptions", []))
        ).lower(), name
        output_names = {item["name"] for item in contract["output_specs"]}
        assert "parameter_identifiability" in output_names, name
        disclosure = " ".join([
            contract.get("note", ""),
            entry.get("note", ""),
            entry.get("caveat", ""),
            " ".join(contract.get("limitations", [])),
        ]).lower()
        assert "experimental validation" in disclosure, name


def test_thomson_cts_and_spectral_distribution_contracts_match_fixed_operator_and_covariance_apis():
    import json
    from pathlib import Path

    import numpy as np
    import pytest

    from tokamak_analysis.cts_analysis import _convolve

    root = Path(__file__).resolve().parents[1]
    root_registry = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_registry = json.loads(
        (root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert root_registry == packaged_registry

    cts = next(
        item for item in root_registry
        if item.get("diagnostic") == "Collective Thomson / coherent scattering (CTS)"
    )["execution_contract"]
    cts_inputs = {item["name"]: item for item in cts["input_specs"]}
    response_description = cts_inputs["instrument_response"]["description"].lower()
    assert "already" in response_description and "unit-sum columns" in response_description
    assert "applied unchanged" in response_description
    assert "normalized as a redistribution probability by the forward model" not in response_description
    uncertainty_description = cts_inputs["instrument_response_uncertainty"]["description"].lower()
    assert "base matrix" in uncertainty_description and "used unchanged" in uncertainty_description
    assert "perturbed" in uncertainty_description

    frequency = np.linspace(-2.0, 2.0, 9)
    response = np.eye(frequency.size)
    response[0, 0] = 0.8
    response[1, 0] = 0.2
    supplied = response.copy()
    _convolve(response, np.ones(frequency.size), frequency)
    np.testing.assert_array_equal(response, supplied)
    invalid = response.copy()
    invalid[:, 0] *= 0.5
    with pytest.raises(ValueError, match="already sum to one"):
        _convolve(invalid, np.ones(frequency.size), frequency)

    distribution = next(
        item for item in root_registry
        if item.get("diagnostic") == "Thomson spectral distribution"
    )["execution_contract"]
    distribution_inputs = {item["name"]: item for item in distribution["input_specs"]}
    background_covariance_description = distribution_inputs["background_signal_covariance"]["description"].lower()
    assert "signal_background_cross_covariance" in background_covariance_description
    assert "not currently represented" not in background_covariance_description
    cross_description = distribution_inputs["signal_background_cross_covariance"]["description"].lower()
    assert "both marginal covariance models" in cross_description
    assert "positive-definite" in cross_description


def test_full_em_cts_local_external_covariance_contract_matches_runtime_json_and_dashboard_api():
    import json
    from pathlib import Path

    runtime = get_analysis(30)["execution_contract"]
    runtime_inputs = {item["name"]: item for item in runtime["input_specs"]}
    expected_inputs = {
        "external_input_covariance", "external_input_parameter_order",
        "external_input_mc_samples", "external_input_mc_seed",
    }
    assert expected_inputs <= set(runtime_inputs)
    assert runtime_inputs["external_input_mc_samples"]["default"] == 100
    assert runtime_inputs["external_input_mc_seed"]["default"] == 84622
    assert "frequency-resolved ray/equilibrium/beam covariance" in (
        runtime_inputs["external_input_covariance"]["description"]
    )
    assert "propagate_full_em_cts_ray_geometry_uncertainty" in (
        runtime_inputs["external_input_covariance"]["description"]
    )
    assert "geometry_sampler" in runtime_inputs
    runtime_outputs = {item["name"] for item in runtime["output_specs"]}
    expected_outputs = {
        "external_input_covariance_supplied", "external_input_covariance_order",
        "external_input_covariance_propagated", "external_input_parameter_covariance",
        "external_input_mc_successes", "external_input_mc_requested",
        "external_input_covariance_warning", "external_inputs_fixed",
        "external_inputs_with_supplied_covariance", "ray_geometry_parameter_covariance",
        "ray_geometry_uncertainty",
    }
    assert expected_outputs <= runtime_outputs
    full_mode = next(
        item for item in runtime["workflow_modes"]
        if item["name"] == "magnetized_full_electromagnetic_cts"
    )
    assert any("external_input_covariance" in item for item in full_mode["inputs"])
    assert any("external_input_parameter_covariance" in item for item in full_mode["outputs"])
    geometry_mode = next(
        item for item in runtime["workflow_modes"]
        if item["name"] == "full_forward_ray_geometry_uncertainty"
    )
    assert "geometry_sampler" in " ".join(geometry_mode["inputs"])
    assert any("ray_geometry_parameter_covariance" in item for item in geometry_mode["outputs"])

    root = Path(__file__).resolve().parents[1]
    root_registry = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_registry = json.loads(
        (root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    assert root_registry == packaged_registry
    entry = next(
        item for item in root_registry
        if item["diagnostic"] == "Collective Thomson / coherent scattering (CTS)"
    )
    contract = entry["execution_contract"]
    json_inputs = {item["name"]: item for item in contract["input_specs"]}
    assert expected_inputs <= set(json_inputs)
    assert json_inputs["external_input_covariance"]["applies_when_model"] == "full_electromagnetic"
    assert "geometry_sampler" in json_inputs
    json_outputs = {item["name"] for item in contract["output_specs"]}
    assert expected_outputs <= json_outputs
    assert {"external_inputs_fixed", "external_inputs_with_supplied_covariance",
            "ray_geometry_parameter_covariance", "ray_geometry_uncertainty"} <= json_outputs
    geometry_json_mode = next(
        item for item in contract["workflow_modes"]
        if item["name"] == "full_forward_ray_geometry_uncertainty"
    )
    assert "geometry_sampler(rng, realization_index)" in " ".join(geometry_json_mode["inputs"])
    json_mode = next(
        item for item in contract["workflow_modes"]
        if item["name"] == "magnetized_full_electromagnetic_cts"
    )
    assert any("external_input_covariance" in item for item in json_mode["inputs"])
    assert any("external_input_parameter_covariance" in item for item in json_mode["outputs"])


def test_gyrotropic_anisotropic_thomson_is_registered_and_dashboard_resolvable():
    entry = get_analysis(82)
    assert entry["name"] == "Gyrotropic anisotropic optical Thomson inference"
    assert (entry["module"], entry["function"]) == (
        "thomson_anisotropic_inference",
        "fit_gyrotropic_bi_maxwellian_thomson",
    )
    contract = entry["execution_contract"]
    assert contract["raw_data_to_physics"] is True
    assert {"wavelength_nm", "measured_spectra", "magnetic_field_unit",
            "scattering_vector_unit", "instrument_response_matrix"} <= {
        item["name"] for item in contract["input_specs"]
    }
    assert "T_q = T_perp + (T_parallel - T_perp)" in contract["algorithm"]
    assert callable(resolve_analysis(82))

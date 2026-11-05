"""Cross-subanalysis contract regressions for Thomson Analysis 1.

These checks verify executable registry wiring and machine-readable contracts.
They do not claim physical validation or imply that declared model limitations
have been implemented.
"""
from tokamak_analysis.analysis_registry import get_analysis, resolve_analysis
import inspect


def test_analysis_1_optical_profile_and_distribution_routes_are_executable():
    for analysis_id in (1, 30, 46, 81):
        entry = get_analysis(analysis_id)
        assert callable(resolve_analysis(analysis_id)), entry
        contract = entry["execution_contract"]
        assert contract["calibrated_measurements_to_physics"] is True or analysis_id == 81
        assert contract["raw_data_to_physics"] is False


def test_thomson_analysis_1_registry_input_and_output_names_are_unique():
    for analysis_id in (1, 30, 46, 81):
        contract = get_analysis(analysis_id)["execution_contract"]
        input_names = [item["name"] for item in contract["input_specs"]]
        output_names = [item["name"] for item in contract["output_specs"]]
        assert len(input_names) == len(set(input_names)), analysis_id
        assert len(output_names) == len(set(output_names)), analysis_id


def test_parameter_identifiability_is_declared_across_all_thomson_subanalyses():
    for analysis_id in (1, 30, 46, 81):
        outputs = {
            item["name"]
            for item in get_analysis(analysis_id)["execution_contract"]["output_specs"]
        }
        assert "parameter_identifiability" in outputs, analysis_id


def test_thomson_profile_and_spectral_distribution_remain_distinct_workflows():
    primary = get_analysis(1)
    profile = get_analysis(46)
    distribution = get_analysis(81)

    assert (primary["module"], primary["function"]) == (
        "thomson_workflow", "analyze_thomson_spectra"
    )
    assert (profile["module"], profile["function"]) == (
        "thomson_workflow", "reconstruct_thomson_profile_from_spectra"
    )
    assert (distribution["module"], distribution["function"]) == (
        "thomson_distribution", "analyze_thomson_spectral_distribution"
    )

    profile_inputs = {
        item["name"] for item in profile["execution_contract"]["input_specs"]
    }
    assert {
        "profile_coordinate_covariance_m2",
        "profile_coordinate_value_cross_covariance",
    } <= profile_inputs

    distribution_inputs = {
        item["name"] for item in distribution["execution_contract"]["input_specs"]
    }
    distribution_outputs = {
        item["name"] for item in distribution["execution_contract"]["output_specs"]
    }
    assert "laser_wavelength_nm" not in distribution_inputs
    assert "scattering_angle_rad" not in distribution_inputs
    assert "electron_velocity_distribution" not in distribution_outputs
    assert "observed_sigma_nm" in distribution_outputs
    distribution_inputs = {
        item["name"] for item in distribution["execution_contract"]["input_specs"]
    }
    assert {
        "physical_model", "physical_initial", "physical_fit_options",
        "model_discrepancy_covariance", "systematic_covariance",
        "incident_polarization", "include_compton_recoil",
        "physical_background_model",
    } <= distribution_inputs


def test_analysis_1_declares_channelwise_background_basis_and_mj_joint_gls():
    import inspect
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra
    from tokamak_analysis.thomson_kinetic_models import fit_relativistic_maxwell_juttner_channels

    entry = get_analysis(1)
    inputs = {
        item["name"]: item
        for item in entry["execution_contract"]["input_specs"]
    }
    assert {
        "background_model", "background_degree",
        "background_interior_knots", "background_templates",
        "model_discrepancy_covariance", "systematic_covariance",
    } <= set(inputs)
    from pathlib import Path
    dashboard_source = Path("app.py").read_text(encoding="utf-8")
    assert "ts_infer_model_discrepancy_covariance" in dashboard_source
    assert "ts_infer_systematic_covariance" in dashboard_source
    assert "model_discrepancy_covariance" in inspect.signature(analyze_thomson_spectra).parameters
    assert "systematic_covariance" in inspect.signature(analyze_thomson_spectra).parameters
    assert "non_collective_relativistic_maxwell_juttner" in inputs["model"]["unit"]
    import json
    diagnostic_records = json.loads(
        (Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    optical = next(item for item in diagnostic_records if item["diagnostic"] == "Thomson scattering")
    assert "systematic_covariance" in {
        item["name"] for item in optical["execution_contract"]["input_specs"]
    }
    assert "background_model" in inspect.signature(analyze_thomson_spectra).parameters
    parameters = inspect.signature(fit_relativistic_maxwell_juttner_channels).parameters
    assert {
        "background_model", "background_degree",
        "background_interior_knots", "background_templates",
    } <= set(parameters)


def test_thomson_family_maps_all_four_distinct_executable_subanalyses():
    import json
    from pathlib import Path
    from tokamak_analysis.analysis_registry import get_analysis_family

    family = get_analysis_family(1)
    assert family["entry_ids"] == [1, 30, 46, 81]
    assert len({(get_analysis(i)["module"], get_analysis(i)["function"])
                for i in family["entry_ids"]}) == 4

    diagnostic_registry = json.loads(
        (Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(
            encoding="utf-8"
        )
    )
    names = {item["diagnostic"] for item in diagnostic_registry}
    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
        encoding="utf-8"
    )
    assert "analyze_cts_spectrum_model" in app_source
    assert "cts_model=st.selectbox" in app_source
    assert "spectral_distribution" in app_source
    assert "observed_sigma_nm" in app_source
    assert {
        "Thomson scattering",
        "Collective Thomson / coherent scattering (CTS)",
        "Thomson spatial profile reconstruction",
        "Thomson spectral distribution",
    } <= names

    by_name = {item["diagnostic"]: item for item in diagnostic_registry}
    expected_functions = {
        1: ("Thomson scattering", "tokamak_analysis.thomson_workflow.analyze_thomson_spectra"),
        30: ("Collective Thomson / coherent scattering (CTS)", "tokamak_analysis.cts_analysis.analyze_cts_spectrum_model"),
        46: ("Thomson spatial profile reconstruction", "tokamak_analysis.thomson_workflow.reconstruct_thomson_profile_from_spectra"),
        81: ("Thomson spectral distribution", "tokamak_analysis.thomson_distribution.analyze_thomson_spectral_distribution"),
    }
    for analysis_id, (diagnostic_name, function_path) in expected_functions.items():
        assert function_path in by_name[diagnostic_name]["implementation_functions"]
        assert callable(resolve_analysis(analysis_id))
    assert "tokamak_analysis.cts_kappa.kappa_electron_susceptibility" in by_name[
        "Collective Thomson / coherent scattering (CTS)"
    ]["implementation_functions"]


def test_cts_public_dispatcher_signature_and_conditional_outputs_match_registry():
    import inspect
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum_model

    entry = get_analysis(30)
    assert (entry["module"], entry["function"]) == (
        "cts_analysis", "analyze_cts_spectrum_model"
    )
    parameters = inspect.signature(analyze_cts_spectrum_model).parameters
    assert "model" in parameters
    assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())

    outputs = {
        item["name"]: item
        for item in entry["execution_contract"]["output_specs"]
    }
    assert "parameter_identifiability" in outputs
    assert "fit_covariance" in outputs
    assert "model_discrepancy_covariance" in outputs
    assert outputs["parameter_identifiability"]["unit"].startswith("structured")
    assert outputs["electron_density_m3"]["unit"] == "m^-3"
    assert outputs["electron_temperature_eV"]["unit"] == "eV"
    assert "tokamak_analysis.cts_kappa.kappa_electron_susceptibility" in entry["execution_contract"]["function_links"]
    optical = get_analysis(1)["execution_contract"]["output_specs"]
    profile = get_analysis(46)["execution_contract"]["output_specs"]
    optical_units = {item["name"]: item["unit"] for item in optical}
    profile_units = {item["name"]: item["unit"] for item in profile}
    for name in (
        "electron_temperature_profile_eV",
        "electron_density_profile_m3",
        "electron_velocity_projection_profile_m_s",
    ):
        assert optical_units[name] == profile_units[name]


def test_analysis1_public_output_identifiability_contract_matches_runtime_producers():
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra
    from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    from tokamak_analysis.cts_analysis import analyze_cts_spectrum
    from tokamak_analysis.cts_full_em import analyze_full_em_cts_spectrum

    producers = (
        analyze_thomson_spectra,
        analyze_cts_spectrum,
        analyze_full_em_cts_spectrum,
        reconstruct_thomson_profile_from_spectra,
        analyze_thomson_spectral_distribution,
    )
    for producer in producers:
        assert callable(producer)
        assert '"parameter_identifiability"' in inspect.getsource(producer), producer.__name__
    for analysis_id in (1, 30, 46, 81):
        outputs = {
            item["name"]
            for item in get_analysis(analysis_id)["execution_contract"]["output_specs"]
        }
        assert "parameter_identifiability" in outputs, analysis_id


def test_analysis1_runtime_and_packaged_registries_are_byte_and_contract_consistent():
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for registry_name in (
        "ANALYSIS_FAMILIES.json",
        "DIAGNOSTIC_REGISTRY.json",
        "FORMULA_REGISTRY.json",
    ):
        assert (root / registry_name).read_bytes() == (
            root / "tis" / "data" / registry_name
        ).read_bytes(), registry_name

    diagnostic_records = json.loads(
        (root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8")
    )
    by_name = {record["diagnostic"]: record for record in diagnostic_records}
    expected = {
        1: "Thomson scattering",
        30: "Collective Thomson / coherent scattering (CTS)",
        46: "Thomson spatial profile reconstruction",
        81: "Thomson spectral distribution",
    }
    for analysis_id, diagnostic_name in expected.items():
        runtime_contract = get_analysis(analysis_id)["execution_contract"]
        packaged_contract = by_name[diagnostic_name]["execution_contract"]
        assert packaged_contract == runtime_contract, (
            analysis_id,
            diagnostic_name,
        )


def test_optical_model_validity_is_declared_in_python_and_json_registries():
    import json
    from pathlib import Path
    from tokamak_analysis.analysis_registry import get_analysis

    python_outputs = {
        item["name"]
        for item in get_analysis(1)["execution_contract"]["output_specs"]
    }
    assert "model_validity" in python_outputs
    root = json.loads((Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    entry = next(item for item in root if item["diagnostic"] == "Thomson scattering")
    json_outputs = {item["name"] for item in entry["execution_contract"]["output_specs"]}
    assert "model_validity" in json_outputs
    assert "tokamak_analysis.thomson_identifiability.optical_model_validity" in entry["implementation_functions"]


def test_anisotropic_thomson_multi_pitch_model_is_wired_to_dispatcher_dashboard_and_formula_registry():
    import json
    from pathlib import Path
    import pytest
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    root = Path(__file__).resolve().parents[1]
    model = "gyrotropic_bi_maxwellian_anisotropy"
    result = run_advanced_thomson_physics(
        model,
        projected_temperature_eV=[180.0, 105.0, 80.0],
        magnetic_pitch_cosine=[1.0, 0.5, 0.0],
        projected_temperature_covariance_eV2=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        magnetic_pitch_cosine_covariance=[[1.0e-5, 0.0, 0.0], [0.0, 1.0e-5, 0.0], [0.0, 0.0, 1.0e-5]],
    )
    assert result["electron_temperature_parallel_eV"] == pytest.approx(180.0)
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(80.0)
    assert result["geometry_uncertainty_included"] is True
    assert result["geometry_covariance_iterations"] > 0
    assert result["public_physics_entrypoint"] == model

    dashboard = (root/"tis"/"ui"/"advanced_thomson_physics_page.py").read_text(encoding="utf-8")
    assert model in dashboard
    formula = json.loads((root/"FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_formula = json.loads((root/"tis"/"data"/"FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "thomson_gyrotropic_bimaxwellian_anisotropy"
    matches = [item for item in formula if item.get("execution_contract", {}).get("formula_key") == key]
    assert len(matches) == 1
    assert matches == [item for item in packaged_formula if item.get("execution_contract", {}).get("formula_key") == key]


def test_three_dimensional_thomson_flow_is_connected_to_dashboard_and_formula_registry():
    import json
    import numpy as np
    from pathlib import Path
    import pytest
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    root = Path(__file__).resolve().parents[1]
    model = "thomson_three_dimensional_flow"
    result = run_advanced_thomson_physics(
        model,
        velocity_projection_m_s=[1.0e5, 2.0e5, 3.0e5],
        scattering_vector_unit=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        projection_covariance_m2_s2=[[1.0e4, 0.0, 0.0], [0.0, 1.0e4, 0.0], [0.0, 0.0, 1.0e4]],
        scattering_vector_covariance=(1.0e-6*np.eye(9)).tolist(),
    )
    assert result["electron_velocity_vector_m_s"] == pytest.approx([1.0e5, 2.0e5, 3.0e5])
    assert result["geometry_uncertainty_included"] is True
    assert result["geometry_covariance_iterations"] > 0
    assert result["public_physics_entrypoint"] == model
    dashboard = (root/"tis"/"ui"/"advanced_thomson_physics_page.py").read_text(encoding="utf-8")
    assert model in dashboard
    root_formula = json.loads((root/"FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_formula = json.loads((root/"tis"/"data"/"FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "thomson_three_dimensional_flow_from_projections"
    assert len([item for item in root_formula if item.get("execution_contract", {}).get("formula_key") == key]) == 1
    assert root_formula == packaged_formula



def test_profile_registry_exposes_joint_spatial_operator_covariance_inputs_and_output():
    entry = get_analysis(46)
    inputs = {
        item["name"]: item
        for item in entry["execution_contract"]["input_specs"]
    }
    outputs = {
        item["name"]
        for item in entry["execution_contract"]["output_specs"]
    }
    assert {
        "spatial_mixing_covariance",
        "spatial_mixing_mc_samples",
        "spatial_mixing_mc_seed",
    } <= set(inputs)
    assert "spatial_mixing_uncertainty" in outputs

"""Cross-subanalysis contract regressions for Thomson Analysis 1.

These checks verify executable registry wiring and machine-readable contracts.
They do not claim physical validation or imply that declared model limitations
have been implemented.
"""
from tokamak_analysis.analysis_registry import get_analysis, resolve_analysis


def test_analysis_1_optical_profile_and_distribution_routes_are_executable():
    for analysis_id in (1, 46, 81):
        entry = get_analysis(analysis_id)
        assert callable(resolve_analysis(analysis_id)), entry
        contract = entry["execution_contract"]
        assert contract["calibrated_measurements_to_physics"] is True or analysis_id == 81
        assert contract["raw_data_to_physics"] is False


def test_thomson_analysis_1_registry_input_and_output_names_are_unique():
    for analysis_id in (1, 46, 81):
        contract = get_analysis(analysis_id)["execution_contract"]
        input_names = [item["name"] for item in contract["input_specs"]]
        output_names = [item["name"] for item in contract["output_specs"]]
        assert len(input_names) == len(set(input_names)), analysis_id
        assert len(output_names) == len(set(output_names)), analysis_id


def test_parameter_identifiability_is_declared_across_all_thomson_subanalyses():
    for analysis_id in (1, 46, 81):
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
    } <= set(inputs)
    assert "non_collective_relativistic_maxwell_juttner" in inputs["model"]["unit"]
    assert "background_model" in inspect.signature(analyze_thomson_spectra).parameters
    parameters = inspect.signature(fit_relativistic_maxwell_juttner_channels).parameters
    assert {
        "background_model", "background_degree",
        "background_interior_knots", "background_templates",
    } <= set(parameters)

import numpy as np
import pytest

from tis.probe_analysis import floating_potential, langmuir_iv_analysis, probe_iv_summary


def test_langmuir_analysis_reports_preprocessing_quality_and_duplicate_aggregation():
    voltage = np.r_[np.linspace(-5.0, 5.0, 11), 0.0, np.nan]
    current = np.r_[np.linspace(-2e-3, 2e-3, 11), 1e-3, 0.0]

    result = langmuir_iv_analysis(voltage, current)

    quality = result["input_quality"]
    assert quality["input_pair_count"] == 13
    assert quality["finite_pair_count"] == 12
    assert quality["discarded_nonfinite_pair_count"] == 1
    assert quality["duplicate_voltage_pair_count_aggregated"] == 1
    assert quality["processed_unique_voltage_count"] == 11
    assert quality["current_convention_normalized"] == "electron_positive"

    processed_v = np.asarray(result["processed_voltage_V"])
    assert np.all(np.diff(processed_v) > 0)
    assert len(processed_v) == quality["processed_unique_voltage_count"]
    assert len(result["processed_current_A"]) == len(processed_v)
    assert len(result["dIdV_A_per_V"]) == len(processed_v)
    assert processed_v.tolist() == sorted(set(voltage[np.isfinite(voltage)].tolist()))



def test_langmuir_analysis_exposes_electron_fit_residual_diagnostics():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -1e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)

    result = langmuir_iv_analysis(
        voltage,
        current,
        electron_fit_range=(-8.0, 0.0),
    )

    assert result["electron_temperature_eV"] is not None
    assert result["electron_retarding_fit_points"] >= 4
    assert np.isfinite(result["electron_retarding_fit_R2"])
    assert np.isfinite(result["electron_retarding_fit_residual_rms_A"])
    assert np.isfinite(result["electron_retarding_fit_normalized_rmse"])
    assert result["electron_retarding_fit_status"] in {"numerical_fit_review_pass_not_physical_validation", "weak_or_undefined_numerical_fit_review_required"}
    assert result["electron_retarding_regime_review"]["physical_regime_confirmed"] is False


def test_floating_potential_helper_uses_zero_plateau_midpoint():
    assert floating_potential([-2, -1, 0, 1, 2], [-2, 0, 0, 2, 3]) == -0.5
    summary = probe_iv_summary([-2, -1, 0, 1, 2], [-2, 0, 0, 2, 3])
    assert summary["floating_potential_interpolated_V"] == -0.5
    assert summary["floating_potential_candidates_V"] == [-0.5]
    assert summary["floating_potential_status"] == "single_zero_crossing"


def test_floating_potential_helper_does_not_hide_multiple_crossings():
    voltage = [-2, -1, 0, 1, 2]
    current = [-1, 1, -1, 1, 2]
    assert floating_potential(voltage, current) is None
    summary = probe_iv_summary(voltage, current)
    assert summary["floating_potential_interpolated_V"] is None
    assert summary["floating_potential_status"] == "ambiguous_multiple_zero_crossings"
    assert np.allclose(summary["floating_potential_candidates_V"], [-1.5, -0.5, 0.5])


def test_floating_potential_helper_validates_array_shape_and_averages_duplicate_voltage():
    with pytest.raises(ValueError, match="one-dimensional"):
        floating_potential([[0, 1]], [[-1, 1]])
    assert floating_potential([-1, 0, 0, 1], [-1, -1, 1, 1]) == 0.0


def test_langmuir_ion_branch_fit_reports_residual_quality_diagnostics():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-5 * voltage
    result = langmuir_iv_analysis(
        voltage,
        current,
        ion_fit_range=(-10.0, -5.0),
    )

    assert result["ion_saturation_fit_points"] >= 3
    assert np.isfinite(result["ion_saturation_fit_R2"])
    assert np.isfinite(result["ion_saturation_fit_residual_rms_A"])
    assert result["ion_saturation_fit_normalized_rmse"] is not None
    assert result["ion_saturation_fit_status"] in {"numerical_fit_review_pass_not_physical_validation", "weak_or_undefined_numerical_fit_review_required"}
    assert result["ion_saturation_regime_review"]["physical_ion_saturation_confirmed"] is False
    assert result["ion_saturation_current_method"] == "median_user_selected_ion_fit_range"



def test_langmuir_density_uncertainty_propagates_explicit_inputs_only():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    result = langmuir_iv_analysis(
        voltage, current,
        electron_fit_range=(-8.0, 0.0),
        ion_fit_range=(-10.0, -7.0),
        probe_area_m2=1e-4,
        ion_mass_kg=3.34e-27,
        ion_saturation_coefficient=0.61,
        sheath_model="test-model",
        ion_current_uncertainty_A=1e-5,
        probe_area_relative_uncertainty=0.02,
        ion_mass_relative_uncertainty=0.01,
        ion_saturation_coefficient_relative_uncertainty=0.05,
        electron_temperature_relative_uncertainty=0.03,
    )
    assert result["ion_density_estimate_m-3"] > 0
    assert result["ion_density_standard_uncertainty_m-3"] > 0
    assert result["ion_density_relative_standard_uncertainty"] > 0
    assert result["ion_density_uncertainty_status"] == "partial_first_order_propagation_supplied_components_only"


def test_langmuir_density_uncertainty_rejects_negative_components():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        langmuir_iv_analysis(
            voltage, current,
            electron_fit_range=(-8.0, 0.0),
            ion_fit_range=(-10.0, -7.0),
            probe_area_m2=1e-4,
            ion_mass_kg=3.34e-27,
            ion_saturation_coefficient=0.61,
            sheath_model="test-model",
            probe_area_relative_uncertainty=-0.01,
        )


def test_langmuir_proxy_and_scope_are_explicit_without_fit_windows():
    voltage = np.linspace(-5.0, 5.0, 21)
    current = np.sinh(voltage / 3.0) * 1e-3
    result = langmuir_iv_analysis(voltage, current)
    assert result["plasma_potential_method"] == "maximum_numerical_dIdV_proxy"
    assert result["plasma_potential_status"] == "proxy_only_requires_probe_regime_review"
    assert result["analysis_scope"] == "calibrated_single_probe_IV_sweep_only"
    assert result["fit_window_review"]["physical_regime_certified"] is False


def test_public_langmuir_wrapper_forwards_density_uncertainty_contract():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe

    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    uncertainty_inputs = {
        "ion_current_uncertainty_A": 1e-5,
        "probe_area_relative_uncertainty": 0.02,
        "ion_mass_relative_uncertainty": 0.01,
        "ion_saturation_coefficient_relative_uncertainty": 0.05,
        "electron_temperature_relative_uncertainty": 0.03,
    }
    common_inputs = {
        "electron_fit_range": (-8.0, 0.0),
        "ion_fit_range": (-10.0, -7.0),
        "probe_area_m2": 1e-4,
        "ion_mass_kg": 3.34e-27,
        "ion_saturation_coefficient": 0.61,
        "sheath_model": "test-model",
    }

    core_result = langmuir_iv_analysis(
        voltage, current, **common_inputs, **uncertainty_inputs
    )
    public_result = analyze_langmuir_probe(
        voltage, current, **common_inputs, **uncertainty_inputs
    )

    assert public_result["ion_density_estimate_m-3"] == pytest.approx(
        core_result["ion_density_estimate_m-3"]
    )
    assert public_result["ion_density_standard_uncertainty_m-3"] == pytest.approx(
        core_result["ion_density_standard_uncertainty_m-3"]
    )
    assert public_result["ion_density_relative_standard_uncertainty"] == pytest.approx(
        core_result["ion_density_relative_standard_uncertainty"]
    )
    assert public_result["ion_density_uncertainty_components_relative"] == pytest.approx(
        core_result["ion_density_uncertainty_components_relative"]
    )
    assert public_result["ion_density_uncertainty_status"] == (
        "partial_first_order_propagation_supplied_components_only"
    )
    assert set(public_result["ion_density_uncertainty_components_relative"]) == {
        "ion_current", "probe_area", "ion_mass", "sheath_coefficient",
        "electron_temperature",
    }



def test_generic_probe_entry_point_uses_canonical_langmuir_contract():
    from tokamak_analysis.probe_analysis import analyze_probe_voltage_current
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe

    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    kwargs = {"electron_fit_range": (-8.0, 0.0), "ion_fit_range": (-10.0, -7.0)}
    canonical = analyze_langmuir_probe(voltage, current, **kwargs)
    generic = analyze_probe_voltage_current(voltage, current, **kwargs)

    for key in ("processed_voltage_V", "processed_current_A", "dIdV_A_per_V",
                "floating_potential_V", "electron_temperature_eV",
                "ion_saturation_current_A", "ion_saturation_fit_R2",
                "analysis_scope", "input_quality"):
        assert generic[key] == canonical[key]
    assert np.array_equal(generic["voltage_V"], voltage)
    assert np.array_equal(generic["current_A"], current)
    assert np.array_equal(generic["dI_dV_S"], canonical["dIdV_A_per_V"])
    assert generic["electron_temperature_estimate_eV"] == canonical["electron_temperature_eV"]


def test_generic_probe_entry_point_enforces_canonical_minimum_sample_count():
    from tokamak_analysis.probe_analysis import analyze_probe_voltage_current
    with pytest.raises(ValueError, match="At least 8 valid voltage points"):
        analyze_probe_voltage_current(np.arange(7.0), np.arange(7.0))


def test_public_langmuir_contract_omits_optional_estimates_when_not_requested():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe
    voltage = np.linspace(-5.0, 5.0, 21)
    current = np.sinh(voltage / 3.0) * 1e-3
    result = analyze_langmuir_probe(voltage, current)
    assert result["electron_temperature_eV"] is None
    assert "ion_density_estimate_m-3" not in result
    assert result["ion_saturation_current_method"] == "median_lowest_voltage_tail_fallback"
    assert result["analysis_scope"] == "calibrated_single_probe_IV_sweep_only"



def test_langmuir_array_analysis_processes_channels_independently():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe_array

    voltage = np.linspace(-5.0, 5.0, 21)
    currents = np.vstack([
        np.sinh(voltage / 3.0) * 1e-3,
        np.sinh((voltage - 0.25) / 3.5) * 8e-4,
    ])
    result = analyze_langmuir_probe_array(
        voltage, currents, channel_names=["inner", "outer"]
    )
    assert result["channel_count"] == 2
    assert set(result["channels"]) == {"inner", "outer"}
    assert result["analysis_scope"] == "calibrated_multi_channel_independent_IV_sweeps"
    assert result["cross_channel_inference_performed"] is False
    assert result["channels"]["inner"]["analysis_scope"] == "calibrated_single_probe_IV_sweep_only"


def test_langmuir_array_rejects_mismatched_channel_shapes():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe_array

    with pytest.raises(ValueError, match="identical shapes"):
        analyze_langmuir_probe_array(np.zeros((2, 8)), np.zeros((3, 8)))

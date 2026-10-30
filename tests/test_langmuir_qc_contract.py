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
        ion_fit_range=(-10.0, -9.0),
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
    current = np.tanh(voltage / 3.0) * 1e-3
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
        "ion_fit_range": (-10.0, -9.0),
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


def test_generic_probe_entry_point_preserves_documented_short_sweep_compatibility():
    from tokamak_analysis.probe_analysis import analyze_probe_voltage_current
    # The canonical Langmuir workflow requires eight distinct samples. The
    # generic compatibility adapter intentionally retains descriptive-only
    # behavior for 5–7 point legacy sweeps; it does not perform canonical fits.
    result = analyze_probe_voltage_current(np.arange(7.0), np.arange(7.0))
    assert result["sample_count"] == 7
    assert result["floating_potential_V"] == 0.0
    assert result["electron_temperature_eV"] is None
    assert result["analysis_scope"] == "descriptive_short_sweep_compatibility_only"


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


def test_langmuir_array_accepts_named_mapping_without_current_argument():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe_array

    voltage = np.linspace(-5.0, 5.0, 21)
    current = np.sinh(voltage / 3.0) * 1e-3
    result = analyze_langmuir_probe_array({
        "inner": {"voltage_V": voltage, "current_A": current},
        "outer": {"voltage_V": voltage, "current_A": 0.8 * current},
    })
    assert result["channel_count"] == 2
    assert set(result["channels"]) == {"inner", "outer"}


def test_langmuir_array_rejects_empty_channels_and_stringified_duplicate_names():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe_array

    with pytest.raises(ValueError, match="at least one calibrated probe channel"):
        analyze_langmuir_probe_array({})
    voltage = np.linspace(-5.0, 5.0, 8)
    currents = np.vstack([voltage, voltage])
    with pytest.raises(ValueError, match="unique string name"):
        analyze_langmuir_probe_array(voltage, currents, channel_names=[1, "1"])


def test_langmuir_array_rejects_too_few_samples_before_dispatch():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe_array

    with pytest.raises(ValueError, match="at least one channel and eight samples"):
        analyze_langmuir_probe_array(np.arange(7.0), np.ones((1, 7)))


def test_langmuir_array_mapping_rejects_stringified_key_collisions():
    from tokamak_analysis.langmuir_probe import analyze_langmuir_probe_array

    voltage = np.linspace(-5.0, 5.0, 8)
    current = np.sinh(voltage / 3.0) * 1e-3
    channels = {
        1: {"voltage_V": voltage, "current_A": current},
        "1": {"voltage_V": voltage, "current_A": current},
    }
    with pytest.raises(ValueError, match="remain unique when converted to strings"):
        analyze_langmuir_probe_array(channels)

def test_second_derivative_method_reports_inflection_zero_crossings_not_its_maximum():
    voltage = np.linspace(-2.0, 2.0, 41)
    current = voltage**3 + 0.01 * voltage
    result = langmuir_iv_analysis(voltage, current)
    assert result["plasma_potential_candidate_method"].startswith("maximum_first_derivative")
    assert result["plasma_potential_second_derivative_method"] == "zero_crossings_of_second_derivative_for_inflection_candidates"
    assert any(abs(value) < 1e-10 for value in result["plasma_potential_d2IdV2_zero_crossing_candidates_V"])
    assert result["plasma_potential_d2IdV2_max_candidate_method"] == "maximum_second_derivative_extremum_not_inflection_method"


def test_floating_and_plasma_proxy_uncertainties_are_conditional_and_not_grid_spacing():
    voltage = np.linspace(-1.0, 1.0, 9)
    current = 1e-3 * voltage
    no_sigma = langmuir_iv_analysis(voltage, current)
    assert no_sigma["floating_potential_standard_uncertainty_V"] is None
    assert no_sigma["floating_potential_uncertainty_status"] == "not_quantified_current_uncertainty_not_supplied"
    assert no_sigma["plasma_potential_proxy_standard_uncertainty_V"] is None
    assert no_sigma["plasma_potential_voltage_grid_spacing_median_V"] == pytest.approx(0.25)
    with_sigma = langmuir_iv_analysis(voltage, current, current_uncertainty_A=1e-5)
    assert with_sigma["floating_potential_standard_uncertainty_V"] is not None
    assert with_sigma["floating_potential_standard_uncertainty_V"] > 0
    assert with_sigma["plasma_potential_proxy_standard_uncertainty_V"] is not None
    assert with_sigma["plasma_potential_proxy_uncertainty_status"] == "monte_carlo_current_noise_only_not_physical_validation"


def test_current_uncertainty_validation_and_zero_is_explicit():
    voltage = np.linspace(-1.0, 1.0, 9)
    current = 1e-3 * voltage
    with pytest.raises(ValueError, match="finite non-negative"):
        langmuir_iv_analysis(voltage, current, current_uncertainty_A=-1e-5)
    result = langmuir_iv_analysis(voltage, current, current_uncertainty_A=0.0)
    assert result["plasma_potential_proxy_uncertainty_status"] == "monte_carlo_current_noise_only_not_physical_validation"
    assert result["plasma_potential_proxy_standard_uncertainty_V"] == pytest.approx(0.0)


def test_density_inversion_rejects_overlapping_or_touching_fit_windows():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    with pytest.raises(ValueError, match="must be disjoint"):
        langmuir_iv_analysis(voltage, current, electron_fit_range=(-8.0, 1.0), ion_fit_range=(1.0, 3.0))


def test_density_inversion_requires_numerically_acceptable_electron_fit(monkeypatch):
    import tis.probe_analysis as probe_module
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    real_r2 = probe_module._r2
    calls = {"count": 0}
    def controlled_r2(observed, fitted):
        calls["count"] += 1
        return 0.99 if calls["count"] == 1 else 0.5
    monkeypatch.setattr(probe_module, "_r2", controlled_r2)
    with pytest.raises(ValueError, match="electron-retarding fit that passes"):
        langmuir_iv_analysis(
            voltage, current, electron_fit_range=(-8.0, 0.0), ion_fit_range=(-10.0, -9.0),
            probe_area_m2=1e-4, ion_mass_kg=3.34e-27,
            ion_saturation_coefficient=0.61, sheath_model="test sheath",
        )
    monkeypatch.setattr(probe_module, "_r2", real_r2)


def test_invalid_electron_fit_covariance_is_never_reported_as_zero_uncertainty(monkeypatch):
    import tis.probe_analysis as probe_module
    real_curve_fit = probe_module.optimize.curve_fit
    def invalid_covariance(*args, **kwargs):
        parameters, covariance = real_curve_fit(*args, **kwargs)
        return parameters, np.full_like(covariance, np.nan)
    monkeypatch.setattr(probe_module.optimize, "curve_fit", invalid_covariance)
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -1e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    result = langmuir_iv_analysis(voltage, current, electron_fit_range=(-8.0, 0.0))
    assert result["electron_retarding_covariance_valid"] is False
    assert result["electron_temperature_standard_error_eV"] is None
    assert result["electron_temperature_relative_standard_uncertainty"] is None
    assert result["electron_temperature_total_standard_uncertainty_eV"] is None
    assert result["electron_temperature_uncertainty_status"] == "incomplete_fit_covariance_statistical_uncertainty_unavailable"


def test_density_uncertainty_is_not_calculated_when_te_statistical_uncertainty_is_unavailable(monkeypatch):
    import tis.probe_analysis as probe_module
    real_curve_fit = probe_module.optimize.curve_fit
    def invalid_covariance(*args, **kwargs):
        parameters, covariance = real_curve_fit(*args, **kwargs)
        return parameters, np.full_like(covariance, np.nan)
    monkeypatch.setattr(probe_module.optimize, "curve_fit", invalid_covariance)
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    result = langmuir_iv_analysis(
        voltage, current, electron_fit_range=(-8.0, 0.0), ion_fit_range=(-10.0, -9.0),
        probe_area_m2=1e-4, ion_mass_kg=3.34e-27,
        ion_saturation_coefficient=0.61, sheath_model="test sheath",
        ion_current_uncertainty_A=1e-5, probe_area_relative_uncertainty=0.02,
    )
    assert result["ion_density_estimate_m-3"] > 0
    assert result["ion_density_standard_uncertainty_m-3"] is None
    assert result["ion_density_relative_standard_uncertainty"] is None
    assert result["ion_density_uncertainty_status"] == "not_calculated_electron_temperature_fit_uncertainty_unavailable"


def test_registry_declares_corrected_langmuir_output_contract_fields():
    import json
    from pathlib import Path
    registry = json.loads((Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    entry = next(item for item in registry if item["diagnostic"] == "Langmuir Probe")
    declared = {item["name"] for item in entry["execution_contract"]["output_specs"]}
    required = {
        "plasma_potential_candidate_method",
        "plasma_potential_second_derivative_method",
        "plasma_potential_d2IdV2_zero_crossing_candidates_V",
        "fit_window_review",
        "ion_density_numerical_fit_screen_passed",
        "ion_density_current_flatness_screen_passed",
        "ion_density_physical_regime_confirmed",
        "electron_retarding_covariance_valid",
        "floating_potential_standard_uncertainty_V",
        "floating_potential_uncertainty_status",
        "plasma_potential_proxy_standard_uncertainty_V",
        "plasma_potential_proxy_uncertainty_status",
    }
    assert required <= declared


def test_probe_analysis_ui_uses_engine_fit_status_and_exposes_declared_workflows():
    from pathlib import Path
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    start = app.index("### Langmuir probe I–V analysis")
    end = app.index("elif page=='Plasma Parameters':", start)
    section = app[start:end]
    assert "0.95" not in section
    assert "electron_retarding_fit_status" in section
    assert "estimate_electron_density_from_saturation" in section
    assert "estimate_ion_particle_flux" in section
    assert "fit_langmuir_electron_models" in section
    assert "analyze_langmuir_probe_array" in section
    assert "analyze_langmuir_probe_series" in section

def test_canonical_langmuir_result_keys_are_fully_declared_in_registry():
    import json
    from pathlib import Path
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -1e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)
    result = langmuir_iv_analysis(voltage, current, electron_fit_range=(-8.0, 0.0), ion_fit_range=(-10.0, -9.0))
    rows = json.loads((Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    entry = next(item for item in rows if item["diagnostic"] == "Langmuir Probe")
    declared = {item["name"] for item in entry["execution_contract"]["output_specs"]}
    assert set(result) <= declared, sorted(set(result) - declared)


def test_root_and_packaged_langmuir_diagnostic_registries_are_identical():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    root_rows = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    package_rows = json.loads((root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    assert root_rows == package_rows


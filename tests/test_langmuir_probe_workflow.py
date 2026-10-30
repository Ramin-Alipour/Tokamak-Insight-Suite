import numpy as np
import pytest

from tokamak_analysis.langmuir_probe import analyze_langmuir_probe


def synthetic_iv():
    voltage = np.linspace(-5.0, 20.0, 101)
    current = -1.0e-3 + 2.0e-5 * np.exp((voltage - 10.0) / 3.0)
    return voltage, current


def test_langmuir_probe_recovers_floating_potential_and_electron_temperature():
    voltage, current = synthetic_iv()
    # This variant crosses zero within the measured voltage sweep; the default
    # synthetic amplitude intentionally does not, and should report no crossing.
    current = -1.0e-3 + 4.0e-5 * np.exp((voltage - 10.0) / 3.0)
    result = analyze_langmuir_probe(
        voltage,
        current,
        electron_fit_range=(10.0, 20.0),
        ion_fit_range=(-5.0, 0.0),
    )
    assert result["floating_potential_V"] is not None
    assert result["electron_temperature_eV"] == pytest.approx(3.0, rel=0.04)
    assert result["electron_retarding_fit_R2"] > 0.999
    assert result["ion_saturation_current_A"] < 0


def test_langmuir_probe_excludes_optional_geometry_dependent_density_estimate():
    voltage, current = synthetic_iv()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(2.0, 10.0),
        ion_fit_range=(-5.0, -1.0),
    )
    assert result["electron_temperature_eV"] == pytest.approx(3.0, rel=0.04)
    assert "ion_density_estimate_m-3" not in result
    assert "ion_sound_speed_m_s" not in result

def test_langmuir_probe_accepts_ion_positive_circuit_convention():
    voltage, current = synthetic_iv()
    result = analyze_langmuir_probe(
        voltage,
        -current,
        electron_fit_range=(2.0, 10.0),
        ion_fit_range=(-5.0, -1.0),
        current_convention="ion_positive",
    )
    assert result["current_convention_input"] == "ion_positive"
    assert result["current_convention"] == "electron_positive"
    assert result["electron_temperature_eV"] == pytest.approx(3.0, rel=0.04)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"electron_fit_range": (1.0, 1.0)},
        {"ion_fit_range": (1.0, 1.0)},
        {"electron_fit_range": (float("nan"), 2.0)},
        {"current_convention": "unknown"},
    ],
)
def test_langmuir_probe_rejects_invalid_model_inputs(kwargs):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError):
        analyze_langmuir_probe(voltage, current, **kwargs)

def test_langmuir_probe_rejects_mismatched_and_undersampled_sweeps():
    with pytest.raises(ValueError, match="equal lengths"):
        analyze_langmuir_probe([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="8 valid"):
        analyze_langmuir_probe([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])


def test_langmuir_probe_without_fit_windows_returns_labelled_tail_estimates():
    voltage = np.linspace(-10.0, 10.0, 101)
    current = 2e-3 * np.tanh(voltage / 3.0) + 1e-4 * voltage
    result = analyze_langmuir_probe(voltage, current)
    assert np.isfinite(result["ion_saturation_current_A"])
    assert np.isfinite(result["electron_saturation_current_A"])
    assert result["ion_saturation_current_method"] == "median_lowest_voltage_tail_fallback"
    assert result["electron_saturation_current_method"] == "median_highest_voltage_tail_descriptive"
    assert result["electron_temperature_eV"] is None
    assert "interpretation_warning" in result


@pytest.mark.parametrize("voltage,current", [
    (np.zeros((3, 3)), np.zeros((3, 3))),
    (np.arange(8).reshape(2, 4), np.arange(8)),
    (np.arange(8), np.arange(8).reshape(2, 4)),
])
def test_langmuir_probe_rejects_non_one_dimensional_sweeps(voltage, current):
    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_langmuir_probe(voltage, current)


def test_langmuir_probe_selected_ion_range_overrides_tail_fallback():
    voltage = np.linspace(-5.0, 20.0, 101)
    current = -1.0e-3 + 2.0e-5 * np.exp((voltage - 10.0) / 3.0)
    result = analyze_langmuir_probe(
        voltage, current,
        electron_fit_range=(10.0, 20.0),
        ion_fit_range=(-5.0, 0.0),
    )
    assert result["ion_saturation_current_method"] == "median_user_selected_ion_fit_range"
    selected = (voltage >= -5.0) & (voltage <= 0.0)
    assert result["ion_saturation_current_A"] == pytest.approx(float(np.median(current[selected])), abs=1e-12)


def test_langmuir_probe_does_not_infer_density_from_tail_fallback():
    voltage = np.linspace(-10.0, 10.0, 101)
    current = 2e-3 * np.tanh(voltage / 3.0) + 1e-4 * voltage
    result = analyze_langmuir_probe(voltage, current)
    assert "ion_density_estimate_m-3" not in result
    assert "ion_density_status" not in result

def test_langmuir_probe_rejects_legacy_geometry_argument():
    voltage, current = synthetic_iv()
    with pytest.raises(TypeError):
        analyze_langmuir_probe(voltage, current, ion_mass_amu=2.0)

def test_langmuir_probe_rejects_legacy_collection_area_argument():
    voltage, current = synthetic_iv()
    with pytest.raises(TypeError):
        analyze_langmuir_probe(voltage, current, area_m2=1.0e-5)

@pytest.mark.parametrize("fit_range", [(-20.0, -15.0), (15.0, 25.0)])
def test_langmuir_probe_rejects_fit_windows_outside_measured_sweep(fit_range):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="overlap"):
        analyze_langmuir_probe(voltage, current, ion_fit_range=fit_range)
    with pytest.raises(ValueError, match="overlap"):
        analyze_langmuir_probe(voltage, current, electron_fit_range=fit_range)


@pytest.mark.parametrize("fit_range", [(1.0,), (1.0, 2.0, 3.0), (float("nan"), 2.0)])
def test_langmuir_probe_rejects_malformed_fit_windows(fit_range):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="two finite voltage limits"):
        analyze_langmuir_probe(voltage, current, electron_fit_range=fit_range)



@pytest.mark.parametrize(
    "name",
    [
        "electron_temperature_relative_uncertainty",
        "electron_temperature_calibration_relative_uncertainty",
        "electron_temperature_fit_window_relative_uncertainty",
    ],
)
@pytest.mark.parametrize("bad_value", [True, np.bool_(False), float("nan"), float("inf"), -0.1, "0.1"])
def test_langmuir_probe_rejects_invalid_electron_temperature_uncertainty(name, bad_value):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match=name):
        analyze_langmuir_probe(voltage, current, **{name: bad_value})


def density_case(**overrides):
    voltage = np.linspace(-5.0, 20.0, 101)
    current = -1.0e-3 + 4.0e-5 * np.exp((voltage - 10.0) / 3.0)
    kwargs = {
        "electron_fit_range": (10.0, 20.0),
        "ion_fit_range": (-5.0, 0.0),
        "probe_area_m2": 1.0e-5,
        "ion_mass_kg": 3.3435837724e-27,
        "ion_saturation_coefficient": 0.61,
        "sheath_model": "synthetic Bohm/sheath model",
    }
    kwargs.update(overrides)
    return voltage, current, kwargs


def test_langmuir_probe_density_formula_and_units():
    from tis.constants import E_CHARGE

    voltage, current, kwargs = density_case()
    result = analyze_langmuir_probe(voltage, current, **kwargs)
    ion_current = result["ion_saturation_current_A"]
    te = result["electron_temperature_eV"]
    expected = abs(ion_current) / (
        kwargs["ion_saturation_coefficient"] * E_CHARGE * kwargs["probe_area_m2"]
        * np.sqrt(E_CHARGE * te / kwargs["ion_mass_kg"])
    )
    assert result["ion_density_estimate_m-3"] == pytest.approx(expected, rel=1e-12)
    assert result["ion_density_status"] == "model_based_estimate_review_assumptions"
    assert result["ion_density_model"]["formula"].startswith("n_i = |I_i,sat|")
    assert result["ion_density_uncertainty_status"] == "not_propagated_uncertainty_inputs_not_supplied"
    assert result["ion_density_standard_uncertainty_m-3"] is None


@pytest.mark.parametrize(
    "parameter,factor,expected_factor",
    [
        ("probe_area_m2", 2.0, 0.5),
        ("ion_mass_kg", 4.0, 2.0),
        ("ion_saturation_coefficient", 2.0, 0.5),
    ],
)
def test_langmuir_probe_density_responds_to_explicit_model_parameters(
    parameter, factor, expected_factor
):
    voltage, current, kwargs = density_case()
    baseline = analyze_langmuir_probe(voltage, current, **kwargs)
    changed = dict(kwargs)
    changed[parameter] *= factor
    result = analyze_langmuir_probe(voltage, current, **changed)
    assert result["ion_density_estimate_m-3"] == pytest.approx(
        baseline["ion_density_estimate_m-3"] * expected_factor, rel=1e-10
    )


def test_langmuir_probe_density_scales_with_selected_ion_current():
    voltage, current, kwargs = density_case()
    baseline = analyze_langmuir_probe(voltage, current, **kwargs)
    scaled = analyze_langmuir_probe(voltage, current * 2.0, **kwargs)
    assert scaled["electron_temperature_eV"] == pytest.approx(
        baseline["electron_temperature_eV"], rel=1e-8
    )
    assert scaled["ion_density_estimate_m-3"] == pytest.approx(
        2.0 * baseline["ion_density_estimate_m-3"], rel=1e-8
    )


def test_langmuir_probe_propagates_all_supplied_density_uncertainties():
    voltage, current, kwargs = density_case(
        ion_current_uncertainty_A=2.0e-6,
        probe_area_relative_uncertainty=0.04,
        ion_mass_relative_uncertainty=0.02,
        ion_saturation_coefficient_relative_uncertainty=0.03,
        electron_temperature_relative_uncertainty=0.05,
        electron_temperature_calibration_relative_uncertainty=0.06,
        electron_temperature_fit_window_relative_uncertainty=0.07,
        density_model_relative_uncertainty=0.08,
    )
    result = analyze_langmuir_probe(voltage, current, **kwargs)
    ion_current = abs(result["ion_saturation_current_A"])
    te = result["electron_temperature_eV"]
    rel_te_stat = result["electron_temperature_standard_error_eV"] / te
    rel_te = np.sqrt(rel_te_stat**2 + 0.05**2 + 0.06**2 + 0.07**2)
    components = {
        "ion_current": 2.0e-6 / ion_current,
        "probe_area": 0.04,
        "ion_mass": 0.5 * 0.02,
        "sheath_coefficient": 0.03,
        "electron_temperature": 0.5 * rel_te,
        "density_model_discrepancy": 0.08,
    }
    expected_relative = np.sqrt(sum(value**2 for value in components.values()))
    assert result["ion_density_uncertainty_components_relative"] == pytest.approx(components)
    assert result["ion_density_relative_standard_uncertainty"] == pytest.approx(expected_relative)
    assert result["ion_density_standard_uncertainty_m-3"] == pytest.approx(
        result["ion_density_estimate_m-3"] * expected_relative
    )
    assert result["ion_density_uncertainty_status"] == "partial_first_order_propagation_supplied_components_only"


@pytest.mark.parametrize(
    "name",
    [
        "ion_current_uncertainty_A",
        "probe_area_relative_uncertainty",
        "ion_mass_relative_uncertainty",
        "ion_saturation_coefficient_relative_uncertainty",
        "density_model_relative_uncertainty",
    ],
)
@pytest.mark.parametrize("bad_value", [True, np.bool_(True), float("nan"), float("inf"), -0.01, "0.1"])
def test_langmuir_probe_rejects_invalid_density_uncertainty(name, bad_value):
    voltage, current, kwargs = density_case(**{name: bad_value})
    with pytest.raises(ValueError, match=name):
        analyze_langmuir_probe(voltage, current, **kwargs)


def test_langmuir_probe_rejects_density_uncertainty_without_density_parameters():
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="require density inversion parameters"):
        analyze_langmuir_probe(
            voltage, current, ion_current_uncertainty_A=1.0e-6
        )


def test_langmuir_probe_density_requires_numerically_flat_ion_branch():
    voltage, current, kwargs = density_case(
        electron_fit_range=(10.0, 20.0), ion_fit_range=(-5.0, 0.0)
    )
    # Keep the electron-retarding branch exponential while deliberately making
    # the selected low-voltage ion branch non-flat. The windows remain disjoint.
    ion_mask = voltage <= 0.0
    current[ion_mask] = -1.0e-3 + 4.0e-5 * (voltage[ion_mask] + 5.0)
    with pytest.raises(ValueError, match="ion-branch fit that passes|numerically flat ion-branch"):
        analyze_langmuir_probe(voltage, current, **kwargs)


def test_langmuir_probe_preprocessing_sorts_aggregates_and_reports_quality():
    voltage = np.array([3.0, 1.0, 2.0, 0.0, 4.0, 5.0, 6.0, 7.0, 7.0, np.nan])
    current = np.array([30.0, 10.0, 20.0, 0.0, 40.0, 50.0, 60.0, 70.0, 90.0, 123.0])
    result = analyze_langmuir_probe(voltage, current)
    assert result["processed_voltage_V"] == pytest.approx(np.arange(8, dtype=float))
    assert result["processed_current_A"] == pytest.approx([0, 10, 20, 30, 40, 50, 60, 80])
    assert result["input_quality"]["input_pair_count"] == 10
    assert result["input_quality"]["finite_pair_count"] == 9
    assert result["input_quality"]["discarded_nonfinite_pair_count"] == 1
    assert result["input_quality"]["duplicate_voltage_pair_count_aggregated"] == 1
    assert result["input_quality"]["processed_unique_voltage_count"] == 8


def test_langmuir_probe_accepts_exactly_eight_distinct_finite_points():
    result = analyze_langmuir_probe(np.arange(8), np.linspace(-1.0, 1.0, 8))
    assert result["n_points"] == 8
    assert result["input_quality"]["processed_unique_voltage_count"] == 8


def test_langmuir_probe_derivatives_support_irregular_grid_and_label_edge_candidate():
    voltage = np.array([-5.0, -4.7, -3.2, -1.0, 0.0, 1.3, 3.8, 6.0, 9.0, 12.0])
    current = np.exp(voltage / 3.0)
    result = analyze_langmuir_probe(voltage, current)
    assert np.all(np.isfinite(result["dIdV_A_per_V"]))
    assert np.all(np.isfinite(result["d2IdV2_A_per_V2"]))
    assert result["plasma_potential_dIdV_max_candidate_V"] in (voltage[0], voltage[-1])
    assert result["plasma_potential_d2IdV2_max_candidate_V"] in voltage
    assert result["plasma_potential_status"] == "numerical_candidate_requires_regime_review"
    assert "not a statistical uncertainty" in result["plasma_potential_uncertainty_note"]


def test_langmuir_probe_fit_outputs_keep_physical_regime_unconfirmed():
    voltage, current, kwargs = density_case()
    kwargs.pop("probe_area_m2")
    kwargs.pop("ion_mass_kg")
    kwargs.pop("ion_saturation_coefficient")
    kwargs.pop("sheath_model")
    result = analyze_langmuir_probe(voltage, current, **kwargs)
    assert result["electron_retarding_fit_status"] == "numerical_fit_review_pass_not_physical_validation"
    assert result["electron_retarding_regime_review"]["physical_regime_confirmed"] is False
    assert result["ion_saturation_regime_review"]["physical_ion_saturation_confirmed"] is False
    assert result["ion_saturation_current_physical_regime_confirmed"] is False
    assert result["electron_temperature_uncertainty_status"] == (
        "quadrature_of_fit_statistical_and_explicitly_supplied_components"
    )


def test_langmuir_probe_base_and_optional_outputs_match_registry_contract():
    import json
    from pathlib import Path

    registry_path = Path(__file__).resolve().parents[1] / "DIAGNOSTIC_REGISTRY.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = next(item for item in registry if item.get("diagnostic") == "Langmuir Probe")
    contract_names = {
        spec["name"] for spec in entry["execution_contract"]["output_specs"]
    }
    voltage, current, kwargs = density_case(
        ion_current_uncertainty_A=1.0e-6,
        probe_area_relative_uncertainty=0.02,
    )
    result = analyze_langmuir_probe(voltage, current, **kwargs)
    required_runtime_outputs = {
        "n_points", "voltage_min_V", "voltage_max_V", "current_min_A",
        "current_max_A", "floating_potential_V", "floating_potential_candidates_V",
        "floating_potential_status", "plasma_potential_proxy_V", "max_dIdV_A_per_V",
        "min_dIdV_A_per_V", "ion_saturation_current_A", "ion_saturation_current_method",
        "electron_saturation_current_A", "electron_saturation_current_method",
        "electron_temperature_eV", "electron_temperature_standard_error_eV",
        "electron_retarding_fit_R2", "ion_saturation_fit_R2", "current_convention",
        "current_convention_input", "input_quality", "processed_voltage_V",
        "processed_current_A", "dIdV_A_per_V", "d2IdV2_A_per_V2",
        "plasma_potential_estimate_V", "plasma_potential_dIdV_max_candidate_V",
        "plasma_potential_d2IdV2_max_candidate_V", "electron_retarding_regime_review",
        "ion_saturation_regime_review", "ion_density_estimate_m-3",
        "ion_density_standard_uncertainty_m-3", "ion_density_relative_standard_uncertainty",
        "ion_density_uncertainty_components_relative", "ion_density_uncertainty_status",
        "analysis_scope", "interpretation_warning",
    }
    assert required_runtime_outputs <= result.keys()
    assert required_runtime_outputs <= contract_names



def test_langmuir_probe_allows_ideal_flat_ion_plateau_for_density_inversion():
    voltage = np.linspace(-5.0, 20.0, 101)
    current = -1.0e-3 + 4.0e-5 * np.exp((voltage - 10.0) / 3.0)
    current[voltage <= 0.0] = -1.0e-3
    result = analyze_langmuir_probe(
        voltage,
        current,
        electron_fit_range=(10.0, 20.0),
        ion_fit_range=(-5.0, 0.0),
        probe_area_m2=1.0e-5,
        ion_mass_kg=3.3435837724e-27,
        ion_saturation_coefficient=0.61,
        sheath_model="synthetic flat-plateau case",
    )
    assert result["ion_saturation_fit_normalized_rmse"] is None
    assert result["ion_saturation_fit_status"] == "numerical_fit_review_pass_not_physical_validation"
    assert result["ion_saturation_regime_review"]["numerically_flat_candidate"] is True
    assert result["ion_density_estimate_m-3"] > 0.0


def test_langmuir_probe_public_wrapper_matches_canonical_core():
    from tis.probe_analysis import langmuir_iv_analysis

    voltage, current, kwargs = density_case()
    public_result = analyze_langmuir_probe(voltage, current, **kwargs)
    canonical_result = langmuir_iv_analysis(voltage, current, **kwargs)
    assert public_result.keys() == canonical_result.keys()
    for key in public_result:
        left, right = public_result[key], canonical_result[key]
        if isinstance(left, (float, int)) and not isinstance(left, bool):
            assert left == pytest.approx(right)
        else:
            assert left == right

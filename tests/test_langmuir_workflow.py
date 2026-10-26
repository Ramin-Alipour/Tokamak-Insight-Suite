"""End-to-end contract tests for the registered Langmuir I-V analysis."""
import numpy as np
import pytest

from tokamak_analysis.analysis_registry import get_analysis, resolve_analysis
from tokamak_analysis.langmuir_probe import analyze_langmuir_probe


def _synthetic_sweep():
    voltage = np.linspace(-8.0, 8.0, 33)
    current = -2.0e-3 + 2.0e-5 * np.exp((voltage + 8.0) / 2.5)
    return voltage, current


def test_registered_langmuir_workflow_exposes_a_scoped_executable_contract():
    entry = get_analysis(2)
    assert entry["name"] == "Langmuir Probe"
    assert entry["module"] == "langmuir_probe"
    assert callable(resolve_analysis(2))
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is True
    assert {"voltage_V", "current_A"} <= {
        spec["name"] for spec in contract["input_specs"]
    }
    assert {"floating_potential_V", "electron_temperature_eV",
            "ion_saturation_current_A", "input_quality"} <= {
        spec["name"] for spec in contract["output_specs"]
    }


def test_langmuir_workflow_recovers_synthetic_electron_temperature_and_quality():
    voltage, current = _synthetic_sweep()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-5.0, 3.0)
    )
    assert result["electron_temperature_eV"] == pytest.approx(2.5, rel=1e-5)
    assert result["electron_retarding_fit_R2"] > 0.999999
    assert result["electron_retarding_fit_residual_rms_A"] < 1e-10
    assert result["input_quality"]["processed_unique_voltage_count"] == len(voltage)
    assert result["floating_potential_status"] in {
        "single_zero_crossing", "no_zero_crossing_in_measured_sweep"
    }


def test_langmuir_workflow_normalizes_current_polarity_consistently():
    voltage, current = _synthetic_sweep()
    electron_positive = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-5.0, 3.0)
    )
    ion_positive = analyze_langmuir_probe(
        voltage, -current, electron_fit_range=(-5.0, 3.0),
        current_convention="ion_positive",
    )
    assert ion_positive["electron_temperature_eV"] == pytest.approx(
        electron_positive["electron_temperature_eV"], rel=1e-8
    )
    assert ion_positive["ion_saturation_current_A"] == pytest.approx(
        electron_positive["ion_saturation_current_A"], rel=1e-8
    )
    assert ion_positive["current_convention"] == "ion_positive"
    assert ion_positive["input_quality"]["current_convention_normalized"] == "electron_positive"


def test_langmuir_workflow_aggregates_duplicate_voltages_and_discards_bad_pairs():
    voltage, current = _synthetic_sweep()
    voltage = np.concatenate([voltage, [voltage[5], np.nan]])
    current = np.concatenate([current, [current[5] * 1.1, 100.0]])
    result = analyze_langmuir_probe(voltage, current)
    quality = result["input_quality"]
    assert quality["input_pair_count"] == len(voltage)
    assert quality["discarded_nonfinite_pair_count"] == 1
    assert quality["duplicate_voltage_pair_count_aggregated"] == 1
    assert quality["processed_unique_voltage_count"] == 33


def test_langmuir_density_is_not_estimated_from_descriptive_tail_without_fit_window():
    voltage, current = _synthetic_sweep()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-5.0, 3.0),
        area_m2=1e-4, ion_mass_amu=2.0,
    )
    assert "ion_density_estimate_m-3" not in result
    assert result["ion_density_status"].startswith("not_estimated:")
    assert "ion_density_model" not in result


def test_langmuir_density_requires_valid_temperature_and_selected_ion_branch():
    voltage, current = _synthetic_sweep()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-5.0, 3.0),
        ion_fit_range=(-8.0, -5.0), area_m2=1e-4, ion_mass_amu=2.0,
    )
    assert np.isfinite(result["ion_density_estimate_m-3"])
    assert result["ion_density_estimate_m-3"] > 0
    assert np.isfinite(result["ion_sound_speed_m_s"])
    assert "ion_density_model" in result


@pytest.mark.parametrize("kwargs", [
    {"area_m2": 1e-4},
    {"ion_mass_amu": 2.0},
    {"area_m2": 0.0, "ion_mass_amu": 2.0},
    {"area_m2": 1e-4, "ion_mass_amu": float("inf")},
    {"current_convention": "unknown"},
])
def test_langmuir_rejects_invalid_density_or_polarity_configuration(kwargs):
    voltage, current = _synthetic_sweep()
    with pytest.raises(ValueError):
        analyze_langmuir_probe(voltage, current, **kwargs)


def test_langmuir_does_not_silently_accept_an_unphysical_fit_range():
    voltage, current = _synthetic_sweep()
    with pytest.raises(ValueError):
        analyze_langmuir_probe(
            voltage, current, electron_fit_range=(-10.0, -5.0)
        )

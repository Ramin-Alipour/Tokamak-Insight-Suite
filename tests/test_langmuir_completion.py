"""Completion gates for registered analysis #2 (Langmuir probe)."""
import numpy as np
import pytest

from tokamak_analysis.analysis_registry import get_analysis, resolve_analysis
from tokamak_analysis.langmuir_probe import analyze_langmuir_probe


def _synthetic_sweep():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -1.0e-3 + 2.0e-4 * np.exp((voltage + 10.0) / 3.0)
    return voltage, current


def test_analysis_2_registry_contract_matches_executable_langmuir_pipeline():
    entry = get_analysis(2)
    assert entry["name"] == "Langmuir Probe"
    assert entry["module"] == "langmuir_probe"
    assert entry["function"] == "analyze_langmuir_probe"
    contract = entry["execution_contract"]
    assert contract["execution_level"] == "analysis_pipeline"
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is True
    inputs = {item["name"]: item for item in contract["input_specs"]}
    assert {"voltage_V", "current_A"} <= set(inputs)
    assert inputs["voltage_V"]["required"] is True
    assert inputs["current_A"]["required"] is True
    assert {"floating_potential_V", "electron_temperature_eV",
            "ion_density_estimate_m-3"} <= {
                item["name"] for item in contract["output_specs"]
            }
    assert callable(resolve_analysis(2)) is True
    assert resolve_analysis(2) is analyze_langmuir_probe


def test_analysis_2_recovers_synthetic_electron_temperature_and_quality_metrics():
    voltage, current = _synthetic_sweep()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-8.0, 0.0)
    )
    assert result["electron_temperature_eV"] == pytest.approx(3.0, rel=0.08)
    assert result["electron_retarding_fit_points"] >= 4
    assert np.isfinite(result["electron_retarding_fit_R2"])
    assert np.isfinite(result["electron_retarding_fit_residual_rms_A"])
    assert result["input_quality"]["processed_unique_voltage_count"] == len(voltage)
    assert result["floating_potential_status"] == "single_zero_crossing"


def test_analysis_2_density_requires_explicit_ion_window_and_positive_temperature_fit():
    voltage, current = _synthetic_sweep()
    without_window = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-8.0, 0.0),
        area_m2=1.0e-6, ion_mass_amu=2.0,
    )
    assert "ion_density_estimate_m-3" not in without_window
    assert without_window["ion_density_status"].startswith("not_estimated:")

    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-8.0, 0.0),
        ion_fit_range=(-10.0, -5.0), area_m2=1.0e-6, ion_mass_amu=2.0,
    )
    assert np.isfinite(result["ion_density_estimate_m-3"])
    assert result["ion_density_estimate_m-3"] > 0
    assert np.isfinite(result["ion_sound_speed_m_s"])


@pytest.mark.parametrize(
    "voltage,current",
    [
        ([0.0] * 8, list(range(8))),
        ([0.0, 1.0, 2.0], [0.0, 1.0]),
        ([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, "bad"],
         list(range(8))),
    ],
)
def test_analysis_2_rejects_malformed_or_insufficient_sweeps(voltage, current):
    with pytest.raises((TypeError, ValueError)):
        analyze_langmuir_probe(voltage, current)

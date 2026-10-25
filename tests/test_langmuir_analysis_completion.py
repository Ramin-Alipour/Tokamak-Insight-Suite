"""Regression tests for the Langmuir I–V analysis contract."""
import numpy as np
import pytest

from tis.probe_analysis import langmuir_iv_analysis, probe_iv_summary


def _synthetic_sweep():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2.0e-3 + 1.0e-4 * np.exp(voltage / 4.0)
    return voltage, current


def test_langmuir_exponential_fit_recovers_electron_temperature():
    voltage, current = _synthetic_sweep()
    result = langmuir_iv_analysis(
        voltage,
        current,
        electron_fit_range=(0.0, 10.0),
        ion_fit_range=(-10.0, -5.0),
    )
    assert result["electron_temperature_eV"] == pytest.approx(4.0, rel=2e-3)
    assert result["electron_retarding_fit_R2"] > 0.999
    assert result["floating_potential_status"] == "single_zero_crossing"
    assert result["floating_potential_V"] is not None
    assert result["input_quality"]["processed_unique_voltage_count"] == voltage.size


def test_langmuir_current_polarity_conventions_are_equivalent():
    voltage, current = _synthetic_sweep()
    electron_positive = langmuir_iv_analysis(
        voltage, current, electron_fit_range=(0.0, 10.0), ion_fit_range=(-10.0, -5.0)
    )
    ion_positive = langmuir_iv_analysis(
        voltage, -current, electron_fit_range=(0.0, 10.0), ion_fit_range=(-10.0, -5.0),
        current_convention="ion_positive",
    )
    assert ion_positive["electron_temperature_eV"] == pytest.approx(
        electron_positive["electron_temperature_eV"], rel=1e-8
    )
    assert ion_positive["floating_potential_V"] == pytest.approx(
        electron_positive["floating_potential_V"], abs=1e-10
    )


def test_langmuir_duplicate_and_nonfinite_pairs_are_audited():
    voltage, current = _synthetic_sweep()
    voltage = np.r_[voltage, voltage[20], np.nan]
    current = np.r_[current, current[20] * 1.1, 0.0]
    result = langmuir_iv_analysis(voltage, current)
    quality = result["input_quality"]
    assert quality["input_pair_count"] == 83
    assert quality["discarded_nonfinite_pair_count"] == 1
    assert quality["duplicate_voltage_pair_count_aggregated"] == 1
    assert quality["processed_unique_voltage_count"] == 81


def test_langmuir_ambiguous_zero_crossings_are_not_collapsed():
    voltage = np.arange(9.0)
    current = np.array([-2.0, -1.0, 1.0, 2.0, -1.0, -2.0, 1.0, 2.0, 3.0])
    result = langmuir_iv_analysis(voltage, current)
    assert len(result["floating_potential_candidates_V"]) == 3
    assert result["floating_potential_V"] is None
    assert result["floating_potential_status"] == "ambiguous_multiple_zero_crossings"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_convention": "unknown"},
        {"electron_fit_range": (1.0, 1.0)},
        {"ion_fit_range": (0.0, 1.0), "area_m2": 1e-6},
        {"area_m2": 1e-6, "ion_mass_amu": 0.0},
    ],
)
def test_langmuir_rejects_invalid_model_inputs(kwargs):
    voltage, current = _synthetic_sweep()
    with pytest.raises(ValueError):
        langmuir_iv_analysis(voltage, current, **kwargs)


def test_probe_summary_reports_nonfinite_and_insufficient_input_safely():
    with pytest.raises(ValueError, match="At least 5 finite"):
        probe_iv_summary([0.0, np.nan, 1.0, 2.0], [1.0, 2.0, np.nan, 4.0])

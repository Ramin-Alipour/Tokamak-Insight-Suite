"""Regression tests for the canonical Langmuir probe analysis pipeline."""
import numpy as np
import pytest

from tis.probe_analysis import langmuir_iv_analysis, probe_iv_summary


def synthetic_iv():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2.0e-3 + 1.5e-4 * np.exp((voltage + 10.0) / 4.0)
    return voltage, current


def test_langmuir_synthetic_exponential_recovers_temperature_and_fit_quality():
    voltage, current = synthetic_iv()
    result = langmuir_iv_analysis(voltage, current, electron_fit_range=(-8.0, -2.0))
    assert result["electron_temperature_eV"] == pytest.approx(4.0, rel=0.08)
    assert result["electron_retarding_fit_R2"] > 0.99
    assert result["electron_retarding_fit_residual_rms_A"] >= 0
    assert result["electron_retarding_fit_points"] >= 4
    assert result["electron_temperature_standard_error_eV"] is None or result["electron_temperature_standard_error_eV"] >= 0


def test_current_polarity_conventions_produce_equivalent_physics():
    voltage, current = synthetic_iv()
    a = langmuir_iv_analysis(voltage, current, electron_fit_range=(-8.0, -2.0))
    b = langmuir_iv_analysis(voltage, -current, electron_fit_range=(-8.0, -2.0), current_convention="ion_positive")
    assert b["electron_temperature_eV"] == pytest.approx(a["electron_temperature_eV"], rel=1e-8)
    assert b["floating_potential_V"] == pytest.approx(a["floating_potential_V"], abs=1e-10)
    assert b["current_convention"] == "ion_positive"
    assert b["input_quality"]["current_convention_normalized"] == "electron_positive"


def test_invalid_fit_windows_and_conventions_are_rejected():
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="electron_fit_range"):
        langmuir_iv_analysis(voltage, current, electron_fit_range=(2.0, 2.0))
    with pytest.raises(ValueError, match="electron_fit_range"):
        langmuir_iv_analysis(voltage, current, electron_fit_range=("bad", 2.0))
    with pytest.raises(ValueError, match="current_convention"):
        langmuir_iv_analysis(voltage, current, current_convention="automatic")


def test_nonfinite_pairs_are_discarded_and_duplicate_voltage_is_averaged():
    voltage, current = synthetic_iv()
    voltage = np.insert(voltage, [3, 4], [voltage[3], np.nan])
    current = np.insert(current, [3, 4], [current[3] * 1.02, 100.0])
    result = langmuir_iv_analysis(voltage, current)
    assert result["input_quality"]["discarded_nonfinite_pair_count"] == 1
    assert result["input_quality"]["duplicate_voltage_pair_count_aggregated"] == 1
    assert result["input_quality"]["processed_unique_voltage_count"] == 81
    assert result["n_points"] == 81


def test_multiple_crossings_are_flagged_as_ambiguous():
    voltage = np.arange(-4.0, 5.0)
    current = np.array([-1.0, 1.0, -1.0, 1.0, 2.0, 1.0, -1.0, 1.0, 2.0])
    result = langmuir_iv_analysis(voltage, current)
    assert result["floating_potential_V"] is None
    assert len(result["floating_potential_candidates_V"]) > 1
    assert result["floating_potential_status"] == "ambiguous_multiple_zero_crossings"


def test_density_requires_valid_temperature_fit_and_geometry():
    voltage, current = synthetic_iv()
    no_fit = langmuir_iv_analysis(voltage, current, area_m2=1e-5, ion_mass_amu=2.0)
    assert "not_estimated" in no_fit["ion_density_status"]
    with pytest.raises(ValueError, match="valid positive electron temperature"):
        langmuir_iv_analysis(voltage, current, ion_fit_range=(-10.0, -8.0),
                             area_m2=1e-5, ion_mass_amu=2.0)
    result = langmuir_iv_analysis(voltage, current, electron_fit_range=(-8.0, -2.0),
                                  ion_fit_range=(-10.0, -8.0), area_m2=1e-5, ion_mass_amu=2.0)
    assert result["ion_density_estimate_m-3"] > 0
    assert result["ion_sound_speed_m_s"] > 0
    assert "model" in result["ion_density_model"]


def test_probe_summary_does_not_infer_floating_potential_for_nonmonotonic_current():
    result = probe_iv_summary(np.arange(5.0), np.array([-2.0, 1.0, -1.0, 2.0, 3.0]))
    assert result["floating_potential_interpolated_V"] is None

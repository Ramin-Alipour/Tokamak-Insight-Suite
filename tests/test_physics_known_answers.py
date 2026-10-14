"""Independent known-answer checks for core tokamak physics conversions.

These tests validate equation implementation and units against analytic answers.
They are not experimental validation of any device diagnostic.
"""
import numpy as np
import pytest

from tis.core import (
    AMU,
    E_CHARGE,
    EPS0,
    M_E,
    compare_condition_signals,
    corrected_line_ratio,
    electron_cyclotron_resonance_field,
    ion_sound_speed_m_s,
    mach_number_from_velocity,
    reflectometry_cutoff_density,
)


def test_ion_sound_speed_matches_equation_in_si_units():
    te_eV, ti_eV, mass_amu = 14.0, 2.8, 1.0073
    expected = np.sqrt(E_CHARGE * (te_eV + ti_eV) / (mass_amu * AMU))
    result = ion_sound_speed_m_s(te_eV, ti_eV, mass_amu)
    assert result["ion_sound_speed_m_s"] == pytest.approx(expected, rel=1e-13)
    assert result["ion_sound_speed_m_s"] / 1000 == pytest.approx(40.115, rel=2e-4)


def test_mach_number_preserves_flow_direction_and_units():
    # A 20 km/s flow divided by a 40 km/s sound speed has Mach magnitude 0.5.
    assert mach_number_from_velocity(20_000.0, 40_000.0) == pytest.approx(0.5)
    assert mach_number_from_velocity(-20_000.0, 40_000.0) == pytest.approx(-0.5)


def test_o_mode_reflectometry_cutoff_density_matches_plasma_frequency():
    frequency_hz = 60e9
    expected = EPS0 * M_E * (2 * np.pi * frequency_hz) ** 2 / E_CHARGE**2
    result = reflectometry_cutoff_density(frequency_hz, mode="O")
    assert result["cutoff_density_m-3"] == pytest.approx(expected, rel=1e-13)


def test_electron_cyclotron_field_matches_frequency_relation():
    frequency_hz, harmonic = 110e9, 2
    expected = 2 * np.pi * frequency_hz * M_E / (E_CHARGE * harmonic)
    result = electron_cyclotron_resonance_field(frequency_hz, harmonic)
    assert result["resonance_field_T"] == pytest.approx(expected, rel=1e-13)


def test_instrument_response_corrected_line_ratio():
    result = corrected_line_ratio(120.0, 60.0, response_1=2.0, response_2=0.5)
    assert result["raw_intensity_ratio"] == pytest.approx(2.0)
    assert result["instrument_response_corrected_ratio"] == pytest.approx(0.5)


def test_condition_comparison_uses_signed_difference_and_withholds_unstable_mean_percent():
    baseline = np.array([-1.0, 1.0, -1.0, 1.0])
    treatment = np.array([-0.5, 0.5, -0.5, 0.5])
    result = compare_condition_signals(baseline, treatment)
    assert result["mean_difference_treatment_minus_baseline"] == pytest.approx(0.0)
    assert result["mean_change_percent_relative_to_abs_baseline_mean"] is None
    assert result["baseline_mean_percent_change_status"] == "UNSTABLE_NEAR_ZERO_BASELINE"
    assert result["RMS_change_percent"] == pytest.approx(-50.0)

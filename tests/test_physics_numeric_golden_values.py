"""Numerical golden-value checks using fixed SI constants and published relations.

Golden values are independent checkpoints for unit handling and numerical regressions.
They do not constitute experimental validation of a tokamak diagnostic.
"""
import pytest

from tis.core import (
    corrected_line_ratio,
    electron_cyclotron_resonance_field,
    ion_sound_speed_m_s,
    reflectometry_cutoff_density,
)


def test_ir_t1_hydrogen_sound_speed_numeric_golden_value():
    result = ion_sound_speed_m_s(14.0, 2.8, 1.0073)
    assert result["ion_sound_speed_m_s"] == pytest.approx(40114.9146, rel=2e-7)


def test_60_ghz_o_mode_cutoff_density_numeric_golden_value():
    result = reflectometry_cutoff_density(60e9, mode="O")
    assert result["cutoff_density_m-3"] == pytest.approx(4.46559339e19, rel=2e-8)


def test_110_ghz_second_harmonic_ecr_field_numeric_golden_value():
    result = electron_cyclotron_resonance_field(110e9, harmonic=2)
    assert result["resonance_field_T"] == pytest.approx(1.96481272, rel=2e-8)


def test_line_ratio_response_correction_numeric_golden_value():
    result = corrected_line_ratio(120.0, 60.0, response_1=2.0, response_2=0.5)
    assert result["instrument_response_corrected_ratio"] == pytest.approx(0.5, abs=1e-14)


def test_x_mode_reflectometry_does_not_silently_apply_o_mode_formula():
    with pytest.raises(ValueError, match="magnetic field"):
        reflectometry_cutoff_density(60e9, mode="X")

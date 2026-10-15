import math
import pytest

from tis.doppler_temperature import (
    doppler_fwhm_nm_for_temperature,
    doppler_temperature_eV,
)


def test_round_trip_temperature_with_instrument_correction():
    expected = 40.1
    lam = 464.74
    mass = 12.0  # C-12 ion mass, approximate atomic mass in amu
    inst = 0.039
    doppler = doppler_fwhm_nm_for_temperature(lam, mass, expected)
    observed = math.sqrt(doppler**2 + inst**2)
    result = doppler_temperature_eV(
        lam, observed, mass, instrumental_fwhm_nm=inst
    )
    assert result.temperature_eV == pytest.approx(expected, rel=1e-12)
    assert result.doppler_fwhm_nm == pytest.approx(doppler)


def test_missing_instrument_width_is_not_silently_accepted():
    with pytest.raises(ValueError, match="instrumental_fwhm_nm is required"):
        doppler_temperature_eV(656.28, 0.1, 1.0, instrumental_fwhm_nm=None)


def test_unresolved_width_is_rejected():
    with pytest.raises(ValueError, match="unresolved"):
        doppler_temperature_eV(656.28, 0.04, 1.0, instrumental_fwhm_nm=0.04)


def test_invalid_width_rejected():
    with pytest.raises(ValueError):
        doppler_temperature_eV(656.28, -0.1, 1.0, instrumental_fwhm_nm=0.01)


def test_uncertainty_requires_both_inputs():
    with pytest.raises(ValueError, match="both width uncertainties"):
        doppler_temperature_eV(
            464.74, 0.07, 12.0, instrumental_fwhm_nm=0.039,
            observed_fwhm_uncertainty_nm=0.001,
        )

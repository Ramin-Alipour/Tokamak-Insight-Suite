"""Independent analytical and synthetic regression tests for Doppler spectroscopy.

These tests validate the equations and implementation on constructed spectra.
They are not experimental validation of IR-T1 measurements or a reconstruction
of the original spectrometer data.
"""
import numpy as np
import pytest

from tis.core import (
    AMU,
    C_LIGHT,
    E_CHARGE,
    analyze_doppler_line_end_to_end,
    doppler_temperature_eV,
    doppler_velocity_m_s,
)


def test_doppler_temperature_matches_independent_si_expression():
    sigma_nm = 0.012
    wavelength_nm = 500.0
    mass_amu = 47.9
    expected = (
        mass_amu * AMU * C_LIGHT**2 / E_CHARGE
        * (sigma_nm / wavelength_nm) ** 2
    )
    assert doppler_temperature_eV(sigma_nm, wavelength_nm, mass_amu) == pytest.approx(
        expected, rel=1e-13
    )


def test_doppler_velocity_sign_and_units_are_explicit():
    # A +0.01 nm shift at 500 nm corresponds to about +6 km/s.
    velocity = doppler_velocity_m_s(0.01, 500.0)
    assert velocity == pytest.approx(C_LIGHT * 0.01 / 500.0, rel=1e-13)
    assert velocity > 0
    assert velocity == pytest.approx(5995.84916, rel=2e-6)


def test_end_to_end_synthetic_gaussian_recovers_known_temperature_and_velocity():
    # Construct a Gaussian line with a known thermal width and a known +0.006 nm shift.
    rest_nm = 500.0
    mass_amu = 47.9
    target_temperature_eV = 12.0
    instrumental_sigma_nm = 0.008
    target_velocity_m_s = 3_000.0
    center_nm = rest_nm * (1.0 + target_velocity_m_s / C_LIGHT)
    thermal_sigma_nm = rest_nm * np.sqrt(
        target_temperature_eV * E_CHARGE / (mass_amu * AMU * C_LIGHT**2)
    )
    observed_sigma_nm = np.sqrt(
        thermal_sigma_nm**2 + instrumental_sigma_nm**2
    )

    pixels = np.arange(801, dtype=float)
    wavelengths = 499.7 + pixels * (0.6 / 800.0)
    intensity = 7.0 + 1200.0 * np.exp(
        -0.5 * ((wavelengths - center_nm) / observed_sigma_nm) ** 2
    )
    result = analyze_doppler_line_end_to_end(
        raw_coordinate=pixels,
        intensity=intensity,
        reference_pixels=[0.0, 400.0, 800.0],
        reference_wavelengths=[499.7, 500.0, 500.3],
        rest_wavelength=rest_nm,
        ion_mass_amu=mass_amu,
        fit_min_wavelength=499.94,
        fit_max_wavelength=500.06,
        instrument_sigma=instrumental_sigma_nm,
    )

    assert result["doppler_temperature_eV"] == pytest.approx(
        target_temperature_eV, rel=0.02
    )
    assert result["line_of_sight_velocity_m_s"] == pytest.approx(
        target_velocity_m_s, abs=250.0
    )
    assert result["corrected_thermal_doppler_sigma"] > 0
    assert result["calibration_rms_residual_in_wavelength_units"] < 1e-9


def test_end_to_end_doppler_analysis_rejects_unresolved_instrument_width():
    pixels = np.arange(101, dtype=float)
    wavelengths = 499.9 + pixels * 0.002
    intensity = 3.0 + 100.0 * np.exp(-0.5 * ((wavelengths - 500.0) / 0.003) ** 2)
    with pytest.raises(ValueError, match="unresolved"):
        analyze_doppler_line_end_to_end(
            raw_coordinate=pixels,
            intensity=intensity,
            reference_pixels=[0.0, 50.0, 100.0],
            reference_wavelengths=[499.9, 500.0, 500.1],
            rest_wavelength=500.0,
            ion_mass_amu=47.9,
            fit_min_wavelength=499.97,
            fit_max_wavelength=500.03,
            instrument_sigma=0.004,
        )

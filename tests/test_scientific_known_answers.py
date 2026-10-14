"""Independent known-answer regression tests for tokamak physics routines.

These tests verify equations, units, sign conventions, and edge handling with
analytic inputs. They are not a substitute for experimental calibration or
device-specific validation against published measurements.
"""
import numpy as np
import pytest

from tis.core import (
    E_CHARGE, EPS0, M_E, AMU, C_LIGHT,
    doppler_temperature_eV,
    doppler_velocity_m_s,
    ion_sound_speed_m_s,
    mach_number_from_velocity,
    reflectometry_cutoff_density,
    electron_cyclotron_resonance_field,
    linear_phase_density,
    integrate_profile,
    fluctuation_flux,
    equilibrium_grid_derivatives,
    corrected_line_ratio,
)


def test_doppler_temperature_matches_nonrelativistic_thermal_width():
    sigma_lambda, lambda0, mass_amu = 0.02, 500.0, 40.0
    expected = mass_amu * AMU * C_LIGHT**2 / E_CHARGE * (sigma_lambda / lambda0)**2
    assert doppler_temperature_eV(sigma_lambda, lambda0, mass_amu) == pytest.approx(expected)


def test_doppler_velocity_sign_and_units():
    # Δλ/λ0 = v/c; the implementation's sign convention is wavelength-shift based.
    assert doppler_velocity_m_s(0.1, 500.0) == pytest.approx(C_LIGHT * 0.1 / 500.0)
    assert doppler_velocity_m_s(-0.1, 500.0) == pytest.approx(-C_LIGHT * 0.1 / 500.0)


def test_ion_sound_speed_matches_supplied_model():
    te, ti, mass = 20.0, 5.0, 2.0
    result = ion_sound_speed_m_s(te, ti, mass)
    expected = np.sqrt(E_CHARGE * (te + ti) / (mass * AMU))
    assert result["sound_speed_m_s"] == pytest.approx(expected)


def test_mach_number_preserves_flow_sign():
    np.testing.assert_allclose(mach_number_from_velocity([-2000.0, 0.0, 2000.0], 1000.0), [-2.0, 0.0, 2.0])


def test_o_mode_reflectometry_cutoff_density_known_answer():
    f = 60e9
    result = reflectometry_cutoff_density(f, "O")
    expected = EPS0 * M_E * (2 * np.pi * f)**2 / E_CHARGE**2
    assert result["cutoff_density_m-3"] == pytest.approx(expected)
    assert result["mode"] == "O"


def test_x_mode_reflectometry_refuses_underspecified_model():
    with pytest.raises(ValueError, match="X-mode"):
        reflectometry_cutoff_density(60e9, "X")


def test_fundamental_and_second_harmonic_ec_resonance():
    f = 100e9
    fundamental = electron_cyclotron_resonance_field(f, 1)["resonance_field_T"]
    second = electron_cyclotron_resonance_field(f, 2)["resonance_field_T"]
    assert fundamental == pytest.approx(2 * np.pi * f * M_E / E_CHARGE)
    assert second == pytest.approx(fundamental / 2)


def test_interferometer_phase_requires_explicit_sign_and_returns_line_integral():
    f = 140e9
    phase = np.array([0.1, 0.2])
    with pytest.raises(ValueError, match="sign_convention"):
        linear_phase_density(phase, f)
    positive = linear_phase_density(phase, f, +1)
    negative = linear_phase_density(phase, f, -1)
    np.testing.assert_allclose(positive, -negative)
    assert np.all(positive > 0)


def test_profile_integral_of_constant_and_linear_profiles():
    x = np.linspace(0.0, 2.0, 101)
    constant = integrate_profile(x, np.full_like(x, 3.0))
    linear = integrate_profile(x, 2.0 * x + 1.0)
    assert constant["integral_trapezoid"] == pytest.approx(6.0)
    assert linear["integral_trapezoid"] == pytest.approx(6.0)


def test_fluctuation_flux_is_zero_for_uncorrelated_constant_density():
    n = np.full(128, 4e18)
    vr = np.sin(np.linspace(0, 8 * np.pi, 128))
    result = fluctuation_flux(n, vr)
    assert result["particle_flux_m-2_s-1"] == pytest.approx(0.0, abs=1e-12)


def test_equilibrium_grid_derivatives_match_analytic_flux_surface():
    R = np.linspace(1.0, 2.0, 9)
    Z = np.linspace(-0.5, 0.5, 7)
    RR, ZZ = np.meshgrid(R, Z)
    psi = RR**2 + ZZ**2
    result = equilibrium_grid_derivatives(R, Z, psi, 2.0)
    np.testing.assert_allclose(result["dpsi_dR"], 2 * RR, atol=1e-12)
    np.testing.assert_allclose(result["dpsi_dZ"], 2 * ZZ, atol=1e-12)
    np.testing.assert_allclose(result["BR_T"], -2 * ZZ / RR, atol=1e-12)
    np.testing.assert_allclose(result["BZ_T"], 2.0, atol=1e-12)


def test_equilibrium_grid_rejects_wrong_grid_shape():
    with pytest.raises(ValueError, match="shape"):
        equilibrium_grid_derivatives([1.0, 1.5, 2.0], [-1.0, 0.0, 1.0], np.ones((2, 3)), 1.0)


def test_spectroscopic_line_ratio_response_correction():
    result = corrected_line_ratio(20.0, 10.0, response_1=2.0, response_2=1.0)
    assert result["raw_intensity_ratio"] == pytest.approx(2.0)
    assert result["instrument_response_corrected_ratio"] == pytest.approx(1.0)


def test_spectroscopic_line_ratio_rejects_nonpositive_response():
    with pytest.raises(ValueError, match="response"):
        corrected_line_ratio(20.0, 10.0, response_1=0.0, response_2=1.0)

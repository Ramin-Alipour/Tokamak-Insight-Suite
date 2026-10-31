import numpy as np
import pytest

from tokamak_analysis.thomson_kinetic_models import (
    relativistic_maxwell_juttner_spectrum,
    collective_kappa_spectral_density,
)


def test_relativistic_maxwell_juttner_spectrum_is_normalized_and_finite():
    wavelength = np.linspace(800.0, 1328.0, 801)
    result = relativistic_maxwell_juttner_spectrum(
        wavelength, 1064.0, np.pi/2.0, 500.0,
        flow_projection_m_s=2.0e4,
        radial_order=16, angular_order=8,
    )
    spectrum = result["spectral_density_per_nm"]
    assert spectrum.shape == wavelength.shape
    assert np.all(np.isfinite(spectrum))
    assert np.all(spectrum >= 0)
    np.testing.assert_allclose(np.trapz(spectrum, wavelength), 1.0, rtol=2e-3)
    assert result["captured_probability"] >= 0.995
    assert result["relativistic_kinematics"] is True
    assert result["recoil_included"] is False


def test_collective_kappa_uses_dielectric_response_and_non_maxwellian_electrons():
    wavelength = np.linspace(1040.0, 1088.0, 241)
    common = dict(
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2.0,
        electron_density_m3=1.0e19,
        electron_temperature_eV=150.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=[-np.sin(np.pi/4), np.cos(np.pi/4), 0.0],
        scattered_direction=[np.sin(np.pi/4), np.cos(np.pi/4), 0.0],
    )
    low_kappa = collective_kappa_spectral_density(
        wavelength, kappa=2.2, **common
    )
    high_kappa = collective_kappa_spectral_density(
        wavelength, kappa=8.0, **common
    )
    assert low_kappa["spectral_density_per_nm"].shape == wavelength.shape
    assert np.all(np.isfinite(low_kappa["dielectric_function"]))
    assert np.all(np.isfinite(low_kappa["electron_susceptibility"]))
    assert np.all(low_kappa["spectral_density_per_nm"] >= 0)
    assert not np.allclose(
        low_kappa["spectral_density_per_nm"],
        high_kappa["spectral_density_per_nm"],
        rtol=1e-3, atol=0.0,
    )
    assert low_kappa["model"].startswith("unmagnetized collective Thomson")


def test_relativistic_maxwell_juttner_spectrum_fit_recovers_synthetic_parameters():
    from tokamak_analysis.thomson_kinetic_models import fit_relativistic_maxwell_juttner

    wavelength = np.linspace(800.0, 1328.0, 401)
    angle = np.pi/2.0
    laser = 1064.0
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    scattered = np.array([np.sin(angle/2), np.cos(angle/2), 0.0])
    forward = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 500.0,
        flow_projection_m_s=2.0e4,
        radial_order=16, angular_order=8,
    )
    signal = 2.0e4*forward["spectral_density_per_nm"] + 10.0 + 0.001*(wavelength-laser)
    result = fit_relativistic_maxwell_juttner(
        wavelength, signal, laser, angle, np.eye(wavelength.size),
        np.full(wavelength.size, 1.0),
        incident_direction=incident, scattered_direction=scattered,
        radial_order=16, angular_order=8,
    )
    assert result["electron_temperature_eV"] == pytest.approx(500.0, rel=0.15)
    assert result["electron_velocity_projection_m_s"] == pytest.approx(2.0e4, abs=2.0e4)
    assert result["relativistic_kinematics"] is True
    assert result["fit_covariance"].shape == (5, 5)

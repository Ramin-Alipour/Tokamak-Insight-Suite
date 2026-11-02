import numpy as np
import pytest

from tokamak_analysis.thomson_kinetic_models import (
    fit_relativistic_maxwell_juttner_channels,
    relativistic_maxwell_juttner_spectrum,
)


def test_joint_relativistic_mj_fit_uses_full_cross_channel_covariance():
    wavelength = np.linspace(800.0, 1328.0, 241)
    angle = np.pi / 2
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    scattered = np.array([np.sin(angle/2), np.cos(angle/2), 0.0])
    shape = relativistic_maxwell_juttner_spectrum(
        wavelength, 1064.0, angle, 450.0,
        flow_projection_m_s=7.0e5,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=16, angular_order=8,
    )["spectral_density_per_nm"]
    spectra = np.vstack([
        2.0e4*shape + 12.0 + 0.001*(wavelength-1064.0),
        1.3e4*shape + 8.0 - 0.0005*(wavelength-1064.0),
    ])
    n = wavelength.size
    marginal = np.eye(2*n)
    marginal[:n, :n] *= 1.0
    marginal[n:, n:] *= 1.2
    marginal[:n, n:] = np.eye(n)*0.15
    marginal[n:, :n] = np.eye(n)*0.15
    result = fit_relativistic_maxwell_juttner_channels(
        wavelength, spectra, 1064.0, angle, np.eye(n),
        incident_direction=incident, scattered_direction=scattered,
        incident_polarization=[0.0, 0.0, 1.0],
        joint_spectral_covariance=marginal,
        initial_temperature_eV=500.0,
        initial_flow_projection_m_s=5.0e5,
        radial_order=16, angular_order=8,
    )
    assert result["electron_temperature_eV"] == pytest.approx(450.0, rel=0.12)
    assert result["electron_velocity_projection_m_s"] == pytest.approx(7.0e5, abs=2.0e5)
    assert result["spectral_weighting_status"] == "full_channel_major_generalized_least_squares_including_cross_channel_covariance"
    assert result["fit_covariance"].shape == (8, 8)
    assert result["fitted_spectra"].shape == spectra.shape


def test_joint_relativistic_mj_rejects_non_positive_definite_covariance():
    wavelength = np.linspace(800.0, 1328.0, 41)
    angle = np.pi/2
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    scattered = np.array([np.sin(angle/2), np.cos(angle/2), 0.0])
    with pytest.raises(ValueError, match="positive definite"):
        fit_relativistic_maxwell_juttner_channels(
            wavelength, np.ones((2, wavelength.size)), 1064.0, angle,
            np.eye(wavelength.size), incident_direction=incident,
            scattered_direction=scattered, incident_polarization=[0,0,1],
            joint_spectral_covariance=np.zeros((2*wavelength.size, 2*wavelength.size)),
            initial_temperature_eV=300.0,
        )

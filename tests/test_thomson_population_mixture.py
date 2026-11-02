import numpy as np
import pytest

from tokamak_analysis.thomson_population_mixture import (
    fit_two_population_relativistic_thomson_spectrum,
    two_population_relativistic_thomson_spectrum,
)


def test_two_population_relativistic_thomson_spectrum_is_normalized():
    wavelength = np.linspace(350.0, 750.0, 241)
    result = two_population_relativistic_thomson_spectrum(
        wavelength, 532.0, np.pi/2.0, 100.0, 5000.0, 0.2,
        flow_projection_m_s=0.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    assert np.all(np.isfinite(result["spectral_density_per_nm"]))
    assert np.all(result["spectral_density_per_nm"] >= 0)
    assert np.trapz(result["spectral_density_per_nm"], wavelength) == pytest.approx(1.0, rel=2e-3)
    assert result["hot_population_fraction"] == pytest.approx(0.2)
    assert result["absolute_density_inferred"] is False


def test_two_population_fit_recovers_synthetic_shape_parameters():
    wavelength = np.linspace(350.0, 750.0, 121)
    angle = np.pi/2.0
    incident = np.array([-np.sin(angle/2), np.cos(angle/2), 0.0])
    truth = dict(cold_temperature_eV=100.0, hot_temperature_eV=5000.0,
                 hot_population_fraction=0.2, flow_projection_m_s=3.0e5)
    forward = two_population_relativistic_thomson_spectrum(
        wavelength, 532.0, angle,
        truth["cold_temperature_eV"], truth["hot_temperature_eV"],
        truth["hot_population_fraction"],
        flow_projection_m_s=truth["flow_projection_m_s"],
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    measured = 2.0e4*forward["spectral_density_per_nm"] + 2.0
    result = fit_two_population_relativistic_thomson_spectrum(
        wavelength, measured, 532.0, angle, np.eye(wavelength.size),
        incident_polarization=[0.0, 0.0, 1.0],
        cold_temperature_initial_eV=105.0,
        hot_temperature_initial_eV=4800.0,
        hot_population_fraction_initial=0.19,
        flow_projection_initial_m_s=2.8e5,
        signal_uncertainty=np.full(wavelength.size, 1.0),
        radial_order=12, angular_order=8,
    )
    assert result["cold_electron_temperature_eV"] == pytest.approx(100.0, rel=0.15)
    assert result["hot_electron_temperature_eV"] == pytest.approx(5000.0, rel=0.2)
    assert result["hot_population_fraction"] == pytest.approx(0.2, abs=0.08)
    assert result["electron_velocity_projection_m_s"] == pytest.approx(3.0e5, abs=1.0e5)
    assert result["fit_covariance"].shape == (4, 4)


def test_two_population_fit_uses_automatic_deterministic_initialization():
    wavelength = np.linspace(350.0, 750.0, 81)
    angle = np.pi / 2.0
    forward = two_population_relativistic_thomson_spectrum(
        wavelength, 532.0, angle, 120.0, 4200.0, 0.12,
        flow_projection_m_s=1.5e5,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    measured = 1.7e4 * forward["spectral_density_per_nm"] + 0.5
    result = fit_two_population_relativistic_thomson_spectrum(
        wavelength, measured, 532.0, angle, np.eye(wavelength.size),
        incident_polarization=[0.0, 0.0, 1.0],
        signal_uncertainty=np.full(wavelength.size, 0.5),
        radial_order=12, angular_order=8, max_nfev=350,
    )
    assert result["optimizer_initialization"] == "automatic_deterministic_spectral_moment_multistart"
    assert result["optimizer_multistart_count"] == 3
    assert result["optimizer_seed_parameters_are_not_inferred_outputs"] is True
    assert np.all(np.isfinite(result["fit_covariance"]))

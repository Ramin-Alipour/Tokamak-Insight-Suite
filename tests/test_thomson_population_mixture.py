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
        background_model="linear",
        radial_order=12, angular_order=8,
    )
    assert result["fitted_background_model"] == "linear"
    assert result["fitted_background_coefficients_scaled_wavelength"].shape == (2,)
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


def test_relativistic_maxwell_juttner_radial_quadrature_matches_analytic_partition():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(100.0, 1100.0, 401)
    result = relativistic_maxwell_juttner_spectrum(
        wavelength, 532.0, np.pi/2.0, 5000.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    assert result["quadrature"]["radial_scheme"] == (
        "generalized_Gauss_Laguerre_alpha_1_over_2_in_dimensionless_kinetic_energy_on_infinite_domain"
    )
    assert result["quadrature_to_analytic_normalization_ratio"] == pytest.approx(1.0, rel=1e-10)
    assert result["momentum_probability_coverage"] == pytest.approx(1.0, abs=1e-10)


def test_general_multi_population_forward_reduces_to_two_population_limit():
    from tokamak_analysis.thomson_population_mixture import (
        multi_population_relativistic_thomson_spectrum,
    )
    wavelength = np.linspace(350.0, 750.0, 121)
    angle = np.pi/2.0
    kwargs = dict(
        flow_projection_m_s=2.0e5,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    general = multi_population_relativistic_thomson_spectrum(
        wavelength, 532.0, angle, [100.0, 5000.0], [0.8, 0.2], **kwargs
    )
    legacy = two_population_relativistic_thomson_spectrum(
        wavelength, 532.0, angle, 100.0, 5000.0, 0.2, **kwargs
    )
    np.testing.assert_allclose(
        general["spectral_density_per_nm"],
        legacy["spectral_density_per_nm"], rtol=2e-12, atol=1e-14,
    )
    assert general["population_count"] == 2
    assert np.sum(general["population_number_density_fractions"]) == pytest.approx(1.0)


def test_general_multi_population_forward_uses_number_density_simplex():
    from tokamak_analysis.thomson_population_mixture import (
        multi_population_relativistic_thomson_spectrum,
    )
    wavelength = np.linspace(350.0, 750.0, 121)
    result = multi_population_relativistic_thomson_spectrum(
        wavelength, 532.0, np.pi/2.0, [50.0, 500.0, 5000.0],
        [0.7, 0.2, 0.1], flow_projection_m_s=0.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    assert result["component_spectra_per_nm"].shape == (3, wavelength.size)
    assert result["population_effective_cross_sections_m2_sr"].shape == (3,)
    assert np.trapz(result["spectral_density_per_nm"], wavelength) == pytest.approx(1.0, rel=2e-3)
    with pytest.raises(ValueError, match="sum to one"):
        multi_population_relativistic_thomson_spectrum(
            wavelength, 532.0, np.pi/2.0, [50.0, 500.0], [0.7, 0.2],
            flow_projection_m_s=0.0, incident_polarization=[0.0, 0.0, 1.0],
            radial_order=12, angular_order=8,
        )

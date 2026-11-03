import numpy as np
import pytest

from tokamak_analysis.cts_analysis import analyze_cts_spectrum, cts_spectral_density


def _cts_case():
    f0 = 140e9
    frequency = np.linspace(-5.0e9, 5.0e9, 121)
    geometry = {"scattering_angle_rad": np.deg2rad(110.0)}
    species = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    fractions = [1.0]
    return f0, frequency, geometry, species, fractions


def test_cts_reference_pca_background_is_fitted_as_detector_space_nuisance():
    f0, frequency, geometry, species, fractions = _cts_case()
    truth = {
        "electron_density_m3": 2.0e19,
        "electron_temperature_eV": 1200.0,
        "ion_temperature_eV": 800.0,
        "electron_velocity_projection_m_s": 0.0,
        "ion_velocity_projection_m_s": 1.5e4,
    }
    physical = cts_spectral_density(
        frequency, f0, geometry, **truth,
        ion_species=species, ion_fractions=fractions,
    )
    shape = np.linspace(-1.0, 1.0, frequency.size)
    curvature = shape**2 - np.mean(shape**2)
    reference = np.asarray([
        4.0 + scale * shape + 0.15 * offset * curvature
        for scale, offset in [(-1.0, -1.0), (-0.5, 1.0), (0.0, -0.5),
                              (0.5, 0.5), (1.0, -1.0), (0.8, 1.0)]
    ])
    background = reference.mean(axis=0) + 0.37 * (
        np.linalg.svd(reference-reference.mean(axis=0), full_matrices=False)[2][0]
    )
    measured = 2.4e4 * physical + background
    result = analyze_cts_spectrum(
        frequency, measured, probe_frequency_Hz=f0,
        scattering_geometry=geometry, ion_species=species,
        ion_fractions=fractions, instrument_response=np.eye(frequency.size),
        signal_uncertainty=np.full(frequency.size, 0.01*np.max(measured)),
        electron_density_initial_m3=truth["electron_density_m3"],
        electron_temperature_initial_eV=truth["electron_temperature_eV"],
        ion_temperature_initial_eV=truth["ion_temperature_eV"],
        ion_velocity_initial_m_s=truth["ion_velocity_projection_m_s"],
        background_reference_spectra=reference, background_components=1,
        global_search=False, inversion_method="local_least_squares",
    )
    assert result["background_model"] == "reference_spectrum_PCA"
    assert result["background_pca_components_used"] == 1
    assert result["background_spectrum"].shape == frequency.shape
    assert result["background_coefficients"].shape == (2,)
    assert np.all(np.isfinite(result["fitted_spectrum"]))
    assert result["electron_density_m3"] == pytest.approx(truth["electron_density_m3"], rel=0.1)


def test_cts_reference_pca_background_requires_explicit_component_count():
    f0, frequency, geometry, species, fractions = _cts_case()
    references = np.vstack([np.ones_like(frequency), 1.0 + np.linspace(0.0, 1.0, frequency.size)])
    physical = cts_spectral_density(
        frequency, f0, geometry, electron_density_m3=2e19,
        electron_temperature_eV=1200.0, ion_temperature_eV=800.0,
        electron_velocity_projection_m_s=0.0, ion_velocity_projection_m_s=0.0,
        ion_species=species, ion_fractions=fractions,
    )
    with pytest.raises(ValueError, match="background_components must be an explicit"):
        analyze_cts_spectrum(
            frequency, physical + 1.0, probe_frequency_Hz=f0,
            scattering_geometry=geometry, ion_species=species,
            ion_fractions=fractions, instrument_response=np.eye(frequency.size),
            electron_density_initial_m3=2e19, electron_temperature_initial_eV=1200.0,
            ion_temperature_initial_eV=800.0, background_reference_spectra=references,
            global_search=False, inversion_method="local_least_squares",
        )


def test_cts_reference_pca_background_reports_conditional_coefficient_and_spectrum_covariance():
    f0, frequency, geometry, species, fractions = _cts_case()
    truth = {
        "electron_density_m3": 2.0e19,
        "electron_temperature_eV": 1200.0,
        "ion_temperature_eV": 800.0,
        "electron_velocity_projection_m_s": 0.0,
        "ion_velocity_projection_m_s": 1.5e4,
    }
    physical = cts_spectral_density(
        frequency, f0, geometry, **truth,
        ion_species=species, ion_fractions=fractions,
    )
    coordinate = np.linspace(-1.0, 1.0, frequency.size)
    references = np.asarray([
        3.0 + scale*coordinate + curvature*(coordinate**2-np.mean(coordinate**2))
        for scale, curvature in [(-1.0, -0.2), (-0.5, 0.3), (0.0, -0.4),
                                 (0.5, 0.4), (1.0, -0.1), (0.8, 0.2)]
    ])
    measured = 2.4e4*physical + references.mean(axis=0)
    sigma = np.full(frequency.size, 0.01*np.max(measured))
    result = analyze_cts_spectrum(
        frequency, measured, probe_frequency_Hz=f0,
        scattering_geometry=geometry, ion_species=species,
        ion_fractions=fractions, instrument_response=np.eye(frequency.size),
        signal_uncertainty=sigma,
        electron_density_initial_m3=truth["electron_density_m3"],
        electron_temperature_initial_eV=truth["electron_temperature_eV"],
        ion_temperature_initial_eV=truth["ion_temperature_eV"],
        ion_velocity_initial_m_s=truth["ion_velocity_projection_m_s"],
        background_reference_spectra=references, background_components=2,
        background_reference_basis_mc_samples=40,
        global_search=False, inversion_method="local_least_squares",
    )
    assert result["background_reference_basis_covariance"].shape == (3*frequency.size, 3*frequency.size)
    assert result["background_reference_basis_uncertainty_propagated"] is False
    assert "not_yet_propagated" in result["background_reference_basis_uncertainty_status"]
    coefficient_covariance = result["background_coefficient_covariance"]
    spectrum_covariance = result["background_spectrum_covariance"]
    assert coefficient_covariance.shape == (3, 3)
    assert spectrum_covariance.shape == (frequency.size, frequency.size)
    assert result["background_spectrum_uncertainty"].shape == frequency.shape
    assert result["background_reference_basis_uncertainty_propagated"] is False
    assert result["background_uncertainty_status"] == (
        "conditional_linear_nuisance_covariance_from_supplied_spectral_uncertainty"
    )
    np.testing.assert_allclose(coefficient_covariance, coefficient_covariance.T, atol=1e-12)
    np.testing.assert_allclose(spectrum_covariance, spectrum_covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(coefficient_covariance).min() >= -1e-10
    assert np.all(result["background_spectrum_uncertainty"] >= 0.0)

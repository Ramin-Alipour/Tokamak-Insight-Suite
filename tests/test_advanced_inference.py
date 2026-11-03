import numpy as np
import pytest

from tokamak_analysis.advanced_inference import (
    poisson_deviance_residual,
    poisson_negative_log_likelihood,
    gaussian_process_discrepancy_covariance,
    propagate_paraxial_beam_covariance,
    bootstrap_reference_pca_basis,
    reconstruct_multiview_3d_evdf,
)
from tokamak_analysis.cts_em_background import (
    planck_spectral_radiance,
    thermal_bremsstrahlung_emissivity_si,
    integrate_thermal_microwave_background,
    radiance_to_receiver_spectral_power,
)


def test_poisson_likelihood_is_independent_count_model():
    y = np.array([0, 2, 10, 100])
    mu = np.array([0.5, 2.5, 9.0, 110.0])
    residual = poisson_deviance_residual(y, mu)
    assert residual.shape == y.shape
    assert residual[0] < 0 and residual[-1] < 0
    assert np.sum(residual**2) >= 0
    assert poisson_negative_log_likelihood(y, mu) > 0
    with pytest.raises(ValueError, match="integer-valued"):
        poisson_deviance_residual([0.2, 1.0], [1.0, 1.0])


def test_gp_discrepancy_is_correlated_and_positive_definite():
    x = np.linspace(-1, 1, 31)
    cov = gaussian_process_discrepancy_covariance(x, amplitude=0.2, length_scale=0.3)
    assert cov.shape == (x.size, x.size)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10
    assert cov[0, 1] > cov[0, -1]


def test_paraxial_beam_covariance_propagates_symplectically():
    sigma = np.diag([1e-4, 2e-4, 1e-3, 2e-3])
    free_space = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    result = propagate_paraxial_beam_covariance(sigma, [free_space, free_space])
    assert result["phase_space_covariance"].shape == (3, 4, 4)
    assert result["transverse_position_covariance_m2"].shape == (3, 2, 2)
    assert np.all(np.linalg.eigvalsh(result["phase_space_covariance"][-1]) > 0)


def test_reference_pca_bootstrap_reports_basis_uncertainty():
    rng = np.random.default_rng(4)
    x = np.linspace(-1, 1, 41)
    refs = np.asarray([3 + a*x + b*x*x + rng.normal(0, 0.01, x.size)
                       for a, b in zip(np.linspace(-1, 1, 12), np.linspace(0.2, -0.2, 12))])
    result = bootstrap_reference_pca_basis(refs, components=2, bootstrap_samples=40)
    assert result["bootstrap_parameter_covariance"].shape == (refs.shape[1] * 3, refs.shape[1] * 3)
    assert result["pca_components"].shape == (refs.shape[1], 2)


def test_multiview_3d_evdf_requires_data_supported_rank():
    # Identity spectral kernels make all six velocity voxels independently identifiable.
    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    true = np.array([1., 2., 3., 4., 5., 6.])
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (true * volume)).reshape(3, 2)
    penalty = np.zeros((1, 6))
    result = reconstruct_multiview_3d_evdf(
        kernels, data, np.arange(18).reshape(6, 3), voxel_volume_m3_s3=volume,
        regularization_matrix=penalty, regularization_strength=1e-6,
    )
    assert result["status"] == "identified_from_multi_view_measurements"
    assert result["evdf_m3_s3"] is not None
    one_view = reconstruct_multiview_3d_evdf(
        kernels[:1], data[:1], np.arange(18).reshape(6, 3),
        voxel_volume_m3_s3=volume, regularization_matrix=penalty,
        regularization_strength=1e-6,
    )
    assert one_view["status"] == "conditional_regularized_3d_candidate"
    assert one_view["evdf_m3_s3"] is None


def test_free_free_and_ece_radiative_transfer():
    nu = np.linspace(90e9, 110e9, 5)
    b = planck_spectral_radiance(nu, 2e7)
    assert np.all(b > 0)
    j = thermal_bremsstrahlung_emissivity_si(
        nu, 2e19, [1e19], [1], 1500.0, gaunt_factor=1.2
    )
    assert np.all(j > 0)
    result = integrate_thermal_microwave_background(
        nu, [0.1, 0.2], [1000., 1200.], electron_density_m3=[2e19, 2e19],
        ion_densities_m3=[[1e19], [1e19]], ion_charge_states=[1],
        ece_absorption_coefficient_m_inverse=np.full((2, 5), 0.01),
    )
    assert result["spectral_radiance_W_m2_sr_Hz"].shape == nu.shape
    assert np.all(result["ece_optical_depth"] > 0)



def test_nonlinear_nonnegative_distribution_fit_uses_full_forward_callback():
    from tokamak_analysis.advanced_inference import fit_nonlinear_nonnegative_distribution

    rng = np.random.default_rng(12)
    kernel = rng.uniform(0.2, 1.2, size=(24, 6))
    weights = np.linspace(0.5, 1.0, 6)
    truth = np.array([0.05, 0.1, 0.2, 0.25, 0.25, 0.15])
    truth /= weights @ truth
    measured = 3.0 * (kernel @ truth) + 0.1
    initial = np.ones(6) / (weights @ np.ones(6))
    regularizer = np.diff(np.eye(6), n=2, axis=0)
    result = fit_nonlinear_nonnegative_distribution(
        measured, initial, weights,
        forward_model=lambda pdf: kernel @ pdf,
        regularization_matrix=regularizer,
        regularization_strength=1e-5,
        measurement_uncertainty=np.full(measured.size, 0.01),
        max_nfev=500,
    )
    assert result["status"] == "identified_conditional_distribution_shape"
    assert result["distribution"] is not None
    assert np.all(result["distribution"] >= 0)
    assert result["normalization"] == pytest.approx(1.0, rel=1e-8)



def test_optical_maxwellian_fit_accepts_independent_poisson_count_likelihood():
    from tokamak_analysis.thomson_inference import fit_optical_thomson_poisson_counts

    wavelength = np.linspace(530.0, 532.0, 121)
    center = 531.0
    sigma = 0.18
    rng = np.random.default_rng(21)
    expected = 450.0*np.exp(-0.5*((wavelength-center)/sigma)**2) + 18.0 + 2.0*(wavelength-center)
    counts = rng.poisson(np.maximum(expected, 1.0)).astype(int)
    result = fit_optical_thomson_poisson_counts(
        wavelength, counts, laser_wavelength_nm=531.0,
        scattering_angle_rad=np.pi/2.0,
        instrument_response=np.eye(wavelength.size),
    )
    assert result["likelihood"] == "independent_Poisson"
    assert result["density_inference_available"] is False
    assert "integrated_scattered_power_W" not in result
    assert result["integrated_signal_counts"] > 0
    assert np.isfinite(result["electron_temperature_eV"])
    assert result["reduced_chi_square"] is None


def test_cts_accepts_independent_correlated_model_discrepancy_covariance():
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum, cts_spectral_density
    frequency = np.linspace(-5.0e9, 5.0e9, 81)
    f0 = 140e9
    geometry = {"scattering_angle_rad": np.deg2rad(110.0)}
    species = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    truth = dict(
        electron_density_m3=2.0e19, electron_temperature_eV=1200.0,
        ion_temperature_eV=800.0, electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=1.5e4,
    )
    physical = cts_spectral_density(
        frequency, f0, geometry, **truth, ion_species=species, ion_fractions=[1.0]
    )
    measured = 2.4e4*physical + 1.0
    discrepancy = np.eye(frequency.size) * (1e-3*np.max(measured))**2
    result = analyze_cts_spectrum(
        frequency, measured, probe_frequency_Hz=f0,
        scattering_geometry=geometry, ion_species=species, ion_fractions=[1.0],
        instrument_response=np.eye(frequency.size),
        electron_density_initial_m3=truth["electron_density_m3"],
        electron_temperature_initial_eV=truth["electron_temperature_eV"],
        ion_temperature_initial_eV=truth["ion_temperature_eV"],
        ion_velocity_initial_m_s=truth["ion_velocity_projection_m_s"],
        model_discrepancy_covariance=discrepancy,
        global_search=False, inversion_method="local_least_squares",
    )
    assert result["independent_model_discrepancy_supplied"] is True
    assert result["model_discrepancy_model"].startswith("researcher_supplied_correlated")
    assert "model_discrepancy_scale" not in result



def test_radiance_to_receiver_power_applies_etendue_and_efficiency_once():
    radiance = np.array([1.0, 2.0, 3.0])
    power = radiance_to_receiver_spectral_power(
        radiance, receiver_etendue_m2_sr=0.02,
        end_to_end_efficiency=0.5,
    )
    np.testing.assert_allclose(power, radiance*0.01)

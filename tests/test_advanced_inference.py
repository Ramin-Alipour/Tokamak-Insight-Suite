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
    penalty = np.zeros((0, 6))
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

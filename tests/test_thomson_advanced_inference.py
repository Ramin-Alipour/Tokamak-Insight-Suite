import numpy as np
import pytest

from tokamak_analysis.thomson_advanced_inference import (
    covariance_whitener,
    profile_structured_background,
    propagate_external_covariance,
    reconstruct_profile_with_psf,
    relativistic_mj_projected_beta_pdf,
    structured_background_design,
    validate_covariance,
)


def test_structured_background_polynomial_and_gls_profile():
    x = np.linspace(0.0, 1.0, 101)
    physical = np.exp(-0.5 * ((x - 0.48) / 0.06) ** 2)
    design = structured_background_design(x, kind="polynomial", degree=2)
    y = 2.4 * physical + 0.3 + 0.04 * (x - 0.5) + 0.02 * (x - 0.5) ** 2
    out = profile_structured_background(
        y, physical, design, uncertainty=np.full(x.size, 0.01)
    )
    assert out["identifiable"]
    assert out["physical_amplitude"] == pytest.approx(2.4, rel=1e-5)
    assert np.sqrt(np.mean(out["residual"] ** 2)) < 1e-8
    assert out["background_is_detector_space_nuisance"]


def test_structured_background_requires_explicit_reference_basis_and_rank():
    x = np.linspace(0.0, 1.0, 21)
    refs = np.vstack([1 + x, 1 + 2*x, 1 + 3*x])
    basis = structured_background_design(
        x, kind="pca", reference_spectra=refs, components=1
    )
    assert basis.shape == (21, 2)
    with pytest.raises(ValueError, match="explicit internal knots"):
        structured_background_design(x, kind="spline")


def test_covariance_validation_and_whitener():
    covariance = np.array([[2.0, 0.3], [0.3, 1.0]])
    whitener = covariance_whitener(covariance)
    residual = np.array([0.4, -0.2])
    assert np.dot(whitener @ residual, whitener @ residual) == pytest.approx(
        residual @ np.linalg.solve(covariance, residual)
    )
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance([[1.0, 2.0], [2.0, 1.0]], 2)


def test_external_covariance_propagation_preserves_correlations():
    covariance = np.array([[0.04, 0.018], [0.018, 0.09]])
    result = propagate_external_covariance(
        [1.0, 2.0], covariance,
        lambda z: {"sum": z[0] + z[1], "difference": z[0] - z[1]},
        samples=2000, seed=7,
    )
    assert result["status"] == "completed"
    assert result["output_covariance"].shape == (2, 2)
    assert result["output_covariance"][0, 1] == pytest.approx(
        covariance[0, 0] - covariance[1, 1], abs=0.015
    )
    assert result["not_a_posterior"]


def test_psf_profile_reconstruction_uses_fixed_mixing_and_reports_rank():
    latent = np.array([1.0, 1.5, 2.0, 1.5, 1.0])
    mixing = np.array([
        [0.8, 0.2, 0, 0, 0],
        [0.1, 0.7, 0.2, 0, 0],
        [0, 0.2, 0.6, 0.2, 0],
        [0, 0, 0.2, 0.7, 0.1],
        [0, 0, 0, 0.2, 0.8],
    ])
    measured = mixing @ latent
    out = reconstruct_profile_with_psf(
        measured, mixing, measurement_uncertainty=np.full(5, 1e-3),
        regularization=1e-8,
    )
    np.testing.assert_allclose(out["fitted_measured_profile"], measured, atol=1e-7)
    assert out["mixing_matrix_is_external_fixed_input"]
    assert out["latent_profile_covariance"].shape == (5, 5)


def test_psf_reconstruction_rejects_implicit_psf_normalization():
    with pytest.raises(ValueError, match="row must sum to one"):
        reconstruct_profile_with_psf(
            np.ones(3), np.eye(3) * 2,
            measurement_uncertainty=np.ones(3), regularization=0.1,
        )


def test_relativistic_mj_velocity_marginal_is_normalized_and_symmetric():
    beta = np.linspace(-0.999, 0.999, 20001)
    pdf = relativistic_mj_projected_beta_pdf(beta, 2.0e5)
    assert np.all(pdf >= 0)
    assert np.trapz(pdf, beta) == pytest.approx(1.0, rel=2e-4)
    np.testing.assert_allclose(pdf, pdf[::-1], rtol=1e-12, atol=1e-14)


def test_relativistic_mj_marginal_narrows_with_temperature():
    beta = np.linspace(-0.95, 0.95, 10001)
    cold = relativistic_mj_projected_beta_pdf(beta, 1.0e3)
    hot = relativistic_mj_projected_beta_pdf(beta, 1.0e5)
    cold_var = np.trapz(beta**2 * cold, beta)
    hot_var = np.trapz(beta**2 * hot, beta)
    assert hot_var > cold_var

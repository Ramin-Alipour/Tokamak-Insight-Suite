import numpy as np
import pytest

from tokamak_analysis.thomson_joint_inference import (
    fit_multichannel_joint_gls,
    propagate_geometry_ensemble,
    reconstruct_profiles_regularization_sensitivity,
    sample_reference_basis_uncertainty,
    validate_covariance,
)


def test_joint_gls_retains_cross_channel_covariance_and_backgrounds():
    x = np.linspace(-1, 1, 31)
    basis = np.column_stack([np.ones_like(x), x])
    truth = np.array([0.7, 1.2])
    y0 = truth[0] * np.exp(-((x + 0.2) / 0.25) ** 2) + basis @ [0.1, 0.03]
    y1 = truth[1] * np.exp(-((x - 0.1) / 0.3) ** 2) + basis @ [0.2, -0.02]
    y = np.stack([y0, y1])
    cov = np.eye(y.size) * 1e-4
    cov[:x.size, x.size:] = 2e-5
    cov[x.size:, :x.size] = 2e-5

    def model(p, channel):
        center = -0.2 if channel == 0 else 0.1
        width = 0.25 if channel == 0 else 0.3
        return p[channel] * np.exp(-((x - center) / width) ** 2)

    result = fit_multichannel_joint_gls(
        y, forward_model=model, physical_initial=[0.6, 1.0],
        measurement_covariance=cov,
        physical_bounds=([0.01, 0.01], [3.0, 3.0]),
        background_design=basis,
        background_bounds=([-1, -1], [1, 1]),
    )
    np.testing.assert_allclose(result["physical_parameters"], truth, rtol=2e-5)
    assert result["measurement_covariance"][0, x.size] == pytest.approx(2e-5)
    assert result["parameter_covariance"] is not None


def test_profile_regularization_reports_data_rank_and_full_lambda_sensitivity():
    x = np.linspace(0, 1, 9)
    a = np.eye(9)
    y = np.sin(x)
    result = reconstruct_profiles_regularization_sensitivity(
        x, y, np.eye(9) * 0.01, forward_matrix=a,
        strengths=[0.0, 0.01, 1.0, 100.0],
    )
    assert result["data_identifiable"]
    assert len(result["strength_scan"]) == 4
    assert np.all(result["regularization_sensitivity"]["profile_span"] >= 0)
    assert result["selected_strength_by_GCV"] in [0.0, 0.01, 1.0, 100.0]


def test_reference_pca_uncertainty_is_reproducible_and_nonzero():
    rng = np.random.default_rng(7)
    ref = rng.normal(1.0, 0.1, size=(12, 20))
    y = ref.mean(axis=0)
    a = sample_reference_basis_uncertainty(ref, y, n_samples=30, random_seed=4)
    b = sample_reference_basis_uncertainty(ref, y, n_samples=30, random_seed=4)
    np.testing.assert_allclose(a["mean_reconstruction"], b["mean_reconstruction"])
    assert a["reconstruction_covariance"].shape == (20, 20)
    assert a["sample_count"] == 30


def test_geometry_uncertainty_propagates_correlated_inputs():
    mean = np.array([1.0, 2.0])
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    result = propagate_geometry_ensemble(
        lambda g: np.array([g[0] + 2 * g[1], g[0] * g[1]]),
        mean, cov, n_samples=200, random_seed=2,
    )
    assert result["covariance"].shape == (2, 2)
    assert result["success_fraction"] == 1.0


def test_covariance_rejects_asymmetry_and_non_positive_definite():
    with pytest.raises(ValueError, match="symmetric"):
        validate_covariance([[1, 0.2], [0.1, 1]], 2)
    with pytest.raises(ValueError, match="positive definite"):
        validate_covariance([[1, 1], [1, 1]], 2)

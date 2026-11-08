import numpy as np
import pytest

from tokamak_analysis.thomson_joint_uncertainty import (
    assemble_joint_covariance,
    finite_difference_jacobian,
    propagate_joint_covariance,
    propagate_reference_ensemble,
    regularization_sensitivity,
    validate_covariance,
)


def test_joint_covariance_retains_cross_channel_blocks():
    result = assemble_joint_covariance(
        {"channel_1": np.array([[2.0]]), "channel_2": np.array([[3.0]])},
        {("channel_1", "channel_2"): np.array([[0.5]])},
    )
    np.testing.assert_allclose(result["covariance"], [[2.0, 0.5], [0.5, 3.0]])
    assert result["unspecified_cross_covariance_assumed_zero"] == []


def test_joint_covariance_rejects_non_psd_cross_blocks():
    with pytest.raises(ValueError, match="positive semidefinite"):
        assemble_joint_covariance(
            {"a": np.array([[1.0]]), "b": np.array([[1.0]])},
            {("a", "b"): np.array([[2.0]])},
        )


def test_full_parameter_cross_covariance_propagates():
    jac = np.array([[1.0, 2.0], [-1.0, 1.0]])
    covariance = np.array([[1.0, 0.3], [0.3, 4.0]])
    actual = propagate_joint_covariance(jac, covariance)
    np.testing.assert_allclose(actual, jac @ covariance @ jac.T)


def test_finite_difference_jacobian():
    jac = finite_difference_jacobian(
        lambda x: np.array([x[0] ** 2 + 3 * x[1], x[0] - x[1]]),
        [2.0, 4.0],
    )
    np.testing.assert_allclose(jac, [[4.0, 3.0], [1.0, -1.0]], rtol=1e-6, atol=1e-7)


def test_reference_ensemble_bootstrap_is_reproducible():
    refs = np.arange(60.0).reshape(10, 6)
    transform = lambda x: np.array([x.mean(), x.std()])
    a = propagate_reference_ensemble(refs, transform, n_draws=200, seed=7)
    b = propagate_reference_ensemble(refs, transform, n_draws=200, seed=7)
    np.testing.assert_array_equal(a["draws"], b["draws"])
    assert a["covariance"].shape == (2, 2)


def test_regularization_sensitivity_is_not_measurement_covariance():
    result = regularization_sensitivity(
        lambda lam: np.array([1.0 / lam, lam]),
        [0.1, 0.2, 0.4],
    )
    assert result["profiles"].shape == (3, 2)
    assert result["interpretation"].startswith("regularization-choice sensitivity")


def test_covariance_rejects_asymmetry_and_negative_variance():
    with pytest.raises(ValueError, match="symmetric"):
        validate_covariance([[1.0, 0.2], [0.0, 1.0]])
    with pytest.raises(ValueError, match="positive semidefinite"):
        validate_covariance([[1.0, 0.0], [0.0, -0.1]])

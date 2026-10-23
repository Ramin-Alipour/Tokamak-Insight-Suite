import numpy as np
import pytest
from tis.uncertainty import UncertaintyError, propagate_covariance, propagate_monte_carlo


def test_linear_covariance_propagation_includes_correlations():
    J = np.array([[2.0, 1.0]])
    C = np.array([[4.0, 1.0], [1.0, 9.0]])
    result = propagate_covariance(J, C)
    assert result["covariance"][0, 0] == pytest.approx(29.0)
    assert result["standard_uncertainty"][0] == pytest.approx(np.sqrt(29.0))


def test_covariance_propagation_rejects_non_positive_semidefinite_matrix():
    with pytest.raises(UncertaintyError, match="positive semidefinite"):
        propagate_covariance([[1.0, 0.0]], [[1.0, 2.0], [2.0, 1.0]])


def test_monte_carlo_recovers_linear_mean_and_uncertainty():
    result = propagate_monte_carlo(
        lambda samples: 3.0 * samples[:, 0] - 2.0 * samples[:, 1],
        mean=[2.0, 1.0], covariance=[[0.25, 0.1], [0.1, 1.0]],
        n_samples=30000, seed=17)
    assert result["mean"] == pytest.approx(4.0, abs=0.04)
    expected_variance = 9 * 0.25 + 4 * 1.0 - 12 * 0.1
    assert result["standard_uncertainty"] == pytest.approx(np.sqrt(expected_variance), rel=0.03)
    assert result["sample_count"] == 30000


def test_monte_carlo_rejects_nonfinite_model_output():
    with pytest.raises(UncertaintyError, match="non-finite"):
        propagate_monte_carlo(lambda x: np.full(x.shape[0], np.nan), [1.0], [[1.0]], n_samples=100)


def test_monte_carlo_rejects_materially_indefinite_small_covariance():
    covariance = np.array([[1e-12, 0.0], [0.0, -1e-11]])
    with pytest.raises(UncertaintyError, match="positive semidefinite"):
        propagate_monte_carlo(lambda samples: samples[:, 0], [0.0, 0.0], covariance, n_samples=100)

import numpy as np
import pytest

from tokamak_analysis.cts_analysis import _combine_independent_parameter_covariances


def test_independent_reference_and_discrepancy_covariances_are_added_once():
    base = np.array([[4.0, 0.5], [0.5, 9.0]])
    reference = np.array([[1.0, 0.2], [0.2, 2.0]])
    discrepancy = np.array([[0.5, -0.1], [-0.1, 0.25]])
    total = _combine_independent_parameter_covariances(
        base, reference, discrepancy
    )
    np.testing.assert_allclose(total, base + reference + discrepancy)
    np.testing.assert_allclose(total, total.T, atol=1e-14)


def test_covariance_aggregation_accepts_missing_independent_components():
    base = np.diag([1.0, 2.0])
    np.testing.assert_allclose(
        _combine_independent_parameter_covariances(base, None, None), base
    )


@pytest.mark.parametrize(
    "component",
    [
        np.array([[1.0, 0.4], [0.0, 1.0]]),
        np.array([[1.0, 0.0], [0.0, -0.1]]),
        np.eye(3),
    ],
)
def test_covariance_aggregation_rejects_invalid_components(component):
    with pytest.raises(ValueError):
        _combine_independent_parameter_covariances(np.eye(2), component)

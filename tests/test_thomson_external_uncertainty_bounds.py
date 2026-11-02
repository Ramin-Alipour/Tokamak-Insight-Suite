import numpy as np
import pytest

from tokamak_analysis.thomson_external_uncertainty import propagate_external_covariance


def test_bound_aware_jacobian_uses_second_order_forward_difference_at_lower_bound():
    result = propagate_external_covariance(
        lambda x: np.array([x[0] ** 2 + 3.0 * x[1]]),
        [0.0, 2.0],
        np.array([[0.04, 0.01], [0.01, 0.09]]),
        input_names=("nonnegative_parameter", "free_parameter"),
        input_bounds=np.array([[0.0, np.inf], [-np.inf, np.inf]]),
    )
    np.testing.assert_allclose(result["jacobian"], [[0.0, 3.0]], atol=1e-7)
    assert result["finite_difference_methods"] == ["second_order_forward", "central"]
    assert result["input_bounds_status"] == "researcher_supplied"


def test_bound_aware_jacobian_uses_second_order_backward_difference_at_upper_bound():
    result = propagate_external_covariance(
        lambda x: np.array([2.0 * x[0] - x[1] ** 2]),
        [1.0, 0.0],
        np.diag([0.04, 0.09]),
        input_bounds=np.array([[-np.inf, 1.0], [-1.0, 1.0]]),
    )
    np.testing.assert_allclose(result["jacobian"], [[2.0, 0.0]], atol=1e-7)
    assert result["finite_difference_methods"] == ["second_order_backward", "central"]


def test_bound_aware_jacobian_rejects_mean_outside_bounds():
    with pytest.raises(ValueError, match="within input_bounds"):
        propagate_external_covariance(
            lambda x: x, [2.0], np.eye(1), input_bounds=np.array([[0.0, 1.0]])
        )

import numpy as np
import pytest

from tis.tomography import regularized_tomography


@pytest.mark.parametrize("order", [1.0, 2.0, True, np.float64(1.0)])
def test_regularized_tomography_rejects_non_integer_regularization_order(order):
    matrix = np.eye(3)
    measurements = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="must be an integer"):
        regularized_tomography(matrix, measurements, regularization_order=order)


@pytest.mark.parametrize("order", [0, 1, 2, np.int64(1)])
def test_regularized_tomography_accepts_integer_regularization_order(order):
    matrix = np.eye(3)
    measurements = np.array([1.0, 2.0, 3.0])
    result = regularized_tomography(matrix, measurements, lam=0.1, regularization_order=order)
    assert result["success"] is True
    assert result["regularization_order"] == int(order)

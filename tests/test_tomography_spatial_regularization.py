"""Regression tests for spatial regularization of flattened 2-D tomography grids."""
import numpy as np
import pytest

from tis.tomography import regularized_tomography, select_tikhonov_lambda


def test_first_order_2d_regularization_uses_only_spatial_neighbors():
    # Row-major 2x2 grid: the first row is zero, the second row is one.
    # There are two vertical differences and no horizontal differences.
    A = np.eye(4)
    y = np.array([0.0, 0.0, 1.0, 1.0])
    result = regularized_tomography(
        A, y, lam=0.0, nonnegative=False, regularization_order=1, grid_shape=(2, 2)
    )
    assert result["regularization_operator_norm"] == pytest.approx(np.sqrt(2.0) / 2.0)


def test_spatial_regularization_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="product"):
        regularized_tomography(
            np.eye(4), np.ones(4), regularization_order=1, grid_shape=(1, 3)
        )


def test_l_curve_selector_accepts_the_same_spatial_grid_contract():
    result = select_tikhonov_lambda(
        np.eye(4), np.array([0.0, 0.0, 1.0, 1.0]),
        lambdas=[0.01, 0.1, 1.0], nonnegative=False, grid_shape=(2, 2),
    )
    assert result["selected_lambda"] in {0.01, 0.1, 1.0}


@pytest.mark.parametrize("bad_lambda", ["0.1", True, np.bool_(False), np.nan, np.inf, -1.0, 10**1000])
def test_tomography_rejects_invalid_regularization_scalar(bad_lambda):
    with pytest.raises(ValueError, match="Regularization"):
        regularized_tomography(np.eye(2), np.ones(2), lam=bad_lambda)


def test_tomography_rejects_non_boolean_nonnegative_flag():
    with pytest.raises(ValueError, match="nonnegative must be a boolean"):
        regularized_tomography(np.eye(2), np.ones(2), nonnegative="false")

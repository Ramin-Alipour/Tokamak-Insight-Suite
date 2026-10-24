import numpy as np
import pytest

from tis.core import linear_calibration, polynomial_calibration


@pytest.mark.parametrize("bad", [True, "2", np.nan, np.inf])
def test_linear_calibration_rejects_invalid_slope_and_intercept(bad):
    with pytest.raises(ValueError, match="slope and intercept"):
        linear_calibration([1.0, 2.0], slope=bad)
    with pytest.raises(ValueError, match="slope and intercept"):
        linear_calibration([1.0, 2.0], slope=2.0, intercept=bad)


def test_polynomial_calibration_accepts_numpy_coefficient_array():
    values = np.array([0.0, 1.0, 2.0])
    result = polynomial_calibration(values, np.array([2.0, 1.0]))
    assert np.allclose(result, [1.0, 3.0, 5.0])


@pytest.mark.parametrize("bad", [[], [1.0, np.nan], [True, 2.0], ["1", "2"], [[1.0, 2.0]]])
def test_polynomial_calibration_rejects_invalid_coefficients(bad):
    with pytest.raises(ValueError, match="polynomial coefficients"):
        polynomial_calibration([0.0, 1.0], bad)

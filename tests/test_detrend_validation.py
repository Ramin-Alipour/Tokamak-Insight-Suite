import numpy as np
import pytest

from tis.signals import detrend_polynomial


def test_detrend_rejects_non_integer_order_with_clear_error():
    with pytest.raises(ValueError, match="order must be an integer"):
        detrend_polynomial([0, 1, 2], [1, 2, 3], order="1")


@pytest.mark.parametrize(
    ("time_s", "values"),
    [
        ([0, 1, np.nan, 3], [1, 2, 3, 4]),
        ([0, 1, 2, 3], [1, np.inf, 3, 4]),
    ],
)
def test_detrend_rejects_non_finite_inputs(time_s, values):
    with pytest.raises(ValueError, match="only finite values"):
        detrend_polynomial(time_s, values, order=1)


@pytest.mark.parametrize("time_s", [[0, 1, 1, 3], [0, 2, 1, 3]])
def test_detrend_rejects_non_increasing_timestamps(time_s):
    with pytest.raises(ValueError, match="strictly increasing"):
        detrend_polynomial(time_s, [1, 2, 3, 4], order=1)


def test_detrend_rejects_non_finite_fit_window():
    with pytest.raises(ValueError, match="two finite time bounds"):
        detrend_polynomial(
            [0, 1, 2, 3], [1, 2, 3, 4], order=1, fit_window_s=(0, np.inf)
        )


def test_detrend_linear_signal_returns_near_zero_residual():
    time_s = np.linspace(0.0, 1.0, 21)
    values = 3.0 * time_s + 2.0
    out = detrend_polynomial(time_s, values, order=1)
    assert np.allclose(out["values"], 0.0, atol=1e-12)
    assert out["order"] == 1

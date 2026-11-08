import numpy as np
import pytest

from tis.signals import detrend_polynomial


def test_detrend_rejects_non_numeric_polynomial_order():
    with pytest.raises(ValueError, match="order must be an integer"):
        detrend_polynomial(np.arange(8.0), np.arange(8.0), order="1")


@pytest.mark.parametrize(
    "time, values",
    [
        ([0, 1, 2, 3, 4, 5, 6, np.nan], list(range(8))),
        (list(range(8)), [0, 1, 2, 3, 4, 5, 6, np.inf]),
    ],
)
def test_detrend_rejects_non_finite_input(time, values):
    with pytest.raises(ValueError, match="only finite"):
        detrend_polynomial(time, values)


@pytest.mark.parametrize("window", [(0.0,), (0.0, np.inf), ("start", 2.0)])
def test_detrend_rejects_invalid_fit_window(window):
    with pytest.raises(ValueError, match="fit_window_s"):
        detrend_polynomial(np.arange(8.0), np.arange(8.0), fit_window_s=window)


def test_detrend_still_fits_linear_trend_in_selected_window():
    time = np.arange(10.0)
    values = 2.0 * time + 5.0
    result = detrend_polynomial(time, values, order=1, fit_window_s=(2.0, 7.0))
    assert np.allclose(result["values"], 0.0, atol=1e-12)
    assert result["fit_window_s"] == [2.0, 7.0]

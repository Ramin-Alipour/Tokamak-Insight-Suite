import numpy as np
import pytest

from tis.probe_analysis import probe_iv_summary


@pytest.mark.parametrize(
    ("voltage", "current"),
    [
        (np.arange(5.0).reshape(5, 1), np.arange(5.0)),
        (np.arange(5.0), np.arange(5.0).reshape(5, 1)),
        ([False, True, False, True, False], np.arange(5.0)),
        (np.arange(5.0), ["0", "1", "2", "3", "4"]),
    ],
)
def test_probe_iv_summary_rejects_non_1d_or_non_numeric_inputs(voltage, current):
    with pytest.raises(ValueError, match="one-dimensional|real numeric"):
        probe_iv_summary(voltage, current)


def test_probe_iv_summary_accepts_one_dimensional_numeric_arrays():
    voltage = np.linspace(-2.0, 2.0, 9)
    current = 1e-3 * voltage
    result = probe_iv_summary(voltage, current)
    assert result["n_points"] == 9
    assert result["floating_potential_interpolated_V"] == pytest.approx(0.0)

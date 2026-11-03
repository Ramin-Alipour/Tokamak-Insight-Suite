import numpy as np
import pytest

from tokamak_analysis.cts_relativistic import _positive


@pytest.mark.parametrize("value", [1, 1.5, np.float64(2.0), np.int64(3)])
def test_positive_accepts_finite_positive_real_scalars(value):
    assert _positive(value, "temperature_eV") == float(value)


@pytest.mark.parametrize(
    "value",
    [True, False, np.bool_(True), "2", "nan", 1 + 0j, np.array([2.0]),
     0, -1, np.inf, -np.inf, np.nan],
)
def test_positive_rejects_nonphysical_or_ambiguous_scalar_inputs(value):
    with pytest.raises(ValueError, match="positive finite real scalar"):
        _positive(value, "temperature_eV")

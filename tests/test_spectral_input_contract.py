"""Input-contract regressions for spectral fitting."""
import numpy as np
import pytest

from tis.spectral_analysis import fit_spectral_line


@pytest.mark.parametrize("bad_limit", ["0.1", True, np.bool_(False), np.nan, np.inf, 10**1000])
def test_spectral_fit_rejects_non_real_or_non_finite_scalar_limits(bad_limit):
    x = np.linspace(-1.0, 1.0, 31)
    y = 1.0 + np.exp(-((x / 0.2) ** 2))
    with pytest.raises(ValueError, match="Fit limits"):
        fit_spectral_line(x, y, xmin=bad_limit)


def test_spectral_fit_rejects_non_string_model_with_value_error():
    x = np.linspace(-1.0, 1.0, 31)
    y = 1.0 + np.exp(-((x / 0.2) ** 2))
    with pytest.raises(ValueError, match="Model must be a string"):
        fit_spectral_line(x, y, model=None)

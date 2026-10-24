"""Numerical stability regressions for inverse-variance Thomson weighting."""
import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import weighted_mean


@pytest.mark.parametrize(
    "uncertainties",
    [
        [1e-300, 2e-300],  # squaring underflows in a direct inverse-variance formula
        [1e300, 2e300],    # squaring overflows in a direct inverse-variance formula
    ],
)
def test_weighted_mean_is_stable_for_extreme_finite_uncertainties(uncertainties):
    result = weighted_mean([1.0, 2.0], uncertainties)
    assert np.isfinite(result)
    assert result == pytest.approx(1.2)


def test_weighted_mean_preserves_standard_inverse_variance_result():
    assert weighted_mean([10.0, 20.0], [1.0, 2.0]) == pytest.approx(12.0)

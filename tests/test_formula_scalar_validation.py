"""Regression tests for invalid scalar inputs to the physics formula library."""
import numpy as np
import pytest

from tokamak_analysis.formula_library import (
    electron_plasma_frequency,
    magnetic_energy_density,
    safety_factor_large_aspect_ratio,
)

_BAD_SCALARS = [True, np.bool_(True), "1.0", np.nan, np.inf, 10**1000]


@pytest.mark.parametrize("bad", _BAD_SCALARS)
@pytest.mark.parametrize(
    "call",
    [
        lambda bad: electron_plasma_frequency(bad),
        lambda bad: magnetic_energy_density(bad),
        lambda bad: safety_factor_large_aspect_ratio(0.2, 1.0, 2.0, bad),
    ],
)
def test_formula_scalar_inputs_reject_invalid_types_and_extremes(call, bad):
    with pytest.raises(ValueError):
        call(bad)

"""Regression tests for bolometry geometry scalar validation."""
import numpy as np
import pytest

from tokamak_analysis.bolometry import radiated_power_from_axisymmetric_emissivity


@pytest.mark.parametrize("bad", [True, np.bool_(True), "1.0", np.nan, np.inf, 10**1000])
def test_axisymmetric_bolometry_rejects_invalid_major_radius(bad):
    with pytest.raises(ValueError, match="Major radius"):
        radiated_power_from_axisymmetric_emissivity([0.0, 0.1], [1.0, 0.5], bad)

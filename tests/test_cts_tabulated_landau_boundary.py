import numpy as np
import pytest

from tokamak_analysis.cts_analysis import _component_tabulated
from tokamak_analysis.cts_analysis import EPS0


def test_tabulated_vdf_landau_susceptibility_includes_zero_extension_endpoint_jumps():
    velocity = np.linspace(-1.0, 1.0,  nine := 9)
    pdf = np.ones(nine)
    # Trapezoidal normalization gives f=1/2 on [-1,1].  The interior
    # derivative is exactly zero; only the zero-extension endpoint jumps
    # contribute when the resonance z=2 lies outside the support.
    susceptibility, resonant_density = _component_tabulated(
        np.array([2.0]), 1.0, 1.0, 1.0, 1.0, 0.0, velocity, pdf
    )
    expected_landau_integral = (
        0.5 / (-1.0 - 2.0) - 0.5 / (1.0 - 2.0)
    )
    expected = -(1.0 / EPS0) * expected_landau_integral
    assert susceptibility[0] == pytest.approx(expected, rel=2e-12)
    assert resonant_density[0] == 0.0

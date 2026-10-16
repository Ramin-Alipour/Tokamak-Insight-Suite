import numpy as np
import pytest

from tis.magnetic import analyze_poloidal_field


def test_poloidal_field_calibration_and_approximate_q():
    result = analyze_poloidal_field(
        [0.1, 0.2], calibration_T_per_unit=1.0,
        toroidal_field_T=[2.0, 2.0], minor_radius_m=[0.2, 0.4],
        major_radius_m=[1.0, 1.0],
    )
    np.testing.assert_allclose(result["poloidal_field_T"], [0.1, 0.2])
    np.testing.assert_allclose(result["approximate_safety_factor_q"], [4.0, 4.0])


def test_poloidal_field_requires_complete_q_inputs():
    with pytest.raises(ValueError, match="must be supplied together"):
        analyze_poloidal_field([0.1], calibration_T_per_unit=1.0, toroidal_field_T=2.0)

import numpy as np
import pytest

from tis.magnetic import analyze_toroidal_field


def test_toroidal_field_calibration_and_radius_scaling():
    result = analyze_toroidal_field(
        [1.0, 2.0], calibration_T_per_unit=0.5, offset_T=0.1,
        signal_uncertainty=0.02, calibration_relative_uncertainty=0.01,
        reference_radius_m=1.5, evaluation_radius_m=[1.5, 3.0],
    )
    np.testing.assert_allclose(result["toroidal_field_T"], [0.6, 1.1])
    np.testing.assert_allclose(result["field_at_evaluation_radius_T"], [[0.6, 0.3], [1.1, 0.55]])
    assert np.all(result["toroidal_field_standard_uncertainty_T"] > 0)


def test_toroidal_field_rejects_invalid_calibration():
    with pytest.raises(ValueError, match="nonzero"):
        analyze_toroidal_field([1, 2], calibration_T_per_unit=0)

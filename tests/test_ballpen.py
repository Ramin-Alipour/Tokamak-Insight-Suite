import numpy as np
import pytest
from tis.ballpen import analyze_ball_pen_probe


def test_ballpen_reports_proxy_and_optional_calibrated_potential():
    result = analyze_ball_pen_probe([10.0, 11.0, 9.0], electron_temperature_eV=5.0, correction_factor=0.6, time_s=[0, 1, 2])
    np.testing.assert_allclose(result['plasma_potential_estimate_V'], [13.0, 14.0, 12.0])
    assert result['mean_voltage_V'] == pytest.approx(10.0)


def test_ballpen_does_not_silently_apply_partial_correction():
    with pytest.raises(ValueError, match='supplied together'):
        analyze_ball_pen_probe([1, 2, 3], electron_temperature_eV=2.0)

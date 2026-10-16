import numpy as np
import pytest
from tis.hard_xray import analyze_hard_xray_spectrum


def test_hard_xray_energy_calibration_and_dead_time_correction():
    result = analyze_hard_xray_spectrum([0, 1, 2], [10, 20, 30], energy_calibration_coefficients=[2.0, 1.0], live_time_s=10, dead_time_s=0.001)
    np.testing.assert_allclose(result['photon_energy_keV'], [1, 3, 5])
    assert result['corrected_total_count_rate_s-1'] > result['observed_total_count_rate_s-1']


def test_hard_xray_rejects_invalid_dead_time_regime():
    with pytest.raises(ValueError, match='unity'):
        analyze_hard_xray_spectrum([0, 1], [1000, 1], energy_calibration_coefficients=[1, 1], live_time_s=1, dead_time_s=0.01)

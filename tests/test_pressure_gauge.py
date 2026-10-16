import numpy as np
from tis.pressure_gauge import analyze_pressure_gauge


def test_pressure_gauge_unit_conversion_and_range():
    r = analyze_pressure_gauge([1, 2], calibration_Pa_per_unit=100, valid_range_Pa=(50, 150), reading_uncertainty=0.1)
    np.testing.assert_allclose(r['pressure_Pa'], [100, 200])
    assert r['within_valid_range'].tolist() == [True, False]

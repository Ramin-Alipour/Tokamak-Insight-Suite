import numpy as np
from tis.strike_point import analyze_strike_point_tracking


def test_peak_location_tracking():
    maps = np.zeros((3, 4, 4))
    maps[0, 1, 1] = 10
    maps[1, 1, 2] = 10
    maps[2, 1, 3] = 10
    R, Z = np.meshgrid(np.arange(4)*0.01, np.arange(4)*0.02)
    r = analyze_strike_point_tracking(maps, [0, 1, 2], R, Z)
    np.testing.assert_allclose(r['peak_R_m'], [0.01, 0.02, 0.03])

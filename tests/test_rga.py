import numpy as np
from tis.rga import analyze_rga_spectrum


def test_rga_partial_pressure_conversion():
    r = analyze_rga_spectrum([1, 2, 3], [1e-6, 2e-6, 1e-6], sensitivity_Pa_per_A=1e4, background_current_A=1e-7)
    assert r['total_pressure_proxy_Pa'] > 0
    assert abs(np.sum(r['partial_pressure_fraction'])-1) < 1e-12

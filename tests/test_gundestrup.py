import numpy as np
from tis.gundestrup import analyze_gundestrup_probe


def test_gundestrup_harmonic_fit():
    angle = np.linspace(0, 2*np.pi, 12, endpoint=False)
    current = 2.0 + 0.4*np.cos(angle - 0.7)
    result = analyze_gundestrup_probe(angle, current, modulation_per_mach=0.2)
    assert abs(result['flow_direction_rad'] - 0.7) < 1e-10
    assert abs(result['mach_number_estimate'] - 1.0) < 1e-10
    assert result['r_squared'] > 0.999999

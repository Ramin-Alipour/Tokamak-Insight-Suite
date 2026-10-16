import numpy as np
from tis.pumping_speed import analyze_effective_pumping_speed


def test_effective_pumping_speed_and_conductance():
    r = analyze_effective_pumping_speed(1.0, 0.1, pump_speed_m3_s=2.0, conductance_m3_s=2.0)
    assert r['throughput_over_pressure_speed_m3_s'] == 10.0
    assert r['effective_speed_with_series_conductance_m3_s'] == 1.0

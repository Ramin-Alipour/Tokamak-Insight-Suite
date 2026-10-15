import pytest
from tis.lower_hybrid import analyze_lower_hybrid_power


def test_lower_hybrid_power_and_current_drive_efficiency():
    r = analyze_lower_hybrid_power(1000, frequency_hz=3.7e9, transmission_fraction=0.9, reflection_fraction=0.1, absorption_fraction=0.8, driven_current_A=100)
    assert r['total_absorbed_power_W'] == pytest.approx(648.0)
    assert r['current_drive_efficiency_A_W'] == pytest.approx(100/648)

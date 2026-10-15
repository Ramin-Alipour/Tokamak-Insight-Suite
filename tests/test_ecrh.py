import pytest
from tis.ecrh import analyze_ecrh_power


def test_ecrh_absorbed_power_and_resonance():
    r = analyze_ecrh_power(1000, transmission_fraction=0.9, reflection_fraction=0.1, absorption_fraction=0.8, frequency_hz=60e9)
    assert r['total_absorbed_power_W'] == pytest.approx(648.0)
    assert r['resonance']['resonance_field_T'] > 0

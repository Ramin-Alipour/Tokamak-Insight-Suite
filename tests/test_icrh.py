import pytest
from tis.icrh import analyze_icrh_power


def test_icrh_power_and_resonance_frequency():
    r = analyze_icrh_power(1000, magnetic_field_T=2, ion_mass_amu=2, transmission_fraction=0.9, reflection_fraction=0.1, absorption_fraction=0.8)
    assert r['total_absorbed_power_W'] == pytest.approx(648.0)
    assert r['ion_cyclotron_resonance_frequency_Hz'] > 0

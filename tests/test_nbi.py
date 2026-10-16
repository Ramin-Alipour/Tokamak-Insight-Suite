import pytest
from tis.nbi import analyze_nbi_power


def test_nbi_power_from_beam_energy_and_rate():
    r = analyze_nbi_power(50, 1e20, neutralization_fraction=0.8, coupling_fraction=0.5)
    assert r['total_injected_power_W'] == pytest.approx(50e3*1.602176634e-19*1e20*0.4)

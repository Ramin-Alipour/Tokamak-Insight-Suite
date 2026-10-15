import numpy as np
from tis.npa import analyze_neutral_particle_analyzer


def test_npa_efficiency_corrected_energy_moments():
    r = analyze_neutral_particle_analyzer([10, 20, 30], [10, 20, 10], live_time_s=2, efficiency=[0.5, 1, 0.5], solid_angle_sr=0.1)
    assert r['total_efficiency_corrected_rate_s-1'] > 0
    assert r['mean_energy_keV'] > 0
    assert np.all(r['efficiency_corrected_rate_s-1'] > 0)

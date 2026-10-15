import numpy as np
from tis.reflectometry import analyze_reflectometry_sweep


def test_reflectometry_sweep_cutoff_density_increases_with_frequency():
    f = np.linspace(20e9, 60e9, 21)
    phase = 0.02 * (f - f[0]) / (f[-1] - f[0])
    result = analyze_reflectometry_sweep(f, phase)
    assert np.all(np.diff(result['O_mode_cutoff_density_m-3']) > 0)
    assert result['phase_group_delay_proxy_s'].shape == f.shape

import numpy as np
import pytest
from tis.probe_fluctuations import analyze_probe_density_fluctuations


def test_probe_density_fluctuation_temperature_correction():
    current = np.array([9.0, 10.0, 11.0, 10.0])
    te = np.array([9.0, 10.0, 11.0, 10.0])
    result = analyze_probe_density_fluctuations(current, electron_temperature_eV=te, sampling_frequency_Hz=1000)
    assert result['relative_density_fluctuation_proxy'].shape == current.shape
    assert result['relative_density_rms'] >= 0
    assert result['frequency_Hz'][0] == 0


def test_probe_density_fluctuations_reject_zero_mean_current():
    with pytest.raises(ValueError, match='zero'):
        analyze_probe_density_fluctuations([-1, 1, -1, 1])

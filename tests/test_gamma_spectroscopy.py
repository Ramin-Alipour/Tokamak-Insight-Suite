import numpy as np
from tis.gamma_spectroscopy import analyze_gamma_spectrum


def test_gamma_spectrum_calibrates_energy_and_finds_peak():
    channel = np.arange(100)
    counts = np.ones(100)
    counts[40] = 100
    result = analyze_gamma_spectrum(channel, counts, energy_calibration_coefficients=[2.0, 1.0], live_time_s=10, minimum_prominence=10)
    assert result['candidate_peaks'][0]['energy_keV'] == 81

import numpy as np
from tis.noise_floor import analyze_noise_floor_detection_limit


def test_noise_floor_detection_threshold():
    x = np.array([-1., 0., 1., -0.5, 0.5, 0.2, -0.2])
    r = analyze_noise_floor_detection_limit(x, false_alarm_probability=0.01, independent_tests=2, sampling_frequency_Hz=1000)
    assert r['minimum_detectable_excursion'] > 0
    assert r['noise_psd_signal2_per_Hz'].size > 0

import numpy as np
from scipy.stats import norm
from scipy.signal import welch


def analyze_noise_floor_detection_limit(background_signal, *, false_alarm_probability: float = 0.001, independent_tests: int = 1, sampling_frequency_Hz: float | None = None) -> dict:
    """Estimate robust noise floor and Gaussian false-alarm detection threshold."""
    x = np.asarray(background_signal, dtype=float).ravel()
    if x.size < 3 or not np.all(np.isfinite(x)):
        raise ValueError('Background signal must contain at least three finite samples.')
    if not np.isfinite(false_alarm_probability) or not 0 < false_alarm_probability < 0.5:
        raise ValueError('false_alarm_probability must lie between zero and 0.5.')
    if isinstance(independent_tests, bool) or not isinstance(independent_tests, int) or independent_tests < 1:
        raise ValueError('independent_tests must be a positive integer.')
    baseline = float(np.median(x))
    sigma = float(1.4826*np.median(np.abs(x-baseline)))
    if sigma <= np.finfo(float).eps:
        sigma = float(np.std(x, ddof=1))
    per_test_alpha = false_alarm_probability/independent_tests
    z = float(norm.isf(per_test_alpha))
    result = {'diagnostic': 'Noise floor and detection limit', 'baseline_level': baseline,
        'robust_noise_sigma': sigma, 'noise_floor_rms_about_baseline': float(np.sqrt(np.mean((x-baseline)**2))),
        'false_alarm_probability_familywise': float(false_alarm_probability),
        'independent_tests': independent_tests, 'threshold_sigma': z,
        'minimum_detectable_excursion': z*sigma, 'detection_threshold': baseline+z*sigma,
        'assumptions_and_limits': ['Threshold assumes approximately Gaussian stationary background after robust centering.',
            'Correlated samples reduce the effective number of independent tests.',
            'Detection threshold depends on bandwidth, windowing, integration and false-alarm convention.']}
    if sampling_frequency_Hz is not None:
        if not np.isfinite(sampling_frequency_Hz) or sampling_frequency_Hz <= 0:
            raise ValueError('sampling_frequency_Hz must be positive and finite.')
        f, psd = welch(x-baseline, fs=sampling_frequency_Hz, nperseg=min(1024, x.size))
        result['frequency_Hz'] = f
        result['noise_psd_signal2_per_Hz'] = psd
    return result

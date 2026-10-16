from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from .core import polynomial_calibration


def analyze_gamma_spectrum(channel_index, counts, *, energy_calibration_coefficients, live_time_s: float, minimum_prominence: float = 5.0) -> dict:
    """Calibrate a gamma pulse-height spectrum and identify candidate peaks."""
    channel = np.asarray(channel_index, dtype=float).ravel()
    y = np.asarray(counts, dtype=float).ravel()
    if channel.size != y.size or y.size < 5 or not np.all(np.isfinite(channel)) or not np.all(np.isfinite(y)):
        raise ValueError('Channel and count arrays must be finite, equal length and contain at least five bins.')
    if np.any(np.diff(channel) <= 0) or np.any(y < 0):
        raise ValueError('Channels must increase strictly and counts must be nonnegative.')
    if not np.isfinite(live_time_s) or live_time_s <= 0 or not np.isfinite(minimum_prominence) or minimum_prominence <= 0:
        raise ValueError('Live time and minimum prominence must be positive and finite.')
    energy = polynomial_calibration(channel, list(energy_calibration_coefficients))
    if np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0):
        raise ValueError('Energy calibration must yield a finite, strictly increasing energy axis.')
    peaks, properties = find_peaks(y, prominence=minimum_prominence)
    peak_list = [{'channel': int(channel[i]), 'energy_keV': float(energy[i]), 'counts': float(y[i]),
        'prominence_counts': float(properties['prominences'][j]), 'count_rate_s-1': float(y[i]/live_time_s)} for j, i in enumerate(peaks)]
    return {'diagnostic': 'Gamma spectroscopy', 'energy_axis_keV': energy,
        'count_rate_s-1': y/live_time_s, 'candidate_peaks': peak_list,
        'integrated_count_rate_s-1': float(np.sum(y)/live_time_s), 'live_time_s': float(live_time_s),
        'assumptions_and_limits': ['Energy calibration must be derived from known gamma reference lines.',
            'Peak detection reports candidates only; background subtraction, peak fitting and isotope identification are separate.',
            'Absolute activity requires efficiency, branching ratios, geometry and attenuation corrections.']}

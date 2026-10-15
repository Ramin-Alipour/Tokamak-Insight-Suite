from __future__ import annotations
import numpy as np
from .core import polynomial_calibration


def analyze_hard_xray_spectrum(channel_index, counts, *, energy_calibration_coefficients, live_time_s: float, dead_time_s: float = 0.0) -> dict:
    """Calibrate pulse-height channels to photon energy and correct count rates for nonparalyzable dead time."""
    channel = np.asarray(channel_index, dtype=float).ravel()
    counts = np.asarray(counts, dtype=float).ravel()
    if channel.size != counts.size or channel.size < 2 or not np.all(np.isfinite(channel)) or not np.all(np.isfinite(counts)):
        raise ValueError('Channel and count arrays must be finite, equal length and contain at least two bins.')
    if np.any(np.diff(channel) <= 0) or np.any(counts < 0):
        raise ValueError('Channels must increase strictly and counts must be nonnegative.')
    if not np.isfinite(live_time_s) or live_time_s <= 0 or not np.isfinite(dead_time_s) or dead_time_s < 0:
        raise ValueError('live_time_s must be positive; dead_time_s must be finite and nonnegative.')
    energy = polynomial_calibration(channel, list(energy_calibration_coefficients))
    if np.any(~np.isfinite(energy)) or np.any(np.diff(energy) <= 0):
        raise ValueError('Energy calibration must produce a finite, strictly increasing energy axis.')
    observed_rate = counts / live_time_s
    occupancy = observed_rate * dead_time_s
    if np.any(occupancy >= 1):
        raise ValueError('Observed rate times dead time reaches unity; nonparalyzable correction is invalid.')
    corrected_rate = observed_rate / (1 - occupancy)
    return {'diagnostic': 'Hard X-ray detector', 'channel_index': channel, 'photon_energy_keV': energy,
        'counts_per_bin': counts, 'observed_count_rate_s-1': observed_rate,
        'dead_time_corrected_count_rate_s-1': corrected_rate,
        'observed_total_count_rate_s-1': float(np.sum(observed_rate)),
        'corrected_total_count_rate_s-1': float(np.sum(corrected_rate)),
        'peak_energy_keV': float(energy[int(np.argmax(corrected_rate))]),
        'live_time_s': float(live_time_s), 'dead_time_s': float(dead_time_s),
        'assumptions_and_limits': ['Polynomial calibration coefficients must be derived from reference sources for this detector setup.',
            'Nonparalyzable dead-time model is assumed; pile-up, escape peaks, efficiency and background are not corrected.',
            'Photon flux requires energy-dependent detector efficiency and solid-angle calibration.']}

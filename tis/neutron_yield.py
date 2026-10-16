from __future__ import annotations
import numpy as np


def analyze_neutron_yield_monitor(counts, *, bin_width_s: float, dead_time_s: float = 0.0, neutrons_per_count: float | None = None) -> dict:
    """Compute neutron monitor rates with nonparalyzable dead-time correction and optional yield calibration."""
    c = np.asarray(counts, dtype=float).ravel()
    if c.size == 0 or not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError('counts must be finite, nonnegative and nonempty.')
    if not np.isfinite(bin_width_s) or bin_width_s <= 0 or not np.isfinite(dead_time_s) or dead_time_s < 0:
        raise ValueError('bin_width_s must be positive and dead_time_s nonnegative and finite.')
    if neutrons_per_count is not None and (not np.isfinite(neutrons_per_count) or neutrons_per_count <= 0):
        raise ValueError('neutrons_per_count must be positive and finite.')
    observed = c / bin_width_s
    occupancy = observed * dead_time_s
    if np.any(occupancy >= 1):
        raise ValueError('Count rate times dead time reaches unity; correction is invalid.')
    corrected = observed / (1-occupancy)
    duration = c.size * bin_width_s
    result = {'diagnostic': 'Neutron yield monitor', 'observed_count_rate_s-1': observed,
        'dead_time_corrected_count_rate_s-1': corrected, 'total_observed_counts': float(np.sum(c)),
        'integrated_corrected_counts': float(np.sum(corrected)*bin_width_s),
        'duration_s': float(duration), 'dead_time_s': float(dead_time_s),
        'neutron_yield': float(np.sum(corrected)*bin_width_s*neutrons_per_count) if neutrons_per_count is not None else None,
        'neutrons_per_count_calibration': neutrons_per_count,
        'assumptions_and_limits': ['Nonparalyzable dead-time model is assumed.',
            'Absolute neutron yield requires detector efficiency, solid angle, energy response and geometry calibration.',
            'Background, pulse pile-up and saturation beyond the simple dead-time model are not corrected.']}
    return result

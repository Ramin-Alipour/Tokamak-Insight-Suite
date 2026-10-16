from __future__ import annotations
import numpy as np


def analyze_neutral_particle_analyzer(energy_keV, counts, *, live_time_s: float, efficiency, solid_angle_sr: float | None = None) -> dict:
    """Correct neutral-particle analyzer spectra for supplied efficiency and estimate energy moments."""
    energy = np.asarray(energy_keV, dtype=float).ravel()
    c = np.asarray(counts, dtype=float).ravel()
    eff = np.asarray(efficiency, dtype=float)
    if energy.size < 2 or energy.size != c.size or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(c)):
        raise ValueError('Energy and counts must be finite, equal length and contain at least two bins.')
    if np.any(np.diff(energy) <= 0) or np.any(energy < 0) or np.any(c < 0):
        raise ValueError('Energy bins must increase and energy/counts must be nonnegative.')
    if eff.ndim == 0:
        eff = np.full(c.size, float(eff))
    else:
        eff = eff.ravel()
    if eff.size != c.size or not np.all(np.isfinite(eff)) or np.any(eff <= 0) or np.any(eff > 1):
        raise ValueError('Efficiency must be in (0,1] and scalar or match energy bins.')
    if not np.isfinite(live_time_s) or live_time_s <= 0:
        raise ValueError('live_time_s must be positive and finite.')
    if solid_angle_sr is not None and (not np.isfinite(solid_angle_sr) or solid_angle_sr <= 0):
        raise ValueError('solid_angle_sr must be positive and finite.')
    rate = c / (live_time_s * eff)
    total = float(np.sum(rate))
    mean_energy = float(np.sum(energy*rate)/total) if total > 0 else None
    variance = float(np.sum((energy-mean_energy)**2*rate)/total) if total > 0 else None
    return {'diagnostic': 'Neutral particle analyzer (NPA)', 'energy_keV': energy,
        'efficiency_corrected_rate_s-1': rate, 'total_efficiency_corrected_rate_s-1': total,
        'mean_energy_keV': mean_energy, 'energy_standard_deviation_keV': float(np.sqrt(variance)) if variance is not None else None,
        'angular_rate_proxy_s-1_sr-1': rate/solid_angle_sr if solid_angle_sr is not None else None,
        'assumptions_and_limits': ['Efficiency must include detector and analyzer transmission for the relevant species and energy.',
            'Absolute neutral flux requires calibrated aperture, solid angle, energy bin widths and geometry.',
            'Background, charge exchange along the line of sight and energy resolution are not modeled.']}

import numpy as np
from .core import regularized_tomography


def analyze_neutron_spectrometry(channel_counts, response_matrix, neutron_energy_keV, *, live_time_s: float, regularization=1e-2, count_uncertainty=None) -> dict:
    """Unfold a neutron energy spectrum with a supplied detector response matrix."""
    counts = np.asarray(channel_counts, dtype=float).ravel()
    response = np.asarray(response_matrix, dtype=float)
    energy = np.asarray(neutron_energy_keV, dtype=float).ravel()
    if response.ndim != 2 or response.shape != (counts.size, energy.size) or counts.size < 3 or energy.size < 3:
        raise ValueError('Response matrix shape is incompatible with channel counts and energy bins.')
    if not all(np.all(np.isfinite(a)) for a in (counts, response, energy)) or np.any(counts < 0) or np.any(response < 0) or np.any(np.diff(energy) <= 0):
        raise ValueError('Inputs must be finite and nonnegative, with strictly increasing energy bins.')
    if not np.isfinite(live_time_s) or live_time_s <= 0:
        raise ValueError('live_time_s must be positive and finite.')
    sigma = None if count_uncertainty is None else np.asarray(count_uncertainty, dtype=float).ravel()
    if sigma is not None and (sigma.size != counts.size or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
        raise ValueError('count_uncertainty must be positive, finite and match channel counts.')
    fit = regularized_tomography(response, counts, lam=regularization, nonnegative=True, measurement_sigma=sigma, regularization_order=2)
    spectrum = np.asarray(fit['solution'], dtype=float)
    total = float(np.sum(spectrum))
    mean_energy = float(np.sum(energy*spectrum)/total) if total > 0 else None
    return {'diagnostic': 'Neutron spectrometry', 'neutron_energy_keV': energy,
        'unfolded_spectrum_counts_proxy': spectrum, 'mean_neutron_energy_keV': mean_energy,
        'integrated_spectrum_counts_proxy': total, 'rate_proxy_s-1': total/live_time_s,
        'residual_norm': fit['residual_norm'], 'regularization': float(regularization),
        'assumptions_and_limits': ['Response matrix must represent detector, geometry and neutron interaction physics.',
            'Unfolding is ill-conditioned; regularization affects spectral resolution and uncertainty.',
            'Absolute neutron spectrum requires efficiency, geometry and energy-dependent calibration.']}

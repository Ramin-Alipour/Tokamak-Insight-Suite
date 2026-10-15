from __future__ import annotations
import numpy as np
from .core import regularized_tomography


def analyze_fast_bolometry(time_s, channel_signals, response_matrix, voxel_volume_m3, *, regularization=1e-2, measurement_sigma=None) -> dict:
    """Perform time-resolved nonnegative bolometric tomography and power integration."""
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(channel_signals, dtype=float)
    A = np.asarray(response_matrix, dtype=float)
    volume = np.asarray(voxel_volume_m3, dtype=float).ravel()
    if y.ndim != 2 or y.shape[0] != t.size or t.size < 2 or y.shape[1] != A.shape[0]:
        raise ValueError('channel_signals must be time-by-channel and match time_s and response_matrix.')
    if A.ndim != 2 or A.shape[1] != volume.size or volume.size < 1:
        raise ValueError('response_matrix and voxel volumes have incompatible dimensions.')
    if not all(np.all(np.isfinite(a)) for a in (t, y, A, volume)) or np.any(np.diff(t) <= 0) or np.any(volume <= 0):
        raise ValueError('Inputs must be finite, time strictly increasing and voxel volumes positive.')
    sigma = None if measurement_sigma is None else np.asarray(measurement_sigma, dtype=float)
    if sigma is not None and (sigma.shape not in {(y.shape[1],), y.shape} or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
        raise ValueError('measurement_sigma must be positive, finite and per-channel or time-by-channel.')
    emissivity = []
    residual_rms = []
    for idx, row in enumerate(y):
        row_sigma = None if sigma is None else (sigma if sigma.ndim == 1 else sigma[idx])
        fit = regularized_tomography(A, row, lam=regularization, nonnegative=True, measurement_sigma=row_sigma, regularization_order=2)
        emissivity.append(np.asarray(fit['reconstruction'], dtype=float))
        residual_rms.append(float(fit.get('residual_rms', np.sqrt(np.mean((A @ np.asarray(fit['reconstruction']) - row)**2)))))
    eps = np.asarray(emissivity)
    power = eps @ volume
    return {'diagnostic': 'Fast bolometry', 'time_s': t, 'emissivity_reconstruction': eps,
        'integrated_radiated_power_proxy_W': power, 'relative_power_fluctuation': (power-np.mean(power))/np.mean(power) if abs(np.mean(power)) > np.finfo(float).eps else None,
        'per_time_residual_rms': np.asarray(residual_rms), 'regularization': float(regularization),
        'assumptions_and_limits': ['Response matrix and voxel volumes must be calibrated and geometrically consistent.',
            'Power units depend on the calibration/normalization of the response matrix and emissivity.',
            'Tomographic regularization suppresses poorly constrained spatial structure; temporal regularization is not applied.']}

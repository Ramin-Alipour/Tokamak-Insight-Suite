from __future__ import annotations
import numpy as np
from .core import regularized_tomography


def analyze_soft_xray_array(time_s, channel_signals, response_matrix, *, regularization=1e-2, measurement_sigma=None) -> dict:
    """Reconstruct time-resolved soft-X-ray emissivity from calibrated chord data."""
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(channel_signals, dtype=float)
    A = np.asarray(response_matrix, dtype=float)
    if y.ndim != 2 or y.shape[0] != t.size or t.size < 2 or A.ndim != 2 or y.shape[1] != A.shape[0]:
        raise ValueError('Signals must be time-by-channel and match time_s and response_matrix.')
    if A.shape[1] < 3 or not all(np.all(np.isfinite(a)) for a in (t, y, A)) or np.any(np.diff(t) <= 0):
        raise ValueError('Inputs must be finite; time strictly increasing and at least three reconstruction cells required.')
    sigma = None if measurement_sigma is None else np.asarray(measurement_sigma, dtype=float)
    if sigma is not None and (sigma.shape not in {(y.shape[1],), y.shape} or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0)):
        raise ValueError('measurement_sigma must be positive, finite and per-channel or time-by-channel.')
    recon = []
    residual = []
    for i, row in enumerate(y):
        row_sigma = None if sigma is None else (sigma if sigma.ndim == 1 else sigma[i])
        fit = regularized_tomography(A, row, lam=regularization, nonnegative=True, measurement_sigma=row_sigma, regularization_order=2)
        solution = np.asarray(fit['solution'], dtype=float)
        recon.append(solution)
        residual.append(float(np.sqrt(np.mean((A @ solution - row)**2)))
    eps = np.asarray(recon)
    return {'diagnostic': 'Soft X-ray arrays', 'time_s': t, 'emissivity_reconstruction': eps,
        'peak_emissivity_by_time': np.max(eps, axis=1), 'peak_cell_by_time': np.argmax(eps, axis=1),
        'residual_rms_by_time': np.asarray(residual), 'regularization': float(regularization),
        'assumptions_and_limits': ['Response matrix must encode chord geometry and detector calibration.',
            'Tomographic inversion is ill-posed; regularization and cell geometry affect reconstructed emissivity.',
            'Emissivity units depend on the calibrated measurement and response-matrix conventions.']}

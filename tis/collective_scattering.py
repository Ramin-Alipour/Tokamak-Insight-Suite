from __future__ import annotations
import numpy as np
from scipy.optimize import nnls


def fit_collective_scattering_spectrum(frequency_shift_Hz, measured_spectrum, model_templates, component_names, *, uncertainty=None) -> dict:
    """Fit a measured collective-scattering spectrum to supplied physical templates.

    Templates must be generated from the actual scattering geometry and a suitable
    plasma response/fast-ion model on the same frequency-shift grid. The fit only
    estimates nonnegative template amplitudes; it does not create the physical model.
    """
    shift = np.asarray(frequency_shift_Hz, dtype=float).ravel()
    y = np.asarray(measured_spectrum, dtype=float).ravel()
    templates = np.asarray(model_templates, dtype=float)
    names = list(component_names)
    if shift.size != y.size or shift.size < 3 or templates.ndim != 2 or templates.shape[0] != y.size or templates.shape[1] != len(names) or not names:
        raise ValueError('Shift, spectrum, template matrix and component names have incompatible shapes.')
    if not all(np.all(np.isfinite(a)) for a in (shift, y, templates)) or np.any(np.diff(shift) <= 0):
        raise ValueError('Inputs must be finite and frequency_shift_Hz strictly increasing.')
    if np.any(y < 0) or np.any(templates < 0):
        raise ValueError('Spectrum and model templates must be nonnegative after background subtraction.')
    if uncertainty is None:
        sigma = np.sqrt(np.maximum(y, max(float(np.max(y))*1e-6, np.finfo(float).eps)))
    else:
        sigma = np.asarray(uncertainty, dtype=float).ravel()
        if sigma.size != y.size or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError('uncertainty must be positive, finite and match the spectrum.')
    coeff, residual_norm = nnls(templates / sigma[:, None], y / sigma)
    prediction = templates @ coeff
    residual = y - prediction
    total = float(np.sum((y / sigma)**2))
    return {'diagnostic': 'Collective Thomson / coherent scattering (CTS)',
        'component_names': names, 'nonnegative_amplitudes': coeff,
        'normalized_component_amplitudes': coeff/np.sum(coeff) if np.sum(coeff) > 0 else np.full_like(coeff, np.nan),
        'predicted_spectrum': prediction, 'residuals': residual,
        'weighted_residual_norm': float(residual_norm),
        'weighted_residual_fraction': float(residual_norm/np.sqrt(total)) if total > 0 else None,
        'assumptions_and_limits': [
            'Templates must include scattering geometry, instrument response and the selected plasma response model.',
            'Amplitude fractions are not automatically physical densities without calibrated template normalization.',
            'Template degeneracy, background errors and model uncertainty can dominate inferred components.'
        ]}

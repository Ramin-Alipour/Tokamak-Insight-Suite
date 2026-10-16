from __future__ import annotations
import numpy as np
from .core import fit_spectral_line


def characterize_instrument_line_shape(wavelength_nm, reference_intensity, *, known_wavelength_nm: float, fit_min_nm=None, fit_max_nm=None, model='Gaussian', intrinsic_fwhm_nm: float = 0.0) -> dict:
    """Characterize an instrument line profile from a narrow reference feature."""
    if not np.isfinite(known_wavelength_nm) or known_wavelength_nm <= 0:
        raise ValueError('known_wavelength_nm must be positive and finite.')
    if not np.isfinite(intrinsic_fwhm_nm) or intrinsic_fwhm_nm < 0:
        raise ValueError('intrinsic_fwhm_nm must be finite and nonnegative.')
    fit = fit_spectral_line(wavelength_nm, reference_intensity, xmin=fit_min_nm, xmax=fit_max_nm, model=model)
    observed = float(fit['FWHM'])
    if model.lower() == 'gaussian':
        if observed <= intrinsic_fwhm_nm:
            raise ValueError('Observed reference width does not exceed intrinsic width; instrument width is unresolved.')
        instrument_width = float(np.sqrt(observed**2 - intrinsic_fwhm_nm**2))
    else:
        instrument_width = None if intrinsic_fwhm_nm > 0 else observed
    return {'diagnostic': 'Instrumental line-shape characterization', 'line_fit': fit,
        'known_wavelength_nm': float(known_wavelength_nm), 'centroid_error_nm': float(fit['center']-known_wavelength_nm),
        'observed_fwhm_nm': observed, 'intrinsic_fwhm_nm': float(intrinsic_fwhm_nm),
        'instrumental_fwhm_nm': instrument_width,
        'resolving_power_lambda_over_delta_lambda': float(known_wavelength_nm/instrument_width) if instrument_width and instrument_width > 0 else None,
        'assumptions_and_limits': [
            'Reference feature should be much narrower than the instrument response or have a known intrinsic width.',
            'Quadrature deconvolution is used only for Gaussian profiles.',
            'Resolution may vary across wavelength, slit width, grating setting and detector position.'
        ]}

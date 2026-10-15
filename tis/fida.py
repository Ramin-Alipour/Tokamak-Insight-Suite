from __future__ import annotations
import numpy as np
from .core import fit_spectral_line, doppler_velocity_m_s


def analyze_fida_spectrum(wavelength_nm, intensity, *, fast_ion_region_nm, reference_wavelength_nm: float, background_region_nm=None, detector_response_factor: float | None = None) -> dict:
    """Measure a FIDA spectral feature and its Doppler-shifted LOS velocity proxy."""
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if x.size != y.size or x.size < 7 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0):
        raise ValueError('Spectrum must be finite with matching arrays and strictly increasing wavelengths.')
    if len(fast_ion_region_nm) != 2 or not np.all(np.isfinite(fast_ion_region_nm)) or fast_ion_region_nm[0] >= fast_ion_region_nm[1]:
        raise ValueError('fast_ion_region_nm must be a finite increasing pair.')
    if not np.isfinite(reference_wavelength_nm) or reference_wavelength_nm <= 0:
        raise ValueError('reference_wavelength_nm must be positive and finite.')
    if detector_response_factor is not None and (not np.isfinite(detector_response_factor) or detector_response_factor <= 0):
        raise ValueError('detector_response_factor must be positive and finite.')
    fit = fit_spectral_line(x, y, xmin=float(fast_ion_region_nm[0]), xmax=float(fast_ion_region_nm[1]), model='Gaussian')
    background = None
    if background_region_nm is not None:
        if len(background_region_nm) != 2 or not np.all(np.isfinite(background_region_nm)) or background_region_nm[0] >= background_region_nm[1]:
            raise ValueError('background_region_nm must be a finite increasing pair.')
        m = (x >= background_region_nm[0]) & (x <= background_region_nm[1])
        if m.sum() < 3:
            raise ValueError('Background region must include at least three wavelength bins.')
        background = float(np.median(y[m]))
    area = float(fit['area_above_baseline'])
    corrected_area = area / detector_response_factor if detector_response_factor is not None else None
    return {'diagnostic': 'Fast-ion D-alpha (FIDA)', 'feature_fit': fit,
        'reference_wavelength_nm': float(reference_wavelength_nm),
        'line_of_sight_doppler_velocity_m_s': float(doppler_velocity_m_s(float(fit['center']), reference_wavelength_nm)),
        'integrated_feature_signal': area, 'detector_response_corrected_signal': corrected_area,
        'background_region_median': background,
        'assumptions_and_limits': ['FIDA fast-ion interpretation requires separation from cold D-alpha and other emission components.',
            'Doppler velocity is a projected spectral shift, not a unique fast-ion energy or distribution.',
            'Absolute fast-ion density requires neutral-beam, atomic and instrument-response modeling.']}

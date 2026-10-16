from __future__ import annotations
import numpy as np
from .cxrs import analyze_cxrs_line


def analyze_charge_exchange_spectroscopy(wavelength_nm, intensity, *, rest_wavelength_nm: float,
        ion_mass_amu: float, instrumental_fwhm_nm: float = 0.0, fit_min_nm=None, fit_max_nm=None,
        detector_response_factor: float | None = None, neutral_beam_overlap_factor: float | None = None,
        fit_model: str = 'Gaussian') -> dict:
    """Analyze a charge-exchange emission line with optional response/beam-overlap corrections."""
    result = analyze_cxrs_line(wavelength_nm, intensity, rest_wavelength_nm=rest_wavelength_nm,
        ion_mass_amu=ion_mass_amu, instrumental_fwhm_nm=instrumental_fwhm_nm,
        fit_min_nm=fit_min_nm, fit_max_nm=fit_max_nm, fit_model=fit_model)
    if detector_response_factor is not None and (not np.isfinite(detector_response_factor) or detector_response_factor <= 0):
        raise ValueError('detector_response_factor must be positive and finite.')
    if neutral_beam_overlap_factor is not None and (not np.isfinite(neutral_beam_overlap_factor) or neutral_beam_overlap_factor <= 0):
        raise ValueError('neutral_beam_overlap_factor must be positive and finite.')
    area = float(result['line_fit']['area_above_baseline'])
    corrected = area / detector_response_factor if detector_response_factor is not None else None
    beam_normalized = corrected / neutral_beam_overlap_factor if corrected is not None and neutral_beam_overlap_factor is not None else None
    return {'diagnostic': 'Charge-exchange spectroscopy', 'line_analysis': result,
        'raw_integrated_line_signal': area, 'detector_response_corrected_signal': corrected,
        'beam_overlap_normalized_signal': beam_normalized,
        'detector_response_factor': detector_response_factor,
        'neutral_beam_overlap_factor': neutral_beam_overlap_factor,
        'assumptions_and_limits': [
            'Line intensity requires calibrated detector response and background subtraction.',
            'Neutral-beam overlap normalization is only meaningful when the supplied factor is physically defined.',
            'Absolute impurity density and rotation/temperature profiles require atomic data and viewing geometry.'
        ]}

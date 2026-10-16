from __future__ import annotations
import numpy as np
from .optical_spectroscopy import analyze_optical_emission_spectrum


def analyze_vuv_spectrum(wavelength_nm, intensity, line_regions, *, detector_response=None, fit_model='Gaussian') -> dict:
    """Fit VUV emission lines and correct their integrated signals by supplied response factors."""
    fitted = analyze_optical_emission_spectrum(wavelength_nm, intensity, line_regions, fit_model=fit_model)
    response = detector_response or {}
    lines = []
    for line in fitted['lines']:
        label = line['label']
        area = float(line['fit']['area_above_baseline'])
        factor = response.get(label)
        if factor is not None and (not np.isfinite(factor) or factor <= 0):
            raise ValueError(f'Response factor for {label!r} must be positive and finite.')
        lines.append({'label': label, 'raw_integrated_signal': area, 'response_factor': factor,
                      'corrected_signal': float(area / factor) if factor is not None else None})
    valid = [x for x in lines if x['corrected_signal'] is not None]
    ratios = []
    if len(valid) >= 2:
        for row in valid[1:]:
            ratios.append({'numerator': row['label'], 'denominator': valid[0]['label'],
                           'ratio': float(row['corrected_signal'] / valid[0]['corrected_signal'])})
    return {'diagnostic': 'VUV spectroscopy', 'line_fits': fitted['lines'],
            'response_corrected_signals': lines, 'corrected_line_ratios': ratios,
            'assumptions_and_limits': [
                'Wavelength axis and detector response must be calibrated for the VUV optical path.',
                'Impurity density and charge-state inference require atomic collisional-radiative data.',
                'Line blends, opacity and non-equilibrium populations can bias line ratios.'
            ]}

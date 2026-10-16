from __future__ import annotations
import numpy as np
from .optical_spectroscopy import analyze_optical_emission_spectrum


def analyze_balmer_spectroscopy(wavelength_nm, intensity, *, line_regions, detector_response=None) -> dict:
    """Fit Balmer-series lines and compute response-corrected intensity ratios."""
    fitted = analyze_optical_emission_spectrum(wavelength_nm, intensity, line_regions, fit_model='Gaussian')
    response = detector_response or {}
    lines = []
    for line in fitted['lines']:
        label = line['label']
        area = float(line['fit']['area_above_baseline'])
        factor = response.get(label)
        if factor is not None and (not np.isfinite(factor) or factor <= 0):
            raise ValueError(f'Response factor for {label!r} must be positive and finite.')
        lines.append({'label': label, 'integrated_signal': area,
                      'response_corrected_signal': area/factor if factor is not None else None})
    ratios = []
    for i in range(len(lines)):
        for j in range(i+1, len(lines)):
            a, b = lines[i], lines[j]
            if a['response_corrected_signal'] is not None and b['response_corrected_signal'] is not None:
                ratios.append({'numerator': a['label'], 'denominator': b['label'],
                               'ratio': float(a['response_corrected_signal']/b['response_corrected_signal'])})
    return {'diagnostic': 'Hydrogen Balmer / Balmer-alpha spectroscopy', 'lines': lines,
        'response_corrected_ratios': ratios,
        'assumptions_and_limits': [
            'The wavelength scale and detector response must be calibrated.',
            'Balmer intensity ratios are not direct particle-flux measurements; excitation/recombination and opacity models are needed for physical interpretation.',
            'Line blending, background subtraction and spectral resolution affect the ratios.'
        ]}

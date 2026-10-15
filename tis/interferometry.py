from __future__ import annotations
import numpy as np
from .core import linear_phase_density


def analyze_interferometry(phase_rad, *, frequency_hz: float, sign_convention: int, path_length_m=None) -> dict:
    """Convert calibrated interferometer phase to line-integrated and optional average density."""
    phase = np.asarray(phase_rad, dtype=float)
    line_density = linear_phase_density(phase, frequency_hz, sign_convention=sign_convention)
    average_density = None
    if path_length_m is not None:
        length = np.asarray(path_length_m, dtype=float)
        if not np.all(np.isfinite(length)) or np.any(length <= 0):
            raise ValueError('path_length_m must be positive and finite.')
        average_density = line_density / length
    return {'diagnostic': 'Laser / microwave interferometry', 'phase_rad': phase,
        'frequency_Hz': float(frequency_hz), 'line_integrated_electron_density_m-2': line_density,
        'path_length_m': path_length_m, 'path_average_electron_density_m-3': average_density,
        'assumptions_and_limits': [
            'The phase sign convention must come from the instrument calibration.',
            'The result is line-integrated density; path-average density is not a local profile.',
            'Fringe jumps, vibrations, refraction, cutoffs and path geometry require separate handling.'
        ]}

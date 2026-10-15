from __future__ import annotations
import numpy as np
from .core import polynomial_calibration


def analyze_fild_image(image, *, energy_calibration_coefficients, pitch_calibration_coefficients, background_image=None) -> dict:
    """Map a calibrated fast-ion loss detector scintillator image to energy/pitch bins."""
    data = np.asarray(image, dtype=float)
    if data.ndim != 2 or min(data.shape) < 2 or not np.all(np.isfinite(data)):
        raise ValueError('FILD image must be a finite 2-D array with at least two pixels per axis.')
    if background_image is not None:
        bg = np.asarray(background_image, dtype=float)
        if bg.shape != data.shape or not np.all(np.isfinite(bg)):
            raise ValueError('background_image must be finite and match image shape.')
        corrected = data - bg
    else:
        corrected = data.copy()
    corrected = np.clip(corrected, 0, None)
    rows, cols = data.shape
    energy = polynomial_calibration(np.arange(cols, dtype=float), list(energy_calibration_coefficients))
    pitch = polynomial_calibration(np.arange(rows, dtype=float), list(pitch_calibration_coefficients))
    if np.any(np.diff(energy) <= 0) or np.any(np.diff(pitch) == 0) or not np.all(np.isfinite(energy)) or not np.all(np.isfinite(pitch)):
        raise ValueError('Pixel calibration must produce finite, monotonic energy and pitch axes.')
    total = float(np.sum(corrected))
    if total > 0:
        mean_energy = float(np.sum(corrected * energy[None, :]) / total)
        mean_pitch = float(np.sum(corrected * pitch[:, None]) / total)
    else:
        mean_energy = mean_pitch = None
    return {'diagnostic': 'Fast-ion loss detector (FILD)', 'energy_keV_by_column': energy,
        'pitch_angle_deg_by_row': pitch, 'background_corrected_image': corrected,
        'integrated_signal': total, 'intensity_weighted_mean_energy_keV': mean_energy,
        'intensity_weighted_mean_pitch_angle_deg': mean_pitch,
        'peak_pixel_row_col': tuple(int(i) for i in np.unravel_index(np.argmax(corrected), corrected.shape)),
        'assumptions_and_limits': ['Pixel-to-energy and pixel-to-pitch calibrations must come from the actual scintillator/optics geometry.',
            'Image intensity is not an absolute particle flux without scintillator yield and detector calibration.',
            'Scintillator saturation, decay, optical distortion and background can bias moments.']}

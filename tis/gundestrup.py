"""Angular-current analysis for Gundestrup/compound directional probes."""
from __future__ import annotations

import numpy as np


def analyze_gundestrup_probe(
    angle_rad,
    ion_saturation_current_A,
    *,
    modulation_per_mach: float | None = None,
) -> dict:
    """Fit first-harmonic angular current response and estimate flow direction.

    The normalized first-harmonic amplitude is a response proxy. Mach number is
    returned only if a probe-specific calibration factor is provided; sheath,
    geometry and magnetization corrections are not universal.
    """
    angle = np.asarray(angle_rad, dtype=float).ravel()
    current = np.asarray(ion_saturation_current_A, dtype=float).ravel()
    if angle.size != current.size or angle.size < 4:
        raise ValueError("At least four paired angular samples are required.")
    if not np.all(np.isfinite(angle)) or not np.all(np.isfinite(current)):
        raise ValueError("Angles and currents must be finite.")
    if np.unique(np.round(np.mod(angle, 2 * np.pi), 10)).size < 4:
        raise ValueError("At least four distinct probe angles are required.")
    if modulation_per_mach is not None and (
        not np.isfinite(modulation_per_mach) or modulation_per_mach <= 0
    ):
        raise ValueError("modulation_per_mach must be positive and finite.")
    design = np.column_stack([np.ones(angle.size), np.cos(angle), np.sin(angle)])
    coeff, *_ = np.linalg.lstsq(design, current, rcond=None)
    prediction = design @ coeff
    baseline, cosine, sine = map(float, coeff)
    amplitude = float(np.hypot(cosine, sine))
    if abs(baseline) <= np.finfo(float).eps:
        normalized_amplitude = None
        mach = None
    else:
        normalized_amplitude = float(amplitude / abs(baseline))
        mach = (
            float(normalized_amplitude / modulation_per_mach)
            if modulation_per_mach is not None else None
        )
    ss_res = float(np.sum((current - prediction) ** 2))
    ss_tot = float(np.sum((current - np.mean(current)) ** 2))
    return {
        "diagnostic": "Gundestrup / compound directional probe",
        "baseline_current_A": baseline,
        "first_harmonic_cosine_A": cosine,
        "first_harmonic_sine_A": sine,
        "angular_modulation_amplitude_A": amplitude,
        "normalized_angular_modulation": normalized_amplitude,
        "flow_direction_rad": float(np.arctan2(sine, cosine)),
        "mach_number_estimate": mach,
        "modulation_per_mach_calibration": modulation_per_mach,
        "r_squared": float(1 - ss_res / ss_tot) if ss_tot > 0 else None,
        "fitted_current_A": prediction,
        "assumptions_and_limits": [
            "First-harmonic angular response is a descriptive model of the measured current.",
            "Mach number requires probe-specific calibration and sheath/collection model.",
            "Probe orientation and angle convention determine the sign and direction interpretation.",
        ],
    }

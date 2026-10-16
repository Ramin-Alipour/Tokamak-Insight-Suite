"""Ball-pen probe potential proxy and fluctuation analysis."""
from __future__ import annotations

import numpy as np


def analyze_ball_pen_probe(
    voltage_V,
    *,
    electron_temperature_eV: float | None = None,
    correction_factor: float | None = None,
    time_s=None,
) -> dict:
    """Summarize ball-pen probe voltage and optionally apply a supplied Te correction.

    The plasma-potential estimate uses Vp ≈ V_BPP + alpha*Te[eV] only when the
    user supplies a device/configuration-specific alpha and a compatible Te.
    Raw ball-pen voltage is never silently equated with plasma potential.
    """
    v = np.asarray(voltage_V, dtype=float).ravel()
    if v.size < 2 or not np.all(np.isfinite(v)):
        raise ValueError("voltage_V must contain at least two finite samples.")
    if (electron_temperature_eV is None) != (correction_factor is None):
        raise ValueError("electron_temperature_eV and correction_factor must be supplied together.")
    corrected = None
    if electron_temperature_eV is not None:
        if not np.isfinite(electron_temperature_eV) or electron_temperature_eV < 0:
            raise ValueError("electron_temperature_eV must be finite and nonnegative.")
        if not np.isfinite(correction_factor):
            raise ValueError("correction_factor must be finite.")
        corrected = v + correction_factor * electron_temperature_eV
    time_audit = None
    if time_s is not None:
        t = np.asarray(time_s, dtype=float).ravel()
        if t.size != v.size or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("time_s must be finite, strictly increasing and match voltage_V.")
        dt = np.diff(t)
        time_audit = {"duration_s": float(t[-1] - t[0]), "median_sample_interval_s": float(np.median(dt))}
    return {
        "diagnostic": "Ball-pen probe",
        "raw_voltage_V": v,
        "mean_voltage_V": float(np.mean(v)),
        "rms_fluctuation_V": float(np.std(v, ddof=0)),
        "peak_to_peak_V": float(np.ptp(v)),
        "plasma_potential_estimate_V": corrected,
        "correction_factor": correction_factor,
        "electron_temperature_eV": electron_temperature_eV,
        "time_audit": time_audit,
        "assumptions_and_limits": [
            "The correction factor is probe/configuration dependent and must be independently justified.",
            "Potential estimate inherits Te and calibration uncertainty, which are not propagated here.",
            "Probe orientation, magnetic field, collector geometry and electrical response can affect interpretation.",
        ],
    }

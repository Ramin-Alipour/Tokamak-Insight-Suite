"""Toroidal-field coil / magnetic-probe calibrated signal analysis."""
from __future__ import annotations
import numpy as np

def analyze_toroidal_field_coil(time_s, signal, *, calibration_T_per_unit: float,
        signal_kind: str = "field", signal_uncertainty=None,
        calibration_uncertainty_T_per_unit: float = 0.0, baseline=None):
    """Convert calibrated coil signal samples to toroidal field and report quality metrics.

    Voltage input is converted only by the explicit T/V coefficient; no integration
    or coil transfer function is assumed. Geometry and absolute calibration are external.
    """
    t, x = np.asarray(time_s, dtype=float), np.asarray(signal, dtype=float)
    if t.ndim != 1 or x.ndim != 1 or t.size != x.size or t.size < 2:
        raise ValueError("time_s and signal must be 1-D arrays of equal length >= 2")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise ValueError("time_s and signal must contain only finite values")
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")
    if signal_kind not in {"field", "current", "voltage"}:
        raise ValueError("signal_kind must be field, current, or voltage")
    k, sk = float(calibration_T_per_unit), float(calibration_uncertainty_T_per_unit)
    if not np.isfinite(k) or not np.isfinite(sk) or sk < 0:
        raise ValueError("Calibration coefficient must be finite and uncertainty non-negative")
    if signal_kind == "field" and not np.isclose(k, 1.0):
        raise ValueError("field input requires calibration_T_per_unit=1")
    if baseline is not None and not np.isfinite(float(baseline)):
        raise ValueError("baseline must be finite")
    corrected = x - (0.0 if baseline is None else float(baseline))
    bt = corrected * k
    out = {"time_s": t.copy(), "signal_corrected": corrected, "B_toroidal_T": bt,
        "signal_kind": signal_kind, "calibration_T_per_unit": k,
        "baseline_input_units": baseline, "mean_B_toroidal_T": float(np.mean(bt)),
        "rms_B_toroidal_T": float(np.sqrt(np.mean(bt**2))),
        "std_B_toroidal_T": float(np.std(bt, ddof=1)),
        "peak_to_peak_B_toroidal_T": float(np.ptp(bt)),
        "sampling_rate_Hz": float(1.0 / np.median(dt)),
        "sampling_interval_jitter_fraction": float(np.std(dt) / np.mean(dt)),
        "physical_relation": "B_t(t) = k_cal * [signal(t) - baseline]",
        "limitations": ["Calibration must be valid over the signal range and bandwidth.",
        "No coil geometry, mutual inductance, plasma-current contribution, ferromagnetic response, or absolute calibration is inferred.",
        "Pickup-coil voltage generally requires a validated transfer function or Faraday-law integration; this function does not integrate voltage."]}
    if signal_uncertainty is not None:
        sx = np.asarray(signal_uncertainty, dtype=float)
        if sx.shape != x.shape or not np.all(np.isfinite(sx)) or np.any(sx < 0):
            raise ValueError("signal_uncertainty must match signal and be finite/non-negative")
        out["B_toroidal_uncertainty_T"] = np.sqrt((k*sx)**2 + (corrected*sk)**2)
    elif sk > 0:
        out["B_toroidal_uncertainty_T"] = np.abs(corrected)*sk
    return out

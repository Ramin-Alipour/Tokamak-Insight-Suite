"""Spatial potential-array gradients and synchronized fluctuation summaries."""
from __future__ import annotations

import numpy as np


def analyze_potential_array(position_m, potential_V, *, time_s=None) -> dict:
    """Compute one-dimensional electric-field estimates from a calibrated potential array.

    potential_V may be one spatial snapshot (channels,) or a time-by-channel matrix.
    The gradient is along the supplied coordinate; geometry projection to radial or
    poloidal directions must be handled by the caller.
    """
    x = np.asarray(position_m, dtype=float).ravel()
    v = np.asarray(potential_V, dtype=float)
    if x.size < 3 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("position_m must contain at least three strictly increasing finite positions.")
    if v.ndim not in (1, 2) or v.shape[-1] != x.size or not np.all(np.isfinite(v)):
        raise ValueError("potential_V must be finite with the channel dimension matching position_m.")
    electric_field = -np.gradient(v, x, axis=-1, edge_order=2)
    result = {
        "diagnostic": "Floating / plasma potential arrays",
        "position_m": x,
        "potential_V": v,
        "electric_field_V_m": electric_field,
        "mean_potential_by_channel_V": np.mean(v, axis=0) if v.ndim == 2 else v,
        "mean_electric_field_by_interval_V_m": np.mean(electric_field, axis=0) if v.ndim == 2 else electric_field,
        "assumptions_and_limits": [
            "Potential channels must be calibrated and refer to a consistent potential convention.",
            "Electric field is a one-dimensional spatial derivative along the supplied coordinate.",
            "Floating-potential gradients are not automatically plasma electric fields unless Te-related and sheath effects are addressed.",
        ],
    }
    if v.ndim == 2:
        result["potential_fluctuation_rms_by_channel_V"] = np.std(v, axis=0, ddof=0)
        result["potential_correlation_matrix"] = np.corrcoef(v, rowvar=False)
        if time_s is not None:
            t = np.asarray(time_s, dtype=float).ravel()
            if t.size != v.shape[0] or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
                raise ValueError("time_s must be finite, strictly increasing and match the time dimension.")
            result["duration_s"] = float(t[-1] - t[0])
    elif time_s is not None:
        raise ValueError("time_s is only meaningful for time-by-channel potential data.")
    return result

"""Computational workflow for calibrated Thomson-scattering profiles.

The input is already calibrated profile data, not raw detector counts.
"""
from __future__ import annotations

import numpy as np

from .thomson_scattering import build_profile, profile_summary


def analyze_thomson_profile(
    radius_m,
    density_m3,
    temperature_ev,
    *,
    density_uncertainty_m3=None,
    temperature_uncertainty_ev=None,
    major_radius_m=None,
):
    """Calculate profile summaries, radial gradients and local scale lengths."""
    p = build_profile(
        radius_m, density_m3, temperature_ev,
        density_uncertainty_m3, temperature_uncertainty_ev,
    )
    r, ne, te = p.radius_m, p.density_m3, p.temperature_ev
    edge_order = 2 if r.size >= 3 else 1
    dne_dr = np.gradient(ne, r, edge_order=edge_order)
    dte_dr = np.gradient(te, r, edge_order=edge_order)

    def scale_length(value, gradient):
        out = np.full(value.shape, np.nan, dtype=float)
        threshold = np.finfo(float).eps * np.maximum(1.0, np.abs(value))
        valid = (value > 0) & (np.abs(gradient) > threshold)
        out[valid] = -value[valid] / gradient[valid]
        return out

    result = {
        "radius_m": r.copy(),
        "density_m3": ne.copy(),
        "temperature_eV": te.copy(),
        "density_gradient_m-4": dne_dr,
        "temperature_gradient_eV_m-1": dte_dr,
        "density_normalized_gradient_m-1": np.divide(
            dne_dr, ne, out=np.full_like(dne_dr, np.nan), where=ne > 0
        ),
        "temperature_normalized_gradient_m-1": np.divide(
            dte_dr, te, out=np.full_like(dte_dr, np.nan), where=te > 0
        ),
        "density_scale_length_m": scale_length(ne, dne_dr),
        "temperature_scale_length_m": scale_length(te, dte_dr),
        "summary": profile_summary(p, major_radius_m=major_radius_m),
        "notes": [
            "Input must be calibrated Thomson-scattering profile data.",
            "Numerical gradients are sensitive to measurement noise.",
            "Volume averages, when requested, use circular large-aspect-ratio weighting.",
            "NaN scale lengths mark zero/near-zero gradients or profile values.",
        ],
    }
    if p.density_uncertainty_m3 is not None:
        result["density_uncertainty_m3"] = p.density_uncertainty_m3.copy()
    if p.temperature_uncertainty_ev is not None:
        result["temperature_uncertainty_eV"] = p.temperature_uncertainty_ev.copy()
    return result

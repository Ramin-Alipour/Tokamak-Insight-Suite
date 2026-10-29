"""Computational workflow for calibrated Thomson-scattering profiles.

Input arrays are already calibrated profiles, not raw detector counts. Optional
pointwise uncertainties are propagated through linear finite-difference gradients,
normalized gradients, local scale lengths, and profile averages assuming independent
input samples.
"""
from __future__ import annotations

import numpy as np

from .thomson_scattering import build_profile, profile_summary


def _scale_length(value, gradient):
    """Return -value/gradient where the gradient is numerically resolvable.

    The zero-gradient threshold is scaled to the gradient itself, so values
    with different physical units are never compared in the validity test.
    """
    out = np.full(value.shape, np.nan, dtype=float)
    gradient_scale = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    threshold = np.finfo(float).eps * gradient_scale
    valid = (value > 0) & np.isfinite(gradient) & (np.abs(gradient) > threshold)
    out[valid] = -value[valid] / gradient[valid]
    return out


def _gradient_uncertainty(derivative_matrix, uncertainty):
    if uncertainty is None:
        return None, None
    variance = (derivative_matrix**2) @ (uncertainty**2)
    standard_uncertainty = np.sqrt(np.maximum(variance, 0.0))
    covariance_value_gradient = np.diag(derivative_matrix) * uncertainty**2
    return standard_uncertainty, covariance_value_gradient


def _normalized_gradient_uncertainty(value, gradient, value_uncertainty, gradient_uncertainty, covariance):
    out = np.full(value.shape, np.nan, dtype=float)
    valid = value > 0
    if not np.any(valid):
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = (
            gradient_uncertainty[valid]**2 / value[valid]**2
            + gradient[valid]**2 * value_uncertainty[valid]**2 / value[valid]**4
            - 2.0 * gradient[valid] * covariance[valid] / value[valid]**3
        )
    out[valid] = np.sqrt(np.maximum(variance, 0.0))
    return out


def _scale_length_uncertainty(value, gradient, value_uncertainty, gradient_uncertainty, covariance):
    out = np.full(value.shape, np.nan, dtype=float)
    # Use the same gradient-relative, dimensionally consistent resolvability
    # threshold as _scale_length; never compare a gradient to the profile value.
    gradient_scale = float(np.max(np.abs(gradient))) if gradient.size else 0.0
    threshold = np.finfo(float).eps * gradient_scale
    valid = (value > 0) & np.isfinite(gradient) & (np.abs(gradient) > threshold)
    if not np.any(valid):
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = (
            value_uncertainty[valid]**2 / gradient[valid]**2
            + value[valid]**2 * gradient_uncertainty[valid]**2 / gradient[valid]**4
            - 2.0 * value[valid] * covariance[valid] / gradient[valid]**3
        )
    out[valid] = np.sqrt(np.maximum(variance, 0.0))
    return out


def analyze_thomson_profile(
    radius_m,
    density_m3,
    temperature_ev,
    *,
    density_uncertainty_m3=None,
    temperature_uncertainty_ev=None,
):
    """Calculate profile summaries, gradients, scale lengths, and available uncertainties."""
    p = build_profile(
        radius_m, density_m3, temperature_ev,
        density_uncertainty_m3, temperature_uncertainty_ev,
    )
    r, ne, te = p.radius_m, p.density_m3, p.temperature_ev
    edge_order = 2 if r.size >= 3 else 1
    derivative_matrix = np.gradient(np.eye(r.size), r, axis=0, edge_order=edge_order)
    dne_dr = derivative_matrix @ ne
    dte_dr = derivative_matrix @ te
    density_normalized_gradient = np.divide(
        dne_dr, ne, out=np.full_like(dne_dr, np.nan), where=ne > 0
    )
    temperature_normalized_gradient = np.divide(
        dte_dr, te, out=np.full_like(dte_dr, np.nan), where=te > 0
    )
    result = {
        "radius_m": r.copy(),
        "density_m3": ne.copy(),
        "temperature_eV": te.copy(),
        "density_gradient_m-4": dne_dr,
        "temperature_gradient_eV_m-1": dte_dr,
        "density_normalized_gradient_m-1": density_normalized_gradient,
        "temperature_normalized_gradient_m-1": temperature_normalized_gradient,
        "density_scale_length_m": _scale_length(ne, dne_dr),
        "temperature_scale_length_m": _scale_length(te, dte_dr),
        "summary": profile_summary(p),
        "notes": [
            "Input must be calibrated Thomson-scattering profile data.",
            "Gradient and derived uncertainty propagation assumes independent pointwise errors.",
            "NaN scale lengths mark zero/near-zero gradients or profile values.",
            "Peak-location uncertainty is not estimated by this workflow.",
        ],
    }
    if p.density_uncertainty_m3 is not None:
        sne = p.density_uncertainty_m3
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, sne)
        result["density_uncertainty_m3"] = sne.copy()
        result["density_gradient_uncertainty_m-4"] = grad_unc
        result["density_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            ne, dne_dr, sne, grad_unc, cov
        )
        result["density_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            ne, dne_dr, sne, grad_unc, cov
        )
    if p.temperature_uncertainty_ev is not None:
        ste = p.temperature_uncertainty_ev
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, ste)
        result["temperature_uncertainty_eV"] = ste.copy()
        result["temperature_gradient_uncertainty_eV_m-1"] = grad_unc
        result["temperature_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            te, dte_dr, ste, grad_unc, cov
        )
        result["temperature_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            te, dte_dr, ste, grad_unc, cov
        )
    return result

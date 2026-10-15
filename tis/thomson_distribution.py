"""Regularized non-negative inversion of a supplied Thomson-scattering kernel."""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

M_E = 9.1093837139e-31
E_CHARGE = 1.602176634e-19


def invert_thomson_distribution(
    measured_spectrum,
    scattering_kernel,
    velocity_grid_m_s,
    *,
    regularization: float = 1e-2,
    uncertainty=None,
) -> dict:
    """Infer a 1-D projected electron velocity distribution from a supplied kernel.

    The caller must construct the kernel using the actual scattering geometry,
    wavelength calibration, instrument response and chosen plasma response model.
    This routine solves a nonnegative, second-difference-regularized inverse
    problem; it does not generate a Thomson-scattering kernel itself.
    """
    y = np.asarray(measured_spectrum, dtype=float).ravel()
    k = np.asarray(scattering_kernel, dtype=float)
    v = np.asarray(velocity_grid_m_s, dtype=float).ravel()
    if k.ndim != 2 or k.shape != (y.size, v.size):
        raise ValueError("scattering_kernel shape must be (number of spectrum bins, number of velocity bins).")
    if y.size < 3 or v.size < 3 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(k)) or not np.all(np.isfinite(v)):
        raise ValueError("Spectrum, kernel and velocity grid must be finite and have at least three bins.")
    if np.any(y < 0) or np.any(k < 0):
        raise ValueError("Measured spectrum and scattering kernel must be nonnegative after background subtraction.")
    if np.any(np.diff(v) <= 0):
        raise ValueError("velocity_grid_m_s must be strictly increasing.")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and nonnegative.")

    if uncertainty is None:
        sigma = np.sqrt(np.maximum(y, max(float(np.max(y)) * 1e-6, np.finfo(float).eps)))
    else:
        sigma = np.asarray(uncertainty, dtype=float).ravel()
        if sigma.size != y.size or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("uncertainty must be positive, finite, and match measured_spectrum.")
    weighted_k = k / sigma[:, None]
    weighted_y = y / sigma
    # Scaled second difference discourages oscillatory solutions while preserving
    # broad/non-Maxwellian features; lambda=0 gives nonnegative weighted LSQ.
    d2 = np.diff(np.eye(v.size), n=2, axis=0)
    scale = max(float(np.linalg.norm(weighted_k, ord=2)), np.finfo(float).eps)
    a = np.vstack([weighted_k, np.sqrt(regularization) * scale * d2])
    b = np.r_[weighted_y, np.zeros(d2.shape[0])]
    fit = lsq_linear(a, b, bounds=(0, np.inf), method="trf", lsmr_tol="auto", max_iter=5000)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Thomson distribution inversion failed: {fit.message}")
    distribution = np.maximum(fit.x, 0)
    norm = float(np.trapezoid(distribution, v))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Inversion returned a zero or non-finite distribution.")
    normalized = distribution / norm
    mean_v = float(np.trapezoid(v * normalized, v))
    variance = float(np.trapezoid((v - mean_v) ** 2 * normalized, v))
    prediction = k @ distribution
    residual = y - prediction
    weighted_chi2 = float(np.sum((residual / sigma) ** 2))
    dof = max(1, y.size - int(np.count_nonzero(distribution > np.max(distribution) * 1e-8)))
    return {
        "diagnostic": "Thomson spectral distribution analysis",
        "velocity_grid_m_s": v,
        "distribution_raw": distribution,
        "distribution_normalized": normalized,
        "mean_projected_velocity_m_s": mean_v,
        "projected_velocity_variance_m2_s2": variance,
        "effective_projected_temperature_eV": float(M_E * variance / E_CHARGE),
        "predicted_spectrum": prediction,
        "residuals": residual,
        "weighted_chi_squared": weighted_chi2,
        "reduced_weighted_chi_squared_approx": weighted_chi2 / dof,
        "regularization": float(regularization),
        "solver_status": fit.message,
        "assumptions_and_limits": [
            "The supplied kernel must include the correct scattering geometry and instrument response.",
            "This is a 1-D projected velocity distribution, not a full 3-D phase-space distribution.",
            "Regularization biases fine-scale structure; report sensitivity to regularization and kernel uncertainty.",
            "Background subtraction and uncertainty estimates are supplied/assumed by the caller.",
        ],
    }

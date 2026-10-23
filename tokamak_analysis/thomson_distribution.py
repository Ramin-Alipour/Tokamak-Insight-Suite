"""Model-independent diagnostics for calibrated Thomson-scattering spectra.

This module characterizes the measured spectral distribution after local linear
background removal and Gaussian instrument-width deconvolution. It does not
pretend that arbitrary spectral moments uniquely recover an electron velocity
distribution; physical distribution inference requires an experiment-appropriate
collective/non-collective forward model and instrument response.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _vector(values, name, *, minimum=9):
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum} samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _gaussian_background(x, amplitude, center, sigma, intercept, slope):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + intercept + slope * (x - center)


def analyze_thomson_spectral_distribution(
    wavelength_nm,
    signal,
    *,
    instrument_sigma_nm: float,
    signal_uncertainty=None,
    background_fraction: float = 0.1,
) -> dict:
    """Return spectral moments, instrument-corrected width, and Gaussian-fit diagnostics.

    Wavelengths must be calibrated and strictly increasing; instrument_sigma_nm
    is the known Gaussian standard deviation of the line-spread function. A
    robust edge estimate defines a local linear background. Moments are computed
    from positive background-subtracted spectral weights. Instrument deconvolution
    is applied to the variance only and fails explicitly if the width is unresolved.
    """
    x = _vector(wavelength_nm, "wavelength_nm")
    y = _vector(signal, "signal")
    if x.size != y.size:
        raise ValueError("wavelength_nm and signal must have equal lengths")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if not np.isfinite(instrument_sigma_nm) or instrument_sigma_nm <= 0:
        raise ValueError("instrument_sigma_nm must be positive and finite")
    if not np.isfinite(background_fraction) or not (0.02 <= background_fraction <= 0.4):
        raise ValueError("background_fraction must be finite and in [0.02, 0.4]")

    uncertainty = None
    if signal_uncertainty is not None:
        uncertainty = _vector(signal_uncertainty, "signal_uncertainty")
        if uncertainty.size != y.size:
            raise ValueError("signal_uncertainty must have the same length as signal")
        if np.any(uncertainty <= 0):
            raise ValueError("signal_uncertainty must be strictly positive")

    n_edge = max(3, int(np.ceil(x.size * background_fraction)))
    edge_indices = np.r_[np.arange(n_edge), np.arange(x.size - n_edge, x.size)]
    edge_x, edge_y = x[edge_indices], y[edge_indices]
    slope, intercept_at_zero = np.polyfit(edge_x, edge_y, 1)
    background = slope * x + intercept_at_zero
    corrected = y - background
    weights = np.clip(corrected, 0.0, None)
    area = float(np.trapezoid(weights, x) if hasattr(np, "trapezoid") else np.trapz(weights, x))
    area_tolerance = 100.0 * np.finfo(float).eps * max(float(np.max(np.abs(y))), 1.0) * float(x[-1] - x[0])
    if not np.isfinite(area) or area <= area_tolerance:
        raise ValueError("No positive spectral area remains after background subtraction")
    # Use wavelength-space quadrature weights, not raw sample counts. This keeps
    # moments correct for nonuniform wavelength grids (e.g. resampled spectrometers).
    quadrature = np.empty_like(x)
    quadrature[0] = 0.5 * (x[1] - x[0])
    quadrature[-1] = 0.5 * (x[-1] - x[-2])
    quadrature[1:-1] = 0.5 * (x[2:] - x[:-2])
    spectral_measure = weights * quadrature
    measure_sum = float(np.sum(spectral_measure))
    if not np.isfinite(measure_sum) or measure_sum <= 0:
        raise ValueError("No positive integrated spectral weight remains")
    normalized = spectral_measure / measure_sum
    center = float(np.sum(normalized * x))
    centered = x - center
    variance_observed = float(np.sum(normalized * centered**2))
    if variance_observed <= 0:
        raise ValueError("Spectral distribution has zero observed variance")
    sigma_observed = float(np.sqrt(variance_observed))
    variance_physical = variance_observed - float(instrument_sigma_nm)**2
    resolved = variance_physical > 0
    sigma_physical = float(np.sqrt(variance_physical)) if resolved else None
    third = float(np.sum(normalized * centered**3))
    fourth = float(np.sum(normalized * centered**4))
    skewness = third / variance_observed**1.5
    excess_kurtosis = fourth / variance_observed**2 - 3.0

    # Fit a Gaussian plus linear background for a transparent shape comparison.
    peak = int(np.argmax(corrected))
    amplitude0 = float(max(corrected[peak], np.finfo(float).eps))
    spacing = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    p0 = [amplitude0, float(x[peak]), max(sigma_observed, spacing), float(intercept_at_zero), float(slope)]
    lower = [0.0, float(x[0]), max(spacing * 0.25, np.finfo(float).eps), -np.inf, -np.inf]
    upper = [np.inf, float(x[-1]), span, np.inf, np.inf]
    fit_sigma = None if uncertainty is None else uncertainty
    params, covariance = curve_fit(
        _gaussian_background, x, y, p0=p0, bounds=(lower, upper),
        sigma=fit_sigma, absolute_sigma=uncertainty is not None, maxfev=50000,
    )
    fitted = _gaussian_background(x, *params)
    residual = y - fitted
    dof = max(1, x.size - len(params))
    chi_square = None if uncertainty is None else float(np.sum((residual / uncertainty) ** 2))
    reduced_chi_square = None if chi_square is None else chi_square / dof
    ss_total = float(np.sum((y - np.mean(y))**2))
    r_squared = float(1.0 - np.sum(residual**2) / ss_total) if ss_total > 0 else None
    parameter_uncertainties = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return {
        "wavelength_nm": x.copy(),
        "signal": y.copy(),
        "background": background,
        "background_subtracted_signal": corrected,
        "positive_spectral_weights": weights,
        "wavelength_quadrature_weights_nm": quadrature,
        "integrated_spectral_measure": spectral_measure,
        "integrated_positive_spectral_signal": area,
        "spectral_centroid_nm": center,
        "observed_sigma_nm": sigma_observed,
        "observed_variance_nm2": variance_observed,
        "instrument_sigma_nm": float(instrument_sigma_nm),
        "instrument_corrected_sigma_nm": sigma_physical,
        "instrument_corrected_variance_nm2": float(variance_physical) if resolved else None,
        "instrument_width_resolved": bool(resolved),
        "spectral_skewness": float(skewness),
        "spectral_excess_kurtosis": float(excess_kurtosis),
        "gaussian_fit_parameters": {
            "amplitude": float(params[0]), "center_nm": float(params[1]),
            "sigma_nm": float(params[2]), "background_intercept": float(params[3]),
            "background_slope_per_nm": float(params[4]),
        },
        "gaussian_fit_parameter_uncertainties": {
            "amplitude": float(parameter_uncertainties[0]),
            "center_nm": float(parameter_uncertainties[1]),
            "sigma_nm": float(parameter_uncertainties[2]),
            "background_intercept": float(parameter_uncertainties[3]),
            "background_slope_per_nm": float(parameter_uncertainties[4]),
        },
        "fitted_spectrum": fitted,
        "residual": residual,
        "degrees_of_freedom": int(dof),
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "r_squared": r_squared,
        "model": "positive spectral moments after local linear-background subtraction; Gaussian fit for shape diagnostics",
        "assumptions": [
            "Calibrated, strictly increasing wavelength axis",
            "Known Gaussian instrumental line-spread standard deviation",
            "Locally linear background estimated from both spectral-window edges",
            "Positive background-subtracted signal used as spectral weights",
        ],
        "limitations": [
            "Spectral moments characterize measured wavelength distribution and are not a unique electron velocity distribution",
            "Instrument-width deconvolution assumes independent Gaussian variances and is unresolved when observed variance does not exceed instrument variance",
            "No electron temperature or density is inferred by this generic distribution analysis",
            "Collective, relativistic, non-Maxwellian physical inference requires an appropriate validated forward model and instrument response",
            "Uncertainty output for Gaussian-fit parameters is local covariance only; moment uncertainties are not propagated",
        ],
    }

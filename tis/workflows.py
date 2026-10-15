"""Workflow-level helpers for calibrated tokamak diagnostics.

These functions wrap numerical primitives with input contracts and quality flags.
They do not replace instrument-specific calibration or scientific validation.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .core import spectral_calibration


def wavelength_calibration_workflow(
    observed_pixel: Any,
    reference_wavelength_nm: Any,
    *,
    degree: int = 1,
    max_rms_residual_nm: float | None = None,
    monotonic_samples: int = 1001,
) -> dict[str, Any]:
    """Calibrate detector pixel coordinates against known spectral lines.

    Parameters
    ----------
    observed_pixel:
        Fitted reference-line centroids in detector pixel coordinates.
    reference_wavelength_nm:
        Corresponding known wavelengths in nm, in the same order as centroids.
    degree:
        Polynomial degree, >=1. At least degree+2 matched lines are required.
    max_rms_residual_nm:
        Optional application-specific acceptance threshold. If omitted, no
        pass/fail threshold is invented; quality is reported descriptively.
    monotonic_samples:
        Number of points used to check that the fitted mapping is monotonic
        across the calibration span.

    Returns
    -------
    Dictionary containing polynomial coefficients (highest order first),
    per-line residuals, fit metrics, monotonicity and an explicit acceptance
    status. Residuals describe fit agreement, not absolute wavelength accuracy.
    """
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError("degree must be a positive integer.")
    if isinstance(monotonic_samples, (bool, np.bool_)) or not isinstance(monotonic_samples, (int, np.integer)) or monotonic_samples < 3:
        raise ValueError("monotonic_samples must be an integer >= 3.")
    if max_rms_residual_nm is not None and (
        not np.isfinite(max_rms_residual_nm) or max_rms_residual_nm < 0
    ):
        raise ValueError("max_rms_residual_nm must be finite and nonnegative.")

    pixels = np.asarray(observed_pixel, dtype=float).ravel()
    wavelengths = np.asarray(reference_wavelength_nm, dtype=float).ravel()
    if len(pixels) != len(wavelengths):
        raise ValueError("Pixel centroids and reference wavelengths must have equal lengths.")
    if len(pixels) < degree + 2:
        raise ValueError(f"At least {degree + 2} matched lines are required for degree {degree}.")
    if not np.all(np.isfinite(pixels)) or not np.all(np.isfinite(wavelengths)):
        raise ValueError("Calibration pairs must be finite; remove invalid pairs explicitly.")
    if len(np.unique(pixels)) != len(pixels):
        raise ValueError("Observed pixel centroids must be unique.")
    if len(np.unique(wavelengths)) != len(wavelengths):
        raise ValueError("Reference wavelengths must be unique.")
    if np.ptp(pixels) <= 0 or np.ptp(wavelengths) <= 0:
        raise ValueError("Calibration points must span nonzero pixel and wavelength ranges.")

    fit = spectral_calibration(pixels, wavelengths, int(degree))
    coefficients = np.asarray(fit["coefficients_high_to_low"], dtype=float)
    span = np.linspace(float(np.min(pixels)), float(np.max(pixels)), int(monotonic_samples))
    mapped = np.polyval(coefficients, span)
    delta = np.diff(mapped)
    monotonic = bool(np.all(delta > 0) or np.all(delta < 0))
    residuals = np.asarray(fit["residuals"], dtype=float)
    rms = float(np.sqrt(np.mean(residuals**2)))
    acceptance = "not_assessed"
    if max_rms_residual_nm is not None:
        acceptance = "pass" if rms <= max_rms_residual_nm and monotonic else "fail"

    return {
        **fit,
        "observed_pixel": pixels.tolist(),
        "reference_wavelength_nm": wavelengths.tolist(),
        "fitted_wavelength_nm": np.polyval(coefficients, pixels).tolist(),
        "residuals_nm": residuals.tolist(),
        "rms_residual_nm": rms,
        "mean_absolute_residual_nm": float(np.mean(np.abs(residuals))),
        "residual_standard_deviation_nm": float(np.std(residuals, ddof=1)),
        "monotonic_over_calibration_span": monotonic,
        "calibration_span_pixel": [float(np.min(pixels)), float(np.max(pixels))],
        "calibration_span_nm": [float(np.min(wavelengths)), float(np.max(wavelengths))],
        "max_rms_residual_nm": max_rms_residual_nm,
        "acceptance_status": acceptance,
        "interpretation": (
            "Residuals quantify agreement with supplied reference lines only; "
            "they do not independently establish absolute accuracy, line-centroid "
            "uncertainty, instrument drift, or validity outside the calibrated span."
        ),
    }

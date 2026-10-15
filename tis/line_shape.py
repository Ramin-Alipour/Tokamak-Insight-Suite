"""Instrument line-shape workflow for tokamak spectroscopy."""
from __future__ import annotations

from typing import Any
import numpy as np
from .core import fit_spectral_line


def instrument_line_shape_workflow(
    wavelength_nm: Any,
    intensity: Any,
    *,
    reference_wavelength_nm: float,
    model: str = "Gaussian",
    fit_min_nm: float | None = None,
    fit_max_nm: float | None = None,
    intensity_sigma: Any | None = None,
    minimum_r_squared: float | None = None,
) -> dict[str, Any]:
    """Fit a narrow reference line and report observed FWHM and nominal resolving power.

    The observed width approximates the instrument line-spread function only
    when the reference line's intrinsic width is negligible or independently
    known. This routine does not deconvolve intrinsic width.
    """
    if not np.isfinite(reference_wavelength_nm) or reference_wavelength_nm <= 0:
        raise ValueError("reference_wavelength_nm must be positive and finite.")
    if minimum_r_squared is not None and (
        not np.isfinite(minimum_r_squared) or not 0 <= minimum_r_squared <= 1
    ):
        raise ValueError("minimum_r_squared must lie in [0, 1].")
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(intensity, dtype=float).ravel()
    if len(x) != len(y):
        raise ValueError("Wavelength and intensity arrays must have equal lengths.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("The selected reference spectrum must contain finite values.")
    if len(x) < 7 or len(np.unique(x)) != len(x):
        raise ValueError("At least seven samples with unique wavelength coordinates are required.")
    if fit_min_nm is not None and not np.isfinite(fit_min_nm):
        raise ValueError("fit_min_nm must be finite when supplied.")
    if fit_max_nm is not None and not np.isfinite(fit_max_nm):
        raise ValueError("fit_max_nm must be finite when supplied.")
    if fit_min_nm is not None and fit_max_nm is not None and fit_min_nm >= fit_max_nm:
        raise ValueError("fit_min_nm must be below fit_max_nm.")

    fit = fit_spectral_line(
        x, y, xmin=fit_min_nm, xmax=fit_max_nm, model=model, sigma_y=intensity_sigma
    )
    fwhm = float(fit["FWHM"])
    if not np.isfinite(fwhm) or fwhm <= 0:
        raise ValueError("Fitted line width is not positive and finite.")
    center = float(fit["center"])
    result = dict(fit)
    result.update({
        "reference_wavelength_nm": float(reference_wavelength_nm),
        "observed_center_nm": center,
        "center_offset_from_reference_nm": center - float(reference_wavelength_nm),
        "observed_fwhm_nm": fwhm,
        "nominal_resolving_power": float(center / fwhm),
        "fit_quality_status": (
            "not_assessed" if minimum_r_squared is None
            else ("pass" if float(fit["r_squared"]) >= minimum_r_squared else "fail")
        ),
        "interpretation": (
            "Observed width represents the instrument line-spread function only "
            "when the reference line's intrinsic width is negligible or independently "
            "known. Do not subtract widths without a justified convolution model."
        ),
    })
    if "standard_error_FWHM" in fit:
        sigma_width = float(fit["standard_error_FWHM"])
        result["standard_error_observed_fwhm_nm"] = sigma_width
        result["approx_standard_error_resolving_power"] = float(
            abs(center / (fwhm * fwhm)) * sigma_width
        )
    return result

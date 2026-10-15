"""CXRS line-profile workflow for per-chord ion temperature and line-of-sight flow."""
from __future__ import annotations

import numpy as np

from .core import doppler_temperature_from_fwhm_eV, doppler_velocity_m_s, fit_spectral_line


def analyze_cxrs_line(
    wavelength_nm,
    intensity,
    *,
    rest_wavelength_nm: float,
    ion_mass_amu: float,
    instrumental_fwhm_nm: float = 0.0,
    fit_min_nm: float | None = None,
    fit_max_nm: float | None = None,
    fit_model: str = "Gaussian",
    viewing_projection: float | None = None,
) -> dict:
    """Fit a CXRS spectral line and derive per-chord LOS velocity and ion temperature.

    viewing_projection is the cosine/projection factor mapping a specified
    flow component to the measured LOS component. It is not used to infer a
    complete toroidal/poloidal velocity vector; that requires multiple chords
    and a geometry-aware inversion.
    """
    if not np.isfinite(rest_wavelength_nm) or rest_wavelength_nm <= 0:
        raise ValueError("rest_wavelength_nm must be positive and finite.")
    if not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0:
        raise ValueError("ion_mass_amu must be positive and finite.")
    if not np.isfinite(instrumental_fwhm_nm) or instrumental_fwhm_nm < 0:
        raise ValueError("instrumental_fwhm_nm must be finite and nonnegative.")
    if viewing_projection is not None and (
        not np.isfinite(viewing_projection) or abs(viewing_projection) > 1
        or abs(viewing_projection) < np.finfo(float).eps
    ):
        raise ValueError("viewing_projection must be finite, nonzero, and within [-1, 1].")

    line = fit_spectral_line(
        wavelength_nm, intensity, xmin=fit_min_nm, xmax=fit_max_nm, model=fit_model
    )
    center = float(line["center"])
    observed_fwhm = float(line["FWHM"])
    if observed_fwhm <= instrumental_fwhm_nm:
        raise ValueError(
            "Observed line width is unresolved by the supplied instrumental width; "
            "ion temperature is not identifiable from this fit."
        )
    velocity = doppler_velocity_m_s(center, rest_wavelength_nm)
    temp = doppler_temperature_from_fwhm_eV(
        observed_fwhm=observed_fwhm,
        lambda0=rest_wavelength_nm,
        mass_amu=ion_mass_amu,
        instrumental_fwhm=instrumental_fwhm_nm,
    )
    return {
        "diagnostic": "Charge-exchange recombination spectroscopy (CXRS)",
        "line_fit": line,
        "rest_wavelength_nm": float(rest_wavelength_nm),
        "observed_centroid_nm": center,
        "observed_fwhm_nm": observed_fwhm,
        "instrumental_fwhm_nm": float(instrumental_fwhm_nm),
        "line_of_sight_velocity_m_s": float(velocity),
        "ion_temperature": temp,
        "viewing_projection": viewing_projection,
        "projected_flow_component_m_s": (
            float(velocity / viewing_projection) if viewing_projection is not None else None
        ),
        "assumptions_and_limits": [
            "Single isolated line; blends and background structure must be handled in the selected fit window.",
            "Instrumental and observed widths are assumed Gaussian for quadrature deconvolution.",
            "Thermal Doppler broadening is assumed after instrumental correction; Stark, Zeeman, opacity, fine-structure and non-thermal broadening are not separated.",
            "LOS velocity is not a toroidal or poloidal velocity without the appropriate viewing geometry and multi-chord inversion.",
            "Neutral-beam overlap, charge-exchange atomic rates and impurity density inference are outside this single-line workflow.",
        ],
    }

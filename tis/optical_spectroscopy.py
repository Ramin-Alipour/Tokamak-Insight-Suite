"""Multi-line optical emission spectrum fitting."""
from __future__ import annotations

import numpy as np
from .core import fit_spectral_line, doppler_velocity_m_s


def analyze_optical_emission_spectrum(wavelength_nm, intensity, line_regions, *, fit_model="Gaussian") -> dict:
    """Fit user-specified emission-line regions on a calibrated wavelength axis.

    Each region is a mapping with label, minimum_nm and maximum_nm, and optional
    rest_wavelength_nm. Species/charge-state identification remains dependent on
    an external atomic line list and uncertainty in the wavelength calibration.
    """
    wavelength = np.asarray(wavelength_nm, dtype=float).ravel()
    signal = np.asarray(intensity, dtype=float).ravel()
    if wavelength.size != signal.size or wavelength.size < 7:
        raise ValueError("Wavelength and intensity must have equal lengths of at least seven.")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(signal)):
        raise ValueError("Wavelength and intensity must be finite.")
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("Wavelength axis must be strictly increasing.")
    if not isinstance(line_regions, (list, tuple)) or not line_regions:
        raise ValueError("line_regions must be a nonempty list of line-region mappings.")
    lines = []
    for region in line_regions:
        if not isinstance(region, dict):
            raise ValueError("Each line region must be a mapping.")
        label = str(region.get("label", "line"))
        lower, upper = region.get("minimum_nm"), region.get("maximum_nm")
        if lower is None or upper is None or not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(f"Line region {label!r} needs finite minimum_nm < maximum_nm.")
        fit = fit_spectral_line(wavelength, signal, xmin=float(lower), xmax=float(upper), model=fit_model)
        rest = region.get("rest_wavelength_nm")
        velocity = None
        if rest is not None:
            if not np.isfinite(rest) or rest <= 0:
                raise ValueError(f"Line region {label!r} has invalid rest_wavelength_nm.")
            velocity = float(doppler_velocity_m_s(float(fit["center"]), float(rest)))
        lines.append({
            "label": label,
            "fit": fit,
            "rest_wavelength_nm": float(rest) if rest is not None else None,
            "line_of_sight_velocity_m_s": velocity,
        })
    return {
        "diagnostic": "Visible / optical emission spectroscopy",
        "n_lines_fitted": len(lines),
        "lines": lines,
        "assumptions_and_limits": [
            "Wavelength axis must already be calibrated and the background treated appropriately.",
            "Line windows should isolate individual features; blends can bias centroids and widths.",
            "Instrumental, Stark, Zeeman, opacity and fine-structure broadening are not separated automatically.",
            "Species identification requires a suitable atomic line database and compatible plasma conditions.",
        ],
    }

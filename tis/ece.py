"""Basic electron-cyclotron-emission channel mapping and brightness-temperature inference."""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
M_E = 9.1093837139e-31


def analyze_ece_channels(
    frequency_hz,
    *,
    toroidal_field_axis_T: float,
    major_radius_axis_m: float,
    harmonic: int = 1,
    brightness_temperature_eV=None,
    optical_depth=None,
) -> dict:
    """Map ECE channel frequencies to approximate resonance fields/radii.

    Uses f = harmonic * e B / (2 pi m_e) and the large-aspect-ratio
    toroidal-field approximation B_t(R) = B_axis * R_axis / R. This is a
    channel-mapping estimate, not an equilibrium reconstruction.

    Optional brightness temperatures are inferred electron temperatures only
    under the optically thick assumption (tau omitted) or with a supplied
    positive optical depth and negligible incident background:
    T_e = T_b / (1 - exp(-tau)).
    """
    f = np.asarray(frequency_hz, dtype=float).ravel()
    if f.size == 0 or not np.all(np.isfinite(f)) or np.any(f <= 0):
        raise ValueError("frequency_hz must contain positive finite frequencies.")
    if not np.isfinite(toroidal_field_axis_T) or toroidal_field_axis_T <= 0:
        raise ValueError("toroidal_field_axis_T must be positive and finite.")
    if not np.isfinite(major_radius_axis_m) or major_radius_axis_m <= 0:
        raise ValueError("major_radius_axis_m must be positive and finite.")
    if isinstance(harmonic, bool) or not isinstance(harmonic, (int, np.integer)) or harmonic < 1:
        raise ValueError("harmonic must be a positive integer.")

    b_res = 2 * np.pi * M_E * f / (harmonic * E_CHARGE)
    radius = major_radius_axis_m * toroidal_field_axis_T / b_res
    result = {
        "diagnostic": "Electron cyclotron emission",
        "frequency_Hz": f,
        "resonance_field_T": b_res,
        "approximate_major_radius_m": radius,
        "harmonic": int(harmonic),
        "mapping_model": "large-aspect-ratio B_t(R)=B_axis*R_axis/R",
        "electron_temperature_eV": None,
        "temperature_interpretation": None,
        "assumptions_and_limits": [
            "Approximate toroidal-field resonance mapping; use a machine equilibrium for precision.",
            "Relativistic, Doppler, finite-orbit, harmonic-overlap and mode effects are not included.",
            "ECE brightness temperature equals local electron temperature only when emission is sufficiently optically thick.",
        ],
    }
    if brightness_temperature_eV is not None:
        tb = np.asarray(brightness_temperature_eV, dtype=float).ravel()
        if tb.size != f.size or not np.all(np.isfinite(tb)) or np.any(tb < 0):
            raise ValueError("brightness_temperature_eV must match frequency_hz and be finite/nonnegative.")
        if optical_depth is None:
            result["electron_temperature_eV"] = tb
            result["temperature_interpretation"] = "brightness temperature; treated as Te only under optically thick assumption"
        else:
            tau = np.asarray(optical_depth, dtype=float)
            if tau.ndim == 0:
                tau = np.full(f.size, float(tau))
            else:
                tau = tau.ravel()
            if tau.size != f.size or not np.all(np.isfinite(tau)) or np.any(tau <= 0):
                raise ValueError("optical_depth must be positive, finite, and scalar or match frequency_hz.")
            correction = -np.expm1(-tau)
            result["electron_temperature_eV"] = tb / correction
            result["temperature_interpretation"] = "brightness temperature corrected for supplied optical depth; background assumed negligible"
    return result

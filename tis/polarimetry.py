"""Faraday-rotation forward model and density-weighted line-integrated field estimate."""
from __future__ import annotations

import numpy as np

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
M_E = 9.1093837139e-31
C_LIGHT = 299792458.0
FARADAY_COEFFICIENT_SI = E_CHARGE**3 / (8 * np.pi**2 * EPS0 * M_E**2 * C_LIGHT**3)


def analyze_faraday_rotation(
    wavelength_m: float,
    *,
    rotation_angle_rad: float | None = None,
    path_m=None,
    electron_density_m3=None,
    magnetic_field_parallel_T=None,
) -> dict:
    """Compute Faraday rotation or infer a density-weighted B_parallel average.

    For spatial profiles, path_m, electron_density_m3 and magnetic_field_parallel_T
    must be equal-length arrays along the same line of sight. If an observed
    rotation and density profile are supplied, the routine estimates the
    density-weighted mean B_parallel, assuming the density profile is calibrated.
    """
    if not np.isfinite(wavelength_m) or wavelength_m <= 0:
        raise ValueError("wavelength_m must be positive and finite.")
    if rotation_angle_rad is not None and not np.isfinite(rotation_angle_rad):
        raise ValueError("rotation_angle_rad must be finite.")
    supplied = (path_m is not None, electron_density_m3 is not None, magnetic_field_parallel_T is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("path_m, electron_density_m3 and magnetic_field_parallel_T must be supplied together.")
    forward_angle = None
    line_integral_ne_b = None
    line_integral_ne = None
    if all(supplied):
        s = np.asarray(path_m, dtype=float).ravel()
        ne = np.asarray(electron_density_m3, dtype=float).ravel()
        bpar = np.asarray(magnetic_field_parallel_T, dtype=float).ravel()
        if len(s) < 2 or len({len(s), len(ne), len(bpar)}) != 1:
            raise ValueError("Path, density and parallel-field profiles must have equal lengths of at least two.")
        if not all(np.all(np.isfinite(a)) for a in (s, ne, bpar)):
            raise ValueError("Path, density and field profiles must be finite.")
        if np.any(np.diff(s) <= 0) or np.any(ne < 0):
            raise ValueError("Path must be strictly increasing and electron density nonnegative.")
        line_integral_ne_b = float(np.trapezoid(ne * bpar, s))
        line_integral_ne = float(np.trapezoid(ne, s))
        forward_angle = float(FARADAY_COEFFICIENT_SI * wavelength_m**2 * line_integral_ne_b)
    inferred_integral = None
    inferred_mean_field = None
    if rotation_angle_rad is not None:
        inferred_integral = float(rotation_angle_rad / (FARADAY_COEFFICIENT_SI * wavelength_m**2))
        if all(supplied):
            if line_integral_ne <= 0:
                raise ValueError("A positive line-integrated electron density is required to infer mean field.")
            inferred_mean_field = float(inferred_integral / line_integral_ne)
    return {
        "diagnostic": "Polarimetry / Faraday rotation",
        "wavelength_m": float(wavelength_m),
        "predicted_rotation_angle_rad": forward_angle,
        "observed_rotation_angle_rad": rotation_angle_rad,
        "line_integral_ne_Bparallel_m2_T": line_integral_ne_b,
        "line_integral_ne_m2": line_integral_ne,
        "rotation_implied_integral_ne_Bparallel_m2_T": inferred_integral,
        "density_weighted_mean_Bparallel_T": inferred_mean_field,
        "faraday_coefficient_SI": FARADAY_COEFFICIENT_SI,
        "assumptions_and_limits": [
            "Cold-plasma, high-frequency Faraday-rotation approximation.",
            "The inferred field is density-weighted along the line of sight, not a local field or full magnetic reconstruction.",
            "Density calibration, path geometry, wavelength, sign convention and birefringence effects must be accounted for.",
        ],
    }

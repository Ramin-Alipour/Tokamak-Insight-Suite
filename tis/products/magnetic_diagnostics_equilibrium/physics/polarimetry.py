"""Scoped Faraday-rotation polarimetry calculations."""
from __future__ import annotations

import numpy as np

_E = 1.602176634e-19
_EPS0 = 8.8541878128e-12
_ME = 9.1093837139e-31
_C = 299792458.0
_FARADAY_COEFFICIENT = _E**3 / (8.0 * np.pi**2 * _EPS0 * _ME**2 * _C**3)


def analyze_polarimetry(
    wavelength_m,
    rotation_rad,
    *,
    path_length_m=None,
    electron_density_m3=None,
    rotation_uncertainty_rad=None,
    wavelength_uncertainty_m=None,
    density_uncertainty_m3=None,
    path_length_uncertainty_m=None,
):
    """Infer line-integrated n_e B_parallel from calibrated Faraday rotation.

    Uses rotation = C_F * wavelength**2 * integral(n_e * B_parallel dl).
    Average field assumes the supplied density is uniform along one chord.
    All inputs use SI units; rotation is signed under the caller's convention.
    """
    wavelength = np.asarray(wavelength_m, dtype=float)
    rotation = np.asarray(rotation_rad, dtype=float)
    if (
        wavelength.ndim != 1
        or rotation.ndim != 1
        or wavelength.size == 0
        or wavelength.shape != rotation.shape
    ):
        raise ValueError(
            "wavelength_m and rotation_rad must be non-empty matching 1-D arrays"
        )
    if not np.all(np.isfinite(wavelength)) or np.any(wavelength <= 0):
        raise ValueError("wavelength_m must contain finite positive values")
    if not np.all(np.isfinite(rotation)):
        raise ValueError("rotation_rad must contain only finite values")

    def uncertainty(value, shape, name):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            arr = np.full(shape, float(arr))
        if (
            arr.shape != shape
            or not np.all(np.isfinite(arr))
            or np.any(arr < 0)
        ):
            raise ValueError(
                f"{name} must be finite, non-negative, and scalar or match the input"
            )
        return arr

    ur = uncertainty(
        rotation_uncertainty_rad, rotation.shape, "rotation_uncertainty_rad"
    )
    uw = uncertainty(
        wavelength_uncertainty_m, wavelength.shape, "wavelength_uncertainty_m"
    )
    integral = rotation / (_FARADAY_COEFFICIENT * wavelength**2)
    result = {
        "wavelength_m": wavelength,
        "rotation_rad": rotation,
        "line_integrated_ne_B_parallel_m-2_T": integral,
        "faraday_coefficient_SI": float(_FARADAY_COEFFICIENT),
        "equation": (
            "rotation_rad = C_F * wavelength_m^2 * "
            "integral(n_e[m^-3] * B_parallel[T] dl[m])"
        ),
        "sign_convention": (
            "Signed rotation follows the experiment's handedness and "
            "viewing-coordinate convention."
        ),
        "limitations": [
            "Requires calibrated wavelength and polarization rotation; raw electronics calibration is out of scope.",
            "A single chord yields a line integral, not a spatially resolved magnetic field.",
            "Average field assumes uniform supplied density and known chord length.",
            "Refraction, additional birefringence, instrumental offsets and sign conventions require experiment-specific treatment.",
        ],
    }
    if ur is not None or uw is not None:
        variance = np.zeros(rotation.shape)
        if ur is not None:
            variance += (ur / (_FARADAY_COEFFICIENT * wavelength**2)) ** 2
        if uw is not None:
            variance += (2.0 * integral * uw / wavelength) ** 2
        result["line_integrated_ne_B_parallel_uncertainty_m-2_T"] = np.sqrt(
            variance
        )

    if (electron_density_m3 is None) != (path_length_m is None):
        raise ValueError("electron_density_m3 and path_length_m must be provided together")
    if electron_density_m3 is not None:
        ne = np.asarray(electron_density_m3, dtype=float)
        length = np.asarray(path_length_m, dtype=float)
        if ne.ndim == 0:
            ne = np.full(rotation.shape, float(ne))
        if length.ndim == 0:
            length = np.full(rotation.shape, float(length))
        if ne.shape != rotation.shape or length.shape != rotation.shape:
            raise ValueError(
                "electron_density_m3 and path_length_m must be scalar or match the wavelength/rotation arrays"
            )
        if not np.all(np.isfinite(ne)) or np.any(ne <= 0):
            raise ValueError("electron_density_m3 must be finite and positive")
        if not np.all(np.isfinite(length)) or np.any(length <= 0):
            raise ValueError("path_length_m must be finite and positive")

        field = integral / (ne * length)
        result.update(
            {
                "electron_density_m3": ne,
                "path_length_m": length,
                "average_B_parallel_T": field,
                "average_field_scope": (
                    "Chord-averaged B_parallel assuming supplied electron density "
                    "is uniform along the path."
                ),
            }
        )
        une = uncertainty(
            density_uncertainty_m3, rotation.shape, "density_uncertainty_m3"
        )
        ul = uncertainty(
            path_length_uncertainty_m, rotation.shape, "path_length_uncertainty_m"
        )
        if any(v is not None for v in (une, ul, ur, uw)):
            variance = np.zeros(field.shape)
            if ur is not None:
                variance += (
                    ur / (_FARADAY_COEFFICIENT * wavelength**2) / (ne * length)
                ) ** 2
            if uw is not None:
                variance += (2.0 * field * uw / wavelength) ** 2
            if une is not None:
                variance += (field * une / ne) ** 2
            if ul is not None:
                variance += (field * ul / length) ** 2
            result["average_B_parallel_uncertainty_T"] = np.sqrt(variance)
    elif density_uncertainty_m3 is not None or path_length_uncertainty_m is not None:
        raise ValueError(
            "density_uncertainty_m3 and path_length_uncertainty_m require "
            "electron_density_m3 and path_length_m"
        )
    return result

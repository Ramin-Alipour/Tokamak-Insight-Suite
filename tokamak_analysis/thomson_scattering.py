"""Thomson-scattering profile analysis utilities for Tokamak Insight Suite.

Inputs are calibrated/derived Thomson-scattering measurements (not raw photon
counts). Density is in m^-3, electron temperature in eV, and radius in m.
All functions are deterministic and depend only on NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class ThomsonProfile:
    radius_m: np.ndarray
    density_m3: np.ndarray
    temperature_ev: np.ndarray
    density_uncertainty_m3: np.ndarray | None = None
    temperature_uncertainty_ev: np.ndarray | None = None


@dataclass(frozen=True)
class ThomsonSummary:
    point_count: int
    radius_min_m: float
    radius_max_m: float
    density_peak_m3: float
    density_peak_radius_m: float
    temperature_peak_ev: float
    temperature_peak_radius_m: float
    density_line_average_m3: float
    temperature_line_average_ev: float
    density_volume_average_m3: float | None
    temperature_volume_average_ev: float | None


def _array(values: Iterable[float], name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if arr.size < 2:
        raise ValueError(f"{name} must contain at least two samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr


def build_profile(
    radius_m: Iterable[float],
    density_m3: Iterable[float],
    temperature_ev: Iterable[float],
    density_uncertainty_m3: Iterable[float] | None = None,
    temperature_uncertainty_ev: Iterable[float] | None = None,
) -> ThomsonProfile:
    """Validate and sort profile samples by radius; reject physically invalid values."""
    r = _array(radius_m, "radius_m")
    ne = _array(density_m3, "density_m3")
    te = _array(temperature_ev, "temperature_ev")
    if not (r.size == ne.size == te.size):
        raise ValueError("radius, density, and temperature arrays must have equal lengths")
    if np.any(ne < 0):
        raise ValueError("density values must be non-negative")
    if np.any(te < 0):
        raise ValueError("temperature values must be non-negative")
    order = np.argsort(r)
    r, ne, te = r[order], ne[order], te[order]
    if np.any(np.diff(r) <= 0):
        raise ValueError("radius samples must be unique")
    sne = _optional_uncertainty(density_uncertainty_m3, r.size, "density_uncertainty_m3", order)
    ste = _optional_uncertainty(temperature_uncertainty_ev, r.size, "temperature_uncertainty_ev", order)
    return ThomsonProfile(r, ne, te, sne, ste)


def _optional_uncertainty(values, size: int, name: str, order: np.ndarray):
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size != size:
        raise ValueError(f"{name} must have the same length as the profile")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0):
        raise ValueError(f"{name} must contain finite, non-negative uncertainties")
    return arr[order]


def weighted_mean(values: Iterable[float], uncertainties: Iterable[float]) -> float:
    """Inverse-variance weighted mean; uncertainties must be strictly positive."""
    y = _array(values, "values")
    s = _array(uncertainties, "uncertainties")
    if y.size != s.size:
        raise ValueError("values and uncertainties must have equal lengths")
    if np.any(s <= 0):
        raise ValueError("uncertainties must be strictly positive for inverse-variance weighting")
    w = 1.0 / np.square(s)
    return float(np.sum(w * y) / np.sum(w))


def _trapezoid(y, x):
    """Version-compatible trapezoidal integration for the declared NumPy range."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return integrate(y, x)


def profile_summary(profile: ThomsonProfile, major_radius_m: float | None = None) -> ThomsonSummary:
    """Summarize measured profiles.

    If major_radius_m is supplied, cylindrical toroidal-volume weighting uses
    the large-aspect-ratio approximation dV proportional to R_major * r * dr.
    This is a geometric approximation, not a full equilibrium reconstruction.
    """
    r, ne, te = profile.radius_m, profile.density_m3, profile.temperature_ev
    ne_line = float(_trapezoid(ne, r) / (r[-1] - r[0]))
    te_line = float(np.trapezoid(te, r) / (r[-1] - r[0]))
    ne_vol = te_vol = None
    if major_radius_m is not None:
        if not np.isfinite(major_radius_m) or major_radius_m <= 0:
            raise ValueError("major_radius_m must be positive and finite")
        # Volume weighting requires a non-negative minor-radius coordinate.
        # Clipping negative coordinates would silently bias the inferred average.
        if np.any(r < 0):
            raise ValueError(
                "volume averaging requires a non-negative minor-radius coordinate"
            )
        # For a circular, large-aspect-ratio cross-section, dV/dr ∝ r.
        weight = r
        denom = float(np.trapezoid(weight, r))
        if denom > 0:
            ne_vol = float(np.trapezoid(ne * weight, r) / denom)
            te_vol = float(np.trapezoid(te * weight, r) / denom)
    ine, ite = int(np.argmax(ne)), int(np.argmax(te))
    return ThomsonSummary(
        point_count=int(r.size),
        radius_min_m=float(r[0]),
        radius_max_m=float(r[-1]),
        density_peak_m3=float(ne[ine]),
        density_peak_radius_m=float(r[ine]),
        temperature_peak_ev=float(te[ite]),
        temperature_peak_radius_m=float(r[ite]),
        density_line_average_m3=ne_line,
        temperature_line_average_ev=te_line,
        density_volume_average_m3=ne_vol,
        temperature_volume_average_ev=te_vol,
    )


def interpolate_profile(profile: ThomsonProfile, target_radius_m: Iterable[float]) -> dict[str, np.ndarray]:
    """Linearly interpolate density and temperature inside the measured radial range.

    Extrapolation is deliberately forbidden to avoid silently inventing edge/core data.
    """
    target = np.asarray(target_radius_m, dtype=float)
    if target.ndim != 1 or not np.all(np.isfinite(target)):
        raise ValueError("target_radius_m must be a finite one-dimensional array")
    r = profile.radius_m
    if np.any(target < r[0]) or np.any(target > r[-1]):
        raise ValueError("interpolation targets must lie inside the measured radius range")
    result = {
        "radius_m": target,
        "density_m3": np.interp(target, r, profile.density_m3),
        "temperature_ev": np.interp(target, r, profile.temperature_ev),
    }
    if profile.density_uncertainty_m3 is not None:
        result["density_uncertainty_m3"] = np.interp(target, r, profile.density_uncertainty_m3)
    if profile.temperature_uncertainty_ev is not None:
        result["temperature_uncertainty_ev"] = np.interp(target, r, profile.temperature_uncertainty_ev)
    return result


# Classical Thomson-scattering power relation for an optically thin plasma.
ELECTRON_CLASSICAL_RADIUS_M = 2.8179403262e-15


def thomson_differential_cross_section_m2_sr(scattering_angle_rad):
    """Unpolarized classical differential Thomson cross-section, dσ/dΩ = (r_e²/2)(1+cos²θ)."""
    theta = np.asarray(scattering_angle_rad, dtype=float)
    if not np.all(np.isfinite(theta)) or np.any((theta < 0) | (theta > np.pi)):
        raise ValueError("scattering angle must be finite and in [0, pi]")
    return 0.5 * ELECTRON_CLASSICAL_RADIUS_M**2 * (1.0 + np.cos(theta)**2)


def density_from_thomson_scattered_power(
    scattered_power_W,
    incident_power_W,
    interaction_length_m,
    scattering_angle_rad,
    collection_solid_angle_sr,
    *,
    optical_efficiency=1.0,
):
    """Infer electron density from calibrated collected Thomson-scattered power.

    Uses P_det = P_inc * n_e * L * (dσ/dΩ) * ΔΩ * η.
    Inputs must describe the same calibrated spectral band/time gate; this
    single-band relation neglects spectral-response integration and relativistic
    spectral-shape effects.
    """
    ps, pi, length, theta, omega, eta = np.broadcast_arrays(
        np.asarray(scattered_power_W, dtype=float),
        np.asarray(incident_power_W, dtype=float),
        np.asarray(interaction_length_m, dtype=float),
        np.asarray(scattering_angle_rad, dtype=float),
        np.asarray(collection_solid_angle_sr, dtype=float),
        np.asarray(optical_efficiency, dtype=float),
    )
    if not all(np.all(np.isfinite(x)) for x in (ps, pi, length, theta, omega, eta)):
        raise ValueError("All inputs must be finite")
    if np.any(ps < 0) or np.any(pi <= 0) or np.any(length <= 0) or np.any(omega <= 0):
        raise ValueError("Scattered power must be nonnegative; incident power, length, and solid angle must be positive")
    if np.any((eta <= 0) | (eta > 1)):
        raise ValueError("optical_efficiency must be in (0, 1]")
    cross_section = thomson_differential_cross_section_m2_sr(theta)
    denominator = pi * length * cross_section * omega * eta
    if np.any(denominator <= 0):
        raise ValueError("Scattering geometry gives zero differential cross-section")
    density = ps / denominator
    return {
        "electron_density_m3": density,
        "differential_cross_section_m2_sr": cross_section,
        "model_scattered_power_W": pi * density * length * cross_section * omega * eta,
        "relation": "P_det = P_inc * n_e * L * (dσ/dΩ) * ΔΩ * η",
        "caveat": "Requires calibrated collected power and optical efficiency; a full Thomson spectrum fit is needed for temperature and relativistic spectral effects.",
    }

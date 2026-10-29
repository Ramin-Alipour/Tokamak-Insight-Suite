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
    density_line_average_uncertainty_m3: float | None = None
    temperature_line_average_uncertainty_ev: float | None = None


def _is_finite_real_scalar(value) -> bool:
    """Accept finite real scalar values while rejecting booleans and numeric strings."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _array(values: Iterable[float], name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans or strings")
    arr = raw.astype(float, copy=False)
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
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size != size:
        raise ValueError(f"{name} must have the same length as the profile and be one-dimensional")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans or strings")
    arr = raw.astype(float, copy=False)
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
    # Normalize by the smallest uncertainty before squaring. Direct 1/s**2
    # overflows for valid tiny uncertainties (and can turn the weighted mean
    # into NaN even when the relative weights and answer are well-defined).
    relative = float(np.min(s)) / s
    w = np.square(relative)
    weight_sum = float(np.sum(w))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("uncertainties do not yield a numerically usable weighted mean")
    return float(np.sum(w * y) / weight_sum)


def _trapezoid(y, x):
    """Version-compatible trapezoidal integration for the declared NumPy range."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return integrate(y, x)


def _trapezoid_weights(x: np.ndarray) -> np.ndarray:
    """Return linear weights equivalent to trapezoidal integration over x."""
    weights = np.empty_like(x, dtype=float)
    weights[0] = (x[1] - x[0]) / 2.0
    weights[-1] = (x[-1] - x[-2]) / 2.0
    if x.size > 2:
        weights[1:-1] = (x[2:] - x[:-2]) / 2.0
    return weights


def _weighted_standard_uncertainty(weights, uncertainty) -> float | None:
    if uncertainty is None:
        return None
    return float(np.sqrt(np.sum(np.square(weights * uncertainty))))


def profile_summary(profile: ThomsonProfile) -> ThomsonSummary:
    """Summarize calibrated profiles over the measured radial interval only.

    The summary deliberately excludes geometry-weighted volume averages: those
    require additional geometric assumptions and are outside the core workflow.
    """
    r, ne, te = profile.radius_m, profile.density_m3, profile.temperature_ev
    radial_span = float(r[-1] - r[0])
    trap_weights = _trapezoid_weights(r)
    line_weights = trap_weights / radial_span
    ne_line = float(_trapezoid(ne, r) / radial_span)
    te_line = float(_trapezoid(te, r) / radial_span)
    ne_line_unc = _weighted_standard_uncertainty(line_weights, profile.density_uncertainty_m3)
    te_line_unc = _weighted_standard_uncertainty(line_weights, profile.temperature_uncertainty_ev)

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
        density_line_average_uncertainty_m3=ne_line_unc,
        temperature_line_average_uncertainty_ev=te_line_unc,
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
    scattered_power_uncertainty_W=None,
    incident_power_uncertainty_W=None,
    interaction_length_uncertainty_m=None,
    scattering_angle_uncertainty_rad=None,
    collection_solid_angle_uncertainty_sr=None,
    optical_efficiency_uncertainty=None,
):
    """Infer electron density from calibrated collected Thomson-scattered power.

    Uses P_det = P_inc * n_e * L * (dσ/dΩ) * ΔΩ * η. Inputs must describe the
    same calibrated spectral band and time gate. Optional independent 1-sigma
    input uncertainties are propagated by first-order Gaussian error propagation.
    """
    base = np.broadcast_arrays(
        np.asarray(scattered_power_W, dtype=float),
        np.asarray(incident_power_W, dtype=float),
        np.asarray(interaction_length_m, dtype=float),
        np.asarray(scattering_angle_rad, dtype=float),
        np.asarray(collection_solid_angle_sr, dtype=float),
        np.asarray(optical_efficiency, dtype=float),
    )
    ps, pi, length, theta, omega, eta = base
    if not all(np.all(np.isfinite(x)) for x in base):
        raise ValueError("All inputs must be finite")
    if np.any(ps < 0) or np.any(pi <= 0) or np.any(length <= 0) or np.any(omega <= 0):
        raise ValueError("Scattered power must be nonnegative; incident power, length, and solid angle must be positive")
    if np.any(omega > 4.0 * np.pi):
        raise ValueError("collection_solid_angle_sr cannot exceed 4*pi steradians")
    if np.any((eta <= 0) | (eta > 1)):
        raise ValueError("optical_efficiency must be in (0, 1]")
    cross_section = thomson_differential_cross_section_m2_sr(theta)
    denominator = pi * length * cross_section * omega * eta
    if np.any(denominator <= 0):
        raise ValueError("Scattering geometry gives zero differential cross-section")
    density = ps / denominator

    uncertainty_values = {
        "scattered_power_uncertainty_W": scattered_power_uncertainty_W,
        "incident_power_uncertainty_W": incident_power_uncertainty_W,
        "interaction_length_uncertainty_m": interaction_length_uncertainty_m,
        "scattering_angle_uncertainty_rad": scattering_angle_uncertainty_rad,
        "collection_solid_angle_uncertainty_sr": collection_solid_angle_uncertainty_sr,
        "optical_efficiency_uncertainty": optical_efficiency_uncertainty,
    }
    uncertainty_arrays = {}
    for name, values in uncertainty_values.items():
        if values is None:
            continue
        arr = np.broadcast_to(np.asarray(values, dtype=float), density.shape)
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise ValueError(f"{name} must be finite and non-negative")
        uncertainty_arrays[name] = arr

    result = {
        "electron_density_m3": density,
        "differential_cross_section_m2_sr": cross_section,
        "model_scattered_power_W": pi * density * length * cross_section * omega * eta,
        "relation": "P_det = P_inc * n_e * L * (dσ/dΩ) * ΔΩ * η",
        "caveat": "Requires calibrated collected power and optical efficiency; the single-band relation neglects spectral-response integration and relativistic spectral-shape effects.",
    }
    if uncertainty_arrays:
        variance = np.zeros_like(density, dtype=float)
        if "scattered_power_uncertainty_W" in uncertainty_arrays:
            variance += (uncertainty_arrays["scattered_power_uncertainty_W"] / denominator) ** 2
        if "incident_power_uncertainty_W" in uncertainty_arrays:
            variance += (density * uncertainty_arrays["incident_power_uncertainty_W"] / pi) ** 2
        if "interaction_length_uncertainty_m" in uncertainty_arrays:
            variance += (density * uncertainty_arrays["interaction_length_uncertainty_m"] / length) ** 2
        if "collection_solid_angle_uncertainty_sr" in uncertainty_arrays:
            variance += (density * uncertainty_arrays["collection_solid_angle_uncertainty_sr"] / omega) ** 2
        if "optical_efficiency_uncertainty" in uncertainty_arrays:
            variance += (density * uncertainty_arrays["optical_efficiency_uncertainty"] / eta) ** 2
        if "scattering_angle_uncertainty_rad" in uncertainty_arrays:
            dlog_cross_section_dtheta = -2.0 * np.cos(theta) * np.sin(theta) / (1.0 + np.cos(theta) ** 2)
            variance += (density * dlog_cross_section_dtheta * uncertainty_arrays["scattering_angle_uncertainty_rad"]) ** 2
        result["electron_density_uncertainty_m3"] = np.sqrt(variance)
        result["uncertainty_model"] = "First-order propagation of supplied independent 1-sigma input uncertainties; correlations and model/calibration discrepancy are not included."
    return result


def reconstruct_thomson_profile(
    radius_m,
    density_m3,
    temperature_ev,
    *,
    density_uncertainty_m3=None,
    temperature_uncertainty_ev=None,
    method="smoothing_spline",
    smoothing_factor=1.0,
    points=200,
):
    """Reconstruct smooth radial ne/Te curves from spatially resolved measurements.

    Inputs must already be calibrated. The reconstruction is restricted to the
    measured radial interval (no extrapolation). smoothing_factor controls the
    weighted residual target for smoothing_spline: s = factor * N. If
    uncertainties are provided, they are treated as independent 1-sigma errors
    and used as inverse-uncertainty weights. Without uncertainties, each profile
    is scaled before fitting and the residual target is relative to that scale.

    Returns sampled reconstructed profiles and first derivatives. This is a
    one-dimensional profile fit, not an equilibrium or transport reconstruction;
    it does not estimate fit confidence intervals or systematic uncertainties.
    """
    from scipy.interpolate import PchipInterpolator, UnivariateSpline

    profile = build_profile(
        radius_m,
        density_m3,
        temperature_ev,
        density_uncertainty_m3,
        temperature_uncertainty_ev,
    )
    if method not in {"smoothing_spline", "pchip"}:
        raise ValueError("method must be 'smoothing_spline' or 'pchip'")
    if not _is_finite_real_scalar(smoothing_factor) or smoothing_factor < 0:
        raise ValueError("smoothing_factor must be finite and non-negative")
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or not 20 <= points <= 5000:
        raise ValueError("points must be an integer between 20 and 5000")

    r = profile.radius_m
    grid = np.linspace(float(r[0]), float(r[-1]), int(points))

    def fit_channel(y, uncertainty, label):
        if method == "pchip":
            spline = PchipInterpolator(r, y, extrapolate=False)
            return np.asarray(spline(grid), dtype=float), np.asarray(spline.derivative()(grid), dtype=float)

        if uncertainty is not None:
            if np.any(uncertainty <= 0):
                raise ValueError(f"{label} uncertainties must be strictly positive for weighted spline fitting")
            weights = 1.0 / uncertainty
            fitted = UnivariateSpline(r, y, w=weights, s=float(smoothing_factor) * r.size, k=min(3, r.size - 1))
            return np.asarray(fitted(grid), dtype=float), np.asarray(fitted.derivative()(grid), dtype=float)

        scale = float(np.max(np.abs(y)))
        if scale == 0:
            return np.zeros_like(grid), np.zeros_like(grid)
        normalized = y / scale
        fitted = UnivariateSpline(
            r, normalized, s=float(smoothing_factor) * r.size, k=min(3, r.size - 1)
        )
        return (
            np.asarray(fitted(grid), dtype=float) * scale,
            np.asarray(fitted.derivative()(grid), dtype=float) * scale,
        )

    density_fit, density_gradient = fit_channel(
        profile.density_m3, profile.density_uncertainty_m3, "density"
    )
    temperature_fit, temperature_gradient = fit_channel(
        profile.temperature_ev, profile.temperature_uncertainty_ev, "temperature"
    )

    # Unconstrained cubic smoothing splines can overshoot below zero even when
    # every measured value is physical. Project the reconstructed profiles onto
    # the physical non-negative domain and make the derivative consistent with
    # the projected (flat) segment. PCHIP is shape-preserving, but this guard
    # also protects the smoothing-spline path from returning negative ne or Te.
    density_negative = density_fit < 0.0
    temperature_negative = temperature_fit < 0.0
    density_fit = np.maximum(density_fit, 0.0)
    temperature_fit = np.maximum(temperature_fit, 0.0)
    density_gradient = np.where(density_negative, 0.0, density_gradient)
    temperature_gradient = np.where(temperature_negative, 0.0, temperature_gradient)

    return {
        "radius_m": grid,
        "density_m3": density_fit,
        "temperature_eV": temperature_fit,
        "density_gradient_m-4": density_gradient,
        "temperature_gradient_eV_m-1": temperature_gradient,
        "method": method,
        "smoothing_factor": float(smoothing_factor) if method == "smoothing_spline" else None,
        "input_point_count": int(r.size),
        "reconstruction_point_count": int(grid.size),
        "radial_domain_m": (float(r[0]), float(r[-1])),
        "uncertainty_model": (
            "Supplied pointwise uncertainties used as independent inverse-sigma fit weights; "
            "fit confidence intervals and systematic/model uncertainties are not estimated."
            if profile.density_uncertainty_m3 is not None or profile.temperature_uncertainty_ev is not None
            else "No measurement uncertainties supplied; fit confidence intervals are not estimated."
        ),
        "limitations": [
            "Only calibrated spatial measurements are accepted; no detector or wavelength calibration is performed.",
            "No extrapolation beyond the measured radial interval is performed.",
            "This is a one-dimensional radial profile fit, not an equilibrium reconstruction or a transport inversion.",
            "Negative spline overshoot is projected to zero; derivatives are set to zero on clipped segments. This enforces non-negativity but is not a substitute for a physically constrained fit.",
            "Smoothing-spline fitting requires strictly positive uncertainties when uncertainty arrays are supplied; PCHIP accepts non-negative uncertainties because it does not use them as fit weights.",
        ],
    }

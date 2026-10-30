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
    density_min_m3: float
    density_min_radius_m: float
    density_peak_m3: float
    density_peak_radius_m: float
    temperature_min_ev: float
    temperature_min_radius_m: float
    temperature_peak_ev: float
    temperature_peak_radius_m: float
    density_radial_average_m3: float
    temperature_radial_average_ev: float
    density_radial_average_uncertainty_m3: float | None = None
    temperature_radial_average_uncertainty_ev: float | None = None

    @property
    def density_line_average_m3(self):
        """Deprecated alias; this is a radial-coordinate average, not a line average."""
        return self.density_radial_average_m3

    @property
    def temperature_line_average_ev(self):
        """Deprecated alias; this is a radial-coordinate average, not a line average."""
        return self.temperature_radial_average_ev

    @property
    def density_line_average_uncertainty_m3(self):
        return self.density_radial_average_uncertainty_m3

    @property
    def temperature_line_average_uncertainty_ev(self):
        return self.temperature_radial_average_uncertainty_ev


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


def _covariance_is_positive_semidefinite(covariance, *, rtol=1e-10):
    """Check PSD after scaling to a dimensionless correlation matrix.

    Raw covariance eigenvalues are unsafe when a joint matrix mixes quantities
    with very different units, such as density in m^-3 and temperature in eV.
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not np.all(np.isfinite(matrix)):
        return False
    diagonal = np.diag(matrix)
    if np.any(diagonal < 0.0):
        return False
    standard = np.sqrt(diagonal)
    positive = np.flatnonzero(standard > 0.0)
    zero = np.flatnonzero(standard == 0.0)
    if zero.size and np.any(matrix[zero, :] != 0.0):
        return False
    correlation = np.zeros_like(matrix)
    if positive.size:
        scale = np.outer(standard[positive], standard[positive])
        correlation[np.ix_(positive, positive)] = matrix[np.ix_(positive, positive)] / scale
    eigenvalues = np.linalg.eigvalsh(correlation)
    return bool(float(np.min(eigenvalues)) >= -float(rtol))


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
    radial_weights = trap_weights / radial_span
    ne_radial = float(_trapezoid(ne, r) / radial_span)
    te_radial = float(_trapezoid(te, r) / radial_span)
    ne_radial_unc = _weighted_standard_uncertainty(radial_weights, profile.density_uncertainty_m3)
    te_radial_unc = _weighted_standard_uncertainty(radial_weights, profile.temperature_uncertainty_ev)

    ine_min, ine_max = int(np.argmin(ne)), int(np.argmax(ne))
    ite_min, ite_max = int(np.argmin(te)), int(np.argmax(te))
    return ThomsonSummary(
        point_count=int(r.size),
        radius_min_m=float(r[0]),
        radius_max_m=float(r[-1]),
        density_min_m3=float(ne[ine_min]),
        density_min_radius_m=float(r[ine_min]),
        density_peak_m3=float(ne[ine_max]),
        density_peak_radius_m=float(r[ine_max]),
        temperature_min_ev=float(te[ite_min]),
        temperature_min_radius_m=float(r[ite_min]),
        temperature_peak_ev=float(te[ite_max]),
        temperature_peak_radius_m=float(r[ite_max]),
        density_radial_average_m3=ne_radial,
        temperature_radial_average_ev=te_radial,
        density_radial_average_uncertainty_m3=ne_radial_unc,
        temperature_radial_average_uncertainty_ev=te_radial_unc,
    )

def _interpolate_independent_uncertainty(target, radius, uncertainty):
    """Propagate independent endpoint uncertainties through linear interpolation."""
    upper = np.searchsorted(radius, target, side="right")
    upper = np.clip(upper, 1, radius.size - 1)
    lower = upper - 1
    fraction = (target - radius[lower]) / (radius[upper] - radius[lower])
    variance = (
        np.square((1.0 - fraction) * uncertainty[lower])
        + np.square(fraction * uncertainty[upper])
    )
    return np.sqrt(variance)


def interpolate_profile(profile: ThomsonProfile, target_radius_m: Iterable[float]) -> dict[str, np.ndarray]:
    """Linearly interpolate profiles and propagate independent point uncertainties.

    Profile values are linearly interpolated. Their independent 1-sigma endpoint
    uncertainties are propagated using the squared interpolation coefficients.
    Extrapolation is deliberately forbidden to avoid inventing edge/core data.
    """
    raw_target = np.asarray(target_radius_m)
    if raw_target.ndim != 1:
        raise ValueError("target_radius_m must be a one-dimensional array")
    if raw_target.dtype.kind not in "iuf":
        raise ValueError("target_radius_m must contain real numeric values, not booleans or strings")
    if raw_target.size == 0:
        raise ValueError("target_radius_m must contain at least one interpolation point")
    target = raw_target.astype(float, copy=False)
    if not np.all(np.isfinite(target)):
        raise ValueError("target_radius_m must contain only finite values")
    r = profile.radius_m
    if np.any(target < r[0]) or np.any(target > r[-1]):
        raise ValueError("interpolation targets must lie inside the measured radius range")
    result = {
        "radius_m": target,
        "density_m3": np.interp(target, r, profile.density_m3),
        "temperature_ev": np.interp(target, r, profile.temperature_ev),
    }
    if profile.density_uncertainty_m3 is not None:
        result["density_uncertainty_m3"] = _interpolate_independent_uncertainty(
            target, r, profile.density_uncertainty_m3
        )
    if profile.temperature_uncertainty_ev is not None:
        result["temperature_uncertainty_ev"] = _interpolate_independent_uncertainty(
            target, r, profile.temperature_uncertainty_ev
        )
    return result


# Classical Thomson-scattering power relation for an optically thin plasma.
def reconstruct_thomson_profile(
    radius_m,
    density_m3,
    temperature_ev,
    *,
    density_uncertainty_m3=None,
    temperature_uncertainty_ev=None,
    method="smoothing_spline",
    smoothing_factor=None,
    points=200,
    density_covariance_m3_2=None,
    temperature_covariance_eV2=None,
    joint_density_temperature_covariance=None,
    uncertainty_mc_samples=200,
):
    """Reconstruct smooth radial ne/Te curves from spatially resolved measurements.

    Inputs must already be calibrated. The reconstruction is restricted to the
    measured radial interval (no extrapolation). smoothing_factor controls the
    weighted residual target for smoothing_spline: s = factor * N. If
    Pointwise uncertainties are used as inverse-sigma fit weights for smoothing
    splines. Optional full density/temperature covariance matrices preserve
    correlated input errors. Parametric Monte Carlo propagates the supplied
    covariance through the selected reconstruction and returns pointwise
    standard uncertainties and 95% bands when enough non-negative draws succeed.

    Returns sampled reconstructed profiles and first derivatives. This is a
    one-dimensional downstream profile fit, not an equilibrium or transport
    reconstruction. Calibration, systematic uncertainty and model discrepancy
    remain outside this function.
    """
    from scipy.interpolate import PchipInterpolator, UnivariateSpline, make_smoothing_spline

    radius_original = _array(radius_m, "radius_m")
    order = np.argsort(radius_original)

    def validate_covariance(value, name):
        if value is None:
            return None
        covariance = np.asarray(value, dtype=float)
        if covariance.shape != (radius_original.size, radius_original.size) or not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} must be a finite square covariance matrix matching the profile length")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError(f"{name} must be symmetric")
        if not _covariance_is_positive_semidefinite(covariance):
            raise ValueError(f"{name} must be positive semidefinite")
        return covariance

    density_covariance_input = validate_covariance(density_covariance_m3_2, "density_covariance_m3_2")
    temperature_covariance_input = validate_covariance(temperature_covariance_eV2, "temperature_covariance_eV2")
    joint_covariance_input = None
    if joint_density_temperature_covariance is not None:
        joint_covariance_raw = np.asarray(joint_density_temperature_covariance)
        if joint_covariance_raw.dtype.kind not in "iuf":
            raise ValueError("joint_density_temperature_covariance must contain real numeric values, not booleans or strings")
        joint_covariance_input = joint_covariance_raw.astype(float, copy=False)
        expected_joint_shape = (2 * radius_original.size, 2 * radius_original.size)
        if joint_covariance_input.shape != expected_joint_shape or not np.all(np.isfinite(joint_covariance_input)):
            raise ValueError("joint_density_temperature_covariance must be a finite (2N x 2N) matrix ordered [density_0..N-1, temperature_0..N-1]")
        if not np.allclose(joint_covariance_input, joint_covariance_input.T, rtol=1e-10, atol=0.0):
            raise ValueError("joint_density_temperature_covariance must be symmetric")
        if not _covariance_is_positive_semidefinite(joint_covariance_input):
            raise ValueError("joint_density_temperature_covariance must be positive semidefinite")
        n_profile = radius_original.size
        joint_density_block = joint_covariance_input[:n_profile, :n_profile]
        joint_temperature_block = joint_covariance_input[n_profile:, n_profile:]
        if density_covariance_input is not None and not np.allclose(density_covariance_input, joint_density_block, rtol=1e-6, atol=1e-24):
            raise ValueError("density_covariance_m3_2 must match the density block of joint_density_temperature_covariance")
        if temperature_covariance_input is not None and not np.allclose(temperature_covariance_input, joint_temperature_block, rtol=1e-6, atol=1e-24):
            raise ValueError("temperature_covariance_eV2 must match the temperature block of joint_density_temperature_covariance")
        density_covariance_input = joint_density_block
        temperature_covariance_input = joint_temperature_block
    if density_covariance_input is not None:
        if density_uncertainty_m3 is None:
            density_uncertainty_m3 = np.sqrt(np.maximum(np.diag(density_covariance_input), 0.0))
        elif not np.allclose(np.square(np.asarray(density_uncertainty_m3, dtype=float)),
                             np.diag(density_covariance_input), rtol=1e-6, atol=1e-24):
            raise ValueError("density_uncertainty_m3 must match the diagonal of density_covariance_m3_2")
    if temperature_covariance_input is not None:
        if temperature_uncertainty_ev is None:
            temperature_uncertainty_ev = np.sqrt(np.maximum(np.diag(temperature_covariance_input), 0.0))
        elif not np.allclose(np.square(np.asarray(temperature_uncertainty_ev, dtype=float)),
                             np.diag(temperature_covariance_input), rtol=1e-6, atol=1e-24):
            raise ValueError("temperature_uncertainty_ev must match the diagonal of temperature_covariance_eV2")

    density_covariance = (density_covariance_input[np.ix_(order, order)] if density_covariance_input is not None
                          else np.diag(np.square(np.asarray(density_uncertainty_m3, dtype=float)[order])) if density_uncertainty_m3 is not None else None)
    temperature_covariance = (temperature_covariance_input[np.ix_(order, order)] if temperature_covariance_input is not None
                               else np.diag(np.square(np.asarray(temperature_uncertainty_ev, dtype=float)[order])) if temperature_uncertainty_ev is not None else None)
    joint_covariance = None
    if joint_covariance_input is not None:
        n_profile = radius_original.size
        joint_order = np.r_[order, n_profile + order]
        joint_covariance = joint_covariance_input[np.ix_(joint_order, joint_order)]

    profile = build_profile(
        radius_original,
        density_m3,
        temperature_ev,
        density_uncertainty_m3,
        temperature_uncertainty_ev,
    )
    if isinstance(uncertainty_mc_samples, (bool, np.bool_)) or not isinstance(uncertainty_mc_samples, (int, np.integer)) or uncertainty_mc_samples < 20:
        raise ValueError("uncertainty_mc_samples must be an integer >= 20")
    if method not in {"smoothing_spline", "pchip"}:
        raise ValueError("method must be 'smoothing_spline' or 'pchip'")
    if smoothing_factor is not None and (not _is_finite_real_scalar(smoothing_factor) or smoothing_factor < 0):
        raise ValueError("smoothing_factor must be None (automatic GCV selection) or finite and non-negative")
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
            if smoothing_factor is None:
                if r.size < 5:
                    raise ValueError("Automatic GCV smoothing requires at least five radial samples; use PCHIP or supply smoothing_factor.")
                fitted = make_smoothing_spline(r, y, w=weights, lam=None)
            else:
                fitted = UnivariateSpline(r, y, w=weights, s=float(smoothing_factor) * r.size, k=min(3, r.size - 1))
            derivative = fitted.derivative() if hasattr(fitted, "derivative") else fitted.derivative
            return np.asarray(fitted(grid), dtype=float), np.asarray(derivative(grid), dtype=float)

        scale = float(np.max(np.abs(y)))
        if scale == 0:
            return np.zeros_like(grid), np.zeros_like(grid)
        normalized = y / scale
        if smoothing_factor is None:
            if r.size < 5:
                raise ValueError("Automatic GCV smoothing requires at least five radial samples; use PCHIP or supply smoothing_factor.")
            fitted = make_smoothing_spline(r, normalized, lam=None)
        else:
            fitted = UnivariateSpline(r, normalized, s=float(smoothing_factor) * r.size, k=min(3, r.size - 1))
        derivative = fitted.derivative() if hasattr(fitted, "derivative") else fitted.derivative
        return (
            np.asarray(fitted(grid), dtype=float) * scale,
            np.asarray(derivative(grid), dtype=float) * scale,
        )

    density_fit, density_gradient = fit_channel(
        profile.density_m3, profile.density_uncertainty_m3, "density"
    )
    temperature_fit, temperature_gradient = fit_channel(
        profile.temperature_ev, profile.temperature_uncertainty_ev, "temperature"
    )

    # Never clip an unconstrained spline: clipping changes both the fitted
    # profile and its derivative. Reject a non-physical reconstruction explicitly.
    if np.any(density_fit < 0.0) or np.any(temperature_fit < 0.0):
        raise ValueError("Reconstruction produced negative density or temperature overshoot; no clipping is applied. Select PCHIP or change the explicitly supplied smoothing_factor.")

    rng = np.random.default_rng(91473)
    minimum_draws = max(20, int(np.ceil(0.8 * uncertainty_mc_samples)))

    def profile_monte_carlo(values, covariance, uncertainty, label):
        if covariance is None:
            return None
        samples = []
        for _ in range(int(uncertainty_mc_samples)):
            try:
                draw = rng.multivariate_normal(values, covariance, check_valid="raise")
                if np.any(draw < 0):
                    continue
                reconstructed, _ = fit_channel(draw, uncertainty, label)
                if not np.all(np.isfinite(reconstructed)) or np.any(reconstructed < 0):
                    continue
                samples.append(reconstructed)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        if len(samples) < minimum_draws:
            return {"successful_draws": len(samples), "uncertainty_available": False}
        matrix = np.asarray(samples, dtype=float)
        return {
            "successful_draws": len(samples),
            "uncertainty_available": True,
            "standard_uncertainty": np.std(matrix, axis=0, ddof=1),
            "lower_95": np.quantile(matrix, 0.025, axis=0),
            "upper_95": np.quantile(matrix, 0.975, axis=0),
        }

    joint_mc = None
    if joint_covariance is not None:
        joint_samples = []
        joint_values = np.r_[profile.density_m3, profile.temperature_ev]
        for _ in range(int(uncertainty_mc_samples)):
            try:
                draw = rng.multivariate_normal(joint_values, joint_covariance, check_valid="raise")
                density_draw, temperature_draw = draw[:r.size], draw[r.size:]
                if np.any(density_draw < 0) or np.any(temperature_draw < 0):
                    continue
                density_reconstructed, _ = fit_channel(density_draw, profile.density_uncertainty_m3, "density")
                temperature_reconstructed, _ = fit_channel(temperature_draw, profile.temperature_uncertainty_ev, "temperature")
                if (not np.all(np.isfinite(density_reconstructed)) or
                        not np.all(np.isfinite(temperature_reconstructed)) or
                        np.any(density_reconstructed < 0) or np.any(temperature_reconstructed < 0)):
                    continue
                joint_samples.append(np.r_[density_reconstructed, temperature_reconstructed])
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                continue
        if len(joint_samples) >= minimum_draws:
            joint_matrix = np.asarray(joint_samples, dtype=float)
            joint_reconstructed_covariance = np.cov(joint_matrix, rowvar=False, ddof=1)
            density_matrix = joint_matrix[:, :grid.size]
            temperature_matrix = joint_matrix[:, grid.size:]
            joint_mc = {
                "successful_draws": len(joint_samples),
                "uncertainty_available": True,
                "density_standard_uncertainty": np.std(density_matrix, axis=0, ddof=1),
                "density_lower_95": np.quantile(density_matrix, 0.025, axis=0),
                "density_upper_95": np.quantile(density_matrix, 0.975, axis=0),
                "temperature_standard_uncertainty": np.std(temperature_matrix, axis=0, ddof=1),
                "temperature_lower_95": np.quantile(temperature_matrix, 0.025, axis=0),
                "temperature_upper_95": np.quantile(temperature_matrix, 0.975, axis=0),
                "reconstructed_joint_covariance": joint_reconstructed_covariance,
                "reconstructed_density_temperature_cross_covariance": joint_reconstructed_covariance[:grid.size, grid.size:],
            }
            density_mc = {"successful_draws": len(joint_samples), "uncertainty_available": True,
                          "standard_uncertainty": joint_mc["density_standard_uncertainty"],
                          "lower_95": joint_mc["density_lower_95"], "upper_95": joint_mc["density_upper_95"]}
            temperature_mc = {"successful_draws": len(joint_samples), "uncertainty_available": True,
                              "standard_uncertainty": joint_mc["temperature_standard_uncertainty"],
                              "lower_95": joint_mc["temperature_lower_95"], "upper_95": joint_mc["temperature_upper_95"]}
        else:
            joint_mc = {"successful_draws": len(joint_samples), "uncertainty_available": False}
            density_mc = {"successful_draws": len(joint_samples), "uncertainty_available": False}
            temperature_mc = {"successful_draws": len(joint_samples), "uncertainty_available": False}
    else:
        density_mc = profile_monte_carlo(profile.density_m3, density_covariance,
                                         profile.density_uncertainty_m3, "density")
        temperature_mc = profile_monte_carlo(profile.temperature_ev, temperature_covariance,
                                             profile.temperature_uncertainty_ev, "temperature")

    uncertainty_channels = {
        "density": (
            "inverse_sigma_weighted" if method == "smoothing_spline"
            and profile.density_uncertainty_m3 is not None else
            "not_used_by_fit" if profile.density_uncertainty_m3 is not None else
            "not_supplied"
        ),
        "temperature": (
            "inverse_sigma_weighted" if method == "smoothing_spline"
            and profile.temperature_uncertainty_ev is not None else
            "not_used_by_fit" if profile.temperature_uncertainty_ev is not None else
            "not_supplied"
        ),
    }
    gradients = {
        "density_gradient_m-4": density_gradient,
        "temperature_gradient_eV_m-1": temperature_gradient,
    }
    return {
        # Canonical names follow DIAGNOSTIC_REGISTRY.json.
        "reconstructed_radius_m": grid,
        "reconstructed_density_m3": density_fit,
        "reconstructed_temperature_eV": temperature_fit,
        "reconstruction_method": method,
        "reconstruction_gradients": gradients,
        "reconstructed_density_uncertainty_m3": None if density_mc is None or not density_mc.get("uncertainty_available") else density_mc["standard_uncertainty"],
        "reconstructed_temperature_uncertainty_eV": None if temperature_mc is None or not temperature_mc.get("uncertainty_available") else temperature_mc["standard_uncertainty"],
        "reconstructed_density_lower_95_m3": None if density_mc is None or not density_mc.get("uncertainty_available") else density_mc["lower_95"],
        "reconstructed_density_upper_95_m3": None if density_mc is None or not density_mc.get("uncertainty_available") else density_mc["upper_95"],
        "reconstructed_temperature_lower_95_eV": None if temperature_mc is None or not temperature_mc.get("uncertainty_available") else temperature_mc["lower_95"],
        "reconstructed_temperature_upper_95_eV": None if temperature_mc is None or not temperature_mc.get("uncertainty_available") else temperature_mc["upper_95"],
        "reconstruction_domain_m": (float(r[0]), float(r[-1])),
        # Legacy aliases are retained to avoid breaking existing callers.
        "radius_m": grid,
        "density_m3": density_fit,
        "temperature_eV": temperature_fit,
        "density_gradient_m-4": density_gradient,
        "temperature_gradient_eV_m-1": temperature_gradient,
        "method": method,
        "smoothing_factor": float(smoothing_factor) if method == "smoothing_spline" and smoothing_factor is not None else None,
        "smoothing_selection": ("generalized_cross_validation" if method == "smoothing_spline" and smoothing_factor is None else "user_supplied_residual_target" if method == "smoothing_spline" else "shape_preserving_pchip"),
        "input_point_count": int(r.size),
        "reconstruction_point_count": int(grid.size),
        "radial_domain_m": (float(r[0]), float(r[-1])),
        "reconstructed_joint_density_temperature_covariance": (
            None if joint_mc is None or not joint_mc.get("uncertainty_available")
            else joint_mc["reconstructed_joint_covariance"]
        ),
        "reconstructed_density_temperature_cross_covariance": (
            None if joint_mc is None or not joint_mc.get("uncertainty_available")
            else joint_mc["reconstructed_density_temperature_cross_covariance"]
        ),
        "uncertainty_model": {
            "assumption": ("Full joint density-temperature covariance is sampled together." if joint_covariance is not None
                           else "Supplied pointwise uncertainties are independent 1-sigma values unless a channel-specific covariance is supplied."),
            "density_channel": uncertainty_channels["density"],
            "temperature_channel": uncertainty_channels["temperature"],
            "fit_confidence_intervals_estimated": bool((density_mc and density_mc.get("uncertainty_available")) or (temperature_mc and temperature_mc.get("uncertainty_available"))),
            "reconstructed_profile_uncertainty_available": bool((density_mc and density_mc.get("uncertainty_available")) or (temperature_mc and temperature_mc.get("uncertainty_available"))),
            "density_covariance_model": "full_correlated_covariance" if density_covariance_input is not None else "diagonal_independent" if density_covariance is not None else "not_supplied",
            "temperature_covariance_model": "full_correlated_covariance" if temperature_covariance_input is not None else "diagonal_independent" if temperature_covariance is not None else "not_supplied",
            "joint_density_temperature_covariance_model": "full_joint_correlated_covariance" if joint_covariance_input is not None else "not_supplied",
            "density_mc_successful_draws": 0 if density_mc is None else density_mc["successful_draws"],
            "temperature_mc_successful_draws": 0 if temperature_mc is None else temperature_mc["successful_draws"],
            "joint_density_temperature_mc_successful_draws": 0 if joint_mc is None else joint_mc["successful_draws"],
            "fit_weights_are_not_output_uncertainties": True,
            "systematic_and_model_uncertainties_estimated": False,
        },
        "limitations": [
            "Only calibrated spatial measurements are accepted; no detector or wavelength calibration is performed.",
            "No extrapolation beyond the measured radial interval is performed.",
            "This is a one-dimensional radial profile fit, not an equilibrium reconstruction or a transport inversion.",
            "Negative spline overshoot is rejected rather than clipped; clipping would alter the fitted profile and its derivative. PCHIP is available when shape preservation is preferred.",
            "Smoothing-spline fitting requires strictly positive uncertainties when uncertainty arrays are supplied; PCHIP accepts non-negative uncertainties because it does not use them as fit weights.",
            "When smoothing_factor is omitted, the smoothing spline selects its regularization parameter by generalized cross-validation. Supplied uncertainties determine fit weights; output uncertainty bands are estimated separately by Monte Carlo from the supplied pointwise, channel covariance or full joint density-temperature covariance model.",
        ],
    }


# Classical Thomson-scattering differential cross-section and calibrated-power density inversion.
_ELECTRON_CLASSICAL_RADIUS_M = 2.8179403262e-15


def _positive_finite_scalar(value, name: str, *, allow_zero: bool = False) -> float:
    """Validate a finite real scalar, rejecting booleans and numeric strings."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")
    return result


def thomson_differential_cross_section_m2_sr(scattering_angle_rad: float, *, incident_polarization_vector=None, incident_direction_vector=None, scattered_direction_vector=None) -> float:
    """Thomson differential cross-section in m^2 sr^-1.

    Without vectors, use the unpolarized result. With both vectors, calculate
    the polarization-dependent result summed over unanalysed outgoing states.
    """
    theta = _positive_finite_scalar(scattering_angle_rad, "scattering_angle_rad", allow_zero=True)
    if theta > np.pi:
        raise ValueError("scattering_angle_rad must lie between 0 and pi")
    if incident_polarization_vector is None and incident_direction_vector is None and scattered_direction_vector is None:
        return float(0.5 * _ELECTRON_CLASSICAL_RADIUS_M**2 * (1.0 + np.cos(theta)**2))
    if any(v is None for v in (incident_polarization_vector, incident_direction_vector, scattered_direction_vector)):
        raise ValueError("incident polarization, incident direction and scattered direction vectors must be supplied together")
    eps = np.asarray(incident_polarization_vector, dtype=float)
    incident = np.asarray(incident_direction_vector, dtype=float)
    direction = np.asarray(scattered_direction_vector, dtype=float)
    if any(v.shape != (3,) or not np.all(np.isfinite(v)) for v in (eps,incident,direction)):
        raise ValueError("polarization and beam-direction inputs must be finite 3-vectors")
    eps_norm = float(np.linalg.norm(eps)); incident_norm=float(np.linalg.norm(incident)); direction_norm = float(np.linalg.norm(direction))
    if eps_norm <= 0 or incident_norm <= 0 or direction_norm <= 0:
        raise ValueError("polarization and beam-direction vectors must be nonzero")
    eps = eps/eps_norm; incident=incident/incident_norm; direction = direction/direction_norm
    if abs(float(np.dot(eps,incident))) > 1e-6:
        raise ValueError("incident polarization must be perpendicular to the incident beam direction")
    geometric_angle=float(np.arccos(np.clip(np.dot(incident,direction),-1.0,1.0)))
    if abs(geometric_angle-theta)>1e-5:
        raise ValueError("scattering_angle_rad is inconsistent with the supplied beam-direction vectors")
    factor = max(0.0, 1.0-float(np.dot(direction, eps))**2)
    return float(_ELECTRON_CLASSICAL_RADIUS_M**2*factor)


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
    incident_polarization_vector=None,
    incident_direction_vector=None,
    scattered_direction_vector=None,
    polarization_factor_uncertainty=None,
    relative_input_covariance=None,
):
    """Infer density from calibrated collected Thomson power.

    P_s = n_e P_i L (d sigma/d Omega) DeltaOmega eta. Optional independent
    1-sigma uncertainties are propagated with first-order derivatives.
    """
    ps = _positive_finite_scalar(scattered_power_W, "scattered_power_W", allow_zero=True)
    pi = _positive_finite_scalar(incident_power_W, "incident_power_W")
    length = _positive_finite_scalar(interaction_length_m, "interaction_length_m")
    theta = _positive_finite_scalar(scattering_angle_rad, "scattering_angle_rad", allow_zero=True)
    if theta > np.pi:
        raise ValueError("scattering_angle_rad must lie between 0 and pi")
    omega = _positive_finite_scalar(collection_solid_angle_sr, "collection_solid_angle_sr")
    eta = _positive_finite_scalar(optical_efficiency, "optical_efficiency")
    if eta > 1.0:
        raise ValueError("optical_efficiency must not exceed 1")
    dsigma = thomson_differential_cross_section_m2_sr(
        theta, incident_polarization_vector=incident_polarization_vector,
        incident_direction_vector=incident_direction_vector,
        scattered_direction_vector=scattered_direction_vector,
    )
    if dsigma <= 0:
        raise ValueError("The supplied polarization/collection geometry has zero Thomson differential cross-section")
    density = ps / (pi * length * dsigma * omega * eta)
    model_power = density * pi * length * dsigma * omega * eta

    rel_variance = 0.0
    uncertainty_supplied = False
    if relative_input_covariance is not None:
        covariance = np.asarray(relative_input_covariance, dtype=float)
        if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
            raise ValueError("relative_input_covariance must be a finite 6x6 matrix")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError("relative_input_covariance must be symmetric")
        if not _covariance_is_positive_semidefinite(covariance):
            raise ValueError("relative_input_covariance must be positive semidefinite")
        if any(v is not None for v in (scattered_power_uncertainty_W, incident_power_uncertainty_W,
                interaction_length_uncertainty_m, collection_solid_angle_uncertainty_sr,
                optical_efficiency_uncertainty, polarization_factor_uncertainty)):
            raise ValueError("Use relative_input_covariance instead of separate scalar uncertainties; do not double-count uncertainty inputs")
        gradient = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        rel_variance = float(gradient @ covariance @ gradient)
        if rel_variance < -1e-12:
            raise ValueError("relative_input_covariance produced a negative propagated variance")
        rel_variance = max(0.0, rel_variance)
        uncertainty_supplied = True
    inputs = (
        ("scattered_power_uncertainty_W", scattered_power_uncertainty_W, ps),
        ("incident_power_uncertainty_W", incident_power_uncertainty_W, pi),
        ("interaction_length_uncertainty_m", interaction_length_uncertainty_m, length),
        ("collection_solid_angle_uncertainty_sr", collection_solid_angle_uncertainty_sr, omega),
        ("optical_efficiency_uncertainty", optical_efficiency_uncertainty, eta),
    )
    for name, uncertainty, nominal in inputs:
        if uncertainty is None:
            continue
        sigma = _positive_finite_scalar(uncertainty, name, allow_zero=True)
        uncertainty_supplied = True
        if nominal == 0.0:
            if sigma > 0.0:
                raise ValueError(f"{name} cannot be propagated relative to a zero nominal value")
        else:
            rel_variance += (sigma / nominal) ** 2
    if polarization_factor_uncertainty is not None:
        sigma_pol = _positive_finite_scalar(polarization_factor_uncertainty, "polarization_factor_uncertainty", allow_zero=True)
        if incident_polarization_vector is None or dsigma <= 0:
            raise ValueError("polarization-factor uncertainty requires a nonzero polarization-dependent cross-section")
        polarization_factor = dsigma/(_ELECTRON_CLASSICAL_RADIUS_M**2)
        if polarization_factor <= 0:
            raise ValueError("polarization factor must be positive to propagate its uncertainty")
        uncertainty_supplied = True
        rel_variance += (sigma_pol/polarization_factor)**2
    if scattering_angle_uncertainty_rad is not None:
        sigma_theta = _positive_finite_scalar(
            scattering_angle_uncertainty_rad, "scattering_angle_uncertainty_rad", allow_zero=True
        )
        uncertainty_supplied = True
        if incident_polarization_vector is not None and sigma_theta > 0:
            raise ValueError("For polarized geometry, propagate angle uncertainty through the supplied direction vectors or provide polarization_factor_uncertainty; the scalar unpolarized angle derivative is not applicable.")
        if incident_polarization_vector is None:
            relative_sensitivity = abs(
                2.0 * np.sin(theta) * np.cos(theta) / (1.0 + np.cos(theta) ** 2)
            )
            rel_variance += (relative_sensitivity * sigma_theta) ** 2

    return {
        "electron_density_m3": float(density),
        "electron_density_uncertainty_m3": float(density * np.sqrt(rel_variance)) if uncertainty_supplied else None,
        "model_scattered_power_W": float(model_power),
        "differential_cross_section_m2_sr": float(dsigma),
        "assumptions": [
            "Absolutely calibrated scattered and incident laser power.",
            "Optically thin, single-scattering Thomson relation; polarization dependence is included when incident polarization and scattered direction vectors are supplied, otherwise the unpolarized cross-section is used.",
            "The returned density is the interaction-length-averaged effective density; it equals a local density only when density is uniform along the scattering path.",
            "Collection solid angle and optical efficiency describe the actual collection path.",
        ],
        "density_interpretation": "interaction_length_averaged_effective_density",
        "uncertainty_model": ("First-order propagation with a supplied 6x6 relative covariance matrix for scattered power, incident power, interaction length, collection solid angle, optical efficiency and polarization/differential-cross-section factor." if relative_input_covariance is not None else "Optional first-order independent-input propagation; unprovided covariance and model discrepancy are not included."),
    }

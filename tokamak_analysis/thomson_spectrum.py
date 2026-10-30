"""Legacy single-channel Gaussian Thomson-spectrum approximation.

This compatibility module is restricted to a non-collective, non-relativistic
Maxwellian model with a Gaussian instrument width. The primary Analysis 1
multi-channel inference engine is tokamak_analysis.thomson_inference, which
supports measured response kernels/matrices and additional physical models.
Wavelengths and the legacy Gaussian instrument sigma are expressed in nm.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

ELECTRON_REST_ENERGY_EV = 510_998.95069
SPEED_OF_LIGHT_M_S = 299_792_458.0



def _is_finite_real(value) -> bool:
    """Accept finite real scalar parameters, but not booleans or numeric strings."""
    if not isinstance(value, (int, float, np.integer, np.floating)):
        return False
    if isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _finite_vector(values, name: str, *, minimum_size: int = 7) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < minimum_size:
        raise ValueError(f"{name} must be a 1-D array with at least {minimum_size} samples")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans or strings")
    arr = raw.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _gaussian_linear_background(x, amplitude, center, sigma, intercept, slope):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + intercept + slope * (x - center)


def fit_thomson_spectrum(
    wavelength_nm,
    signal,
    *,
    laser_wavelength_nm: float,
    scattering_angle_rad: float,
    instrument_sigma_nm: float,
    signal_uncertainty=None,
    spectral_power_density_calibrated: bool = False,
) -> dict:
    """Fit a raw Thomson spectrum and estimate electron temperature.

    Required conditions:
      * wavelength axis is calibrated and strictly increasing (nm);
      * instrument line-spread function is Gaussian with known sigma (nm);
      * the spectrum is dominated by non-collective, non-relativistic Thomson
        scattering from an approximately Maxwellian electron population;
      * signal is proportional to spectral intensity and includes a locally
        linear background.

    Temperature uses sigma_physical^2 = sigma_fit^2 - sigma_instrument^2 and
    T_e[eV] = (sigma_physical / lambda_0)^2 * (m_e c^2 / 4) /
              sin(scattering_angle / 2)^2.
    By default, signal amplitude is not converted to power or density. If
    spectral_power_density_calibrated=True, signal must be calibrated in W/nm;
    the fitted Gaussian area is then reported as collected power in W. Absolute
    density still requires a separate calibrated optical geometry/power model.
    """
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")

    x = _finite_vector(wavelength_nm, "wavelength_nm")
    y = _finite_vector(signal, "signal")
    if x.size != y.size:
        raise ValueError("wavelength_nm and signal must have equal lengths")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if not _is_finite_real(laser_wavelength_nm) or laser_wavelength_nm <= 0:
        raise ValueError("laser_wavelength_nm must be positive and finite")
    if not _is_finite_real(scattering_angle_rad) or not (0 < scattering_angle_rad <= np.pi):
        raise ValueError("scattering_angle_rad must be finite and in (0, pi]")
    if not _is_finite_real(instrument_sigma_nm) or instrument_sigma_nm <= 0:
        raise ValueError("instrument_sigma_nm must be positive and finite")
    if not np.all(np.isfinite(y)):
        raise ValueError("signal must contain only finite values")

    uncertainty = None
    if signal_uncertainty is not None:
        uncertainty = _finite_vector(signal_uncertainty, "signal_uncertainty")
        if uncertainty.size != y.size:
            raise ValueError("signal_uncertainty must have the same length as signal")
        if np.any(uncertainty <= 0):
            raise ValueError("signal_uncertainty must be strictly positive")

    signal_scale = float(np.max(np.abs(y)))
    if not np.isfinite(signal_scale) or signal_scale <= 0:
        raise ValueError("signal must contain at least one non-zero finite value")
    y_scaled = y / signal_scale
    uncertainty_scaled = None if uncertainty is None else uncertainty / signal_scale
    peak_index = int(np.argmax(y_scaled))
    baseline = float(np.median(np.r_[y_scaled[: max(2, y.size // 10)], y_scaled[-max(2, y.size // 10):]]))
    amplitude0 = float(np.max(y_scaled) - baseline)
    if not np.isfinite(amplitude0) or amplitude0 <= 0:
        raise ValueError("signal must contain a positive peak above its edge baseline")
    center0 = float(x[peak_index])
    spacing = float(np.median(np.diff(x)))
    sigma0 = max(spacing, (float(x[-1] - x[0]) / 10.0))
    span = float(x[-1] - x[0])
    lower = [0.0, float(x[0]), max(spacing * 0.25, np.finfo(float).eps), -np.inf, -np.inf]
    upper = [np.inf, float(x[-1]), span, np.inf, np.inf]
    p0 = [amplitude0, center0, min(sigma0, span * 0.5), baseline, 0.0]

    scaled_params, scaled_covariance = curve_fit(
        _gaussian_linear_background,
        x,
        y_scaled,
        p0=p0,
        bounds=(lower, upper),
        sigma=uncertainty_scaled,
        absolute_sigma=uncertainty is not None,
        maxfev=50_000,
    )
    parameter_scales = np.array([signal_scale, 1.0, 1.0, signal_scale, signal_scale])
    params = np.asarray(scaled_params, dtype=float) * parameter_scales
    covariance = scaled_covariance * np.outer(parameter_scales, parameter_scales)
    amplitude, center, sigma_fit, intercept, slope = [float(v) for v in params]
    if sigma_fit <= instrument_sigma_nm:
        raise ValueError(
            "Fitted width is not larger than the instrumental width; electron temperature is unresolved"
        )
    if center - 3.0 * sigma_fit < x[0] or center + 3.0 * sigma_fit > x[-1]:
        raise ValueError(
            "Wavelength window must cover at least plus/minus 3 fitted standard deviations around the line center"
        )
    sigma_physical = float(np.sqrt(sigma_fit**2 - instrument_sigma_nm**2))
    angular_factor = float(np.sin(scattering_angle_rad / 2.0) ** 2)
    temperature_ev = float(
        (sigma_physical / laser_wavelength_nm) ** 2
        * (ELECTRON_REST_ENERGY_EV / 4.0)
        / angular_factor
    )

    fitted = _gaussian_linear_background(x, *params)
    residual = y - fitted
    dof = max(1, y.size - params.size)
    if uncertainty is None:
        # curve_fit already scales covariance by residual variance when
        # absolute_sigma=False; do not scale it a second time.
        residual_sum_squares = float(np.sum(residual**2))
        chi_square = None
        reduced_chi_square = None
        residual_variance = float(residual_sum_squares / dof)
        covariance_scale = 1.0
    else:
        residual_sum_squares = float(np.sum(residual**2))
        chi_square = float(np.sum((residual / uncertainty) ** 2))
        reduced_chi_square = float(chi_square / dof)
        residual_variance = None
        covariance_scale = 1.0
    sigma_fit_uncertainty = float(np.sqrt(max(0.0, covariance[2, 2] * covariance_scale)))
    temperature_uncertainty = float(
        temperature_ev * 2.0 * sigma_fit * sigma_fit_uncertainty / sigma_physical**2
    )
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

    if spectral_power_density_calibrated:
        # The integral of the fitted Gaussian is preserved by convolution with
        # a normalized instrument response; the signal axis must be in W/nm.
        area_factor = float(np.sqrt(2.0 * np.pi))
        integrated_power = float(amplitude * sigma_fit * area_factor)
        area_variance = 2.0 * np.pi * (
            sigma_fit**2 * covariance[0, 0]
            + amplitude**2 * covariance[2, 2]
            + 2.0 * amplitude * sigma_fit * covariance[0, 2]
        )
        result_area_uncertainty = float(np.sqrt(max(0.0, area_variance)))
    else:
        integrated_power = None
        result_area_uncertainty = None

    return {
        "electron_temperature_eV": temperature_ev,
        "electron_temperature_uncertainty_eV": temperature_uncertainty,
        "amplitude": amplitude,
        "center_wavelength_nm": center,
        "fitted_sigma_nm": sigma_fit,
        "instrument_sigma_nm": float(instrument_sigma_nm),
        "physical_sigma_nm": sigma_physical,
        "background_intercept": intercept,
        "background_slope_per_nm": slope,
        "fitted_spectrum": fitted,
        "residual": residual,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "residual_sum_squares": residual_sum_squares,
        "residual_variance": residual_variance,
        "r_squared": r_squared,
        "degrees_of_freedom": int(dof),
        "fit_covariance": covariance,
        "spectral_power_density_calibrated": bool(spectral_power_density_calibrated),
        "integrated_scattered_power_W": integrated_power,
        "integrated_scattered_power_uncertainty_W": result_area_uncertainty,
        "model": "Gaussian Doppler profile convolved with Gaussian instrument response, plus linear background",
        "assumptions": [
            "Non-collective Thomson scattering",
            "Non-relativistic Maxwellian electron velocity distribution",
            "Gaussian instrumental line-spread function with known sigma",
            "Calibrated wavelength axis and locally linear background",
        ],
        "limitations": [
            "Does not infer electron density from fitted amplitude",
            "Does not model collective scattering, relativistic spectral shape, non-Maxwellian distributions, or non-Gaussian instrument response",
            "Temperature uncertainty is a local covariance-based estimate and excludes uncertainty in laser wavelength, scattering angle, and instrumental width",
        ],
    }

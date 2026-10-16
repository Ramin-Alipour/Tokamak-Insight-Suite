"""Raw-spectrum Thomson-scattering fit for a non-relativistic Maxwellian model.

This module fits a Gaussian Doppler-broadened spectrum plus a linear background.
It is an explicitly scoped approximation for non-collective Thomson scattering;
it is not a general relativistic or collective-scattering spectral solver.
Wavelengths and instrumental Gaussian sigma are expressed in nm.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

ELECTRON_REST_ENERGY_EV = 510_998.95069
SPEED_OF_LIGHT_M_S = 299_792_458.0


def _finite_vector(values, name: str, *, minimum_size: int = 7) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1 or arr.size < minimum_size:
        raise ValueError(f"{name} must be a 1-D array with at least {minimum_size} samples")
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
    T_e[eV] = (sigma_physical / lambda_0)^2 * (m_e c^2 / 2) /
              sin(scattering_angle / 2)^2.
    By default, signal amplitude is not converted to power or density. If
    spectral_power_density_calibrated=True, signal must be calibrated in W/nm;
    the fitted Gaussian area is then reported as collected power in W. Absolute
    density still requires a separate calibrated optical geometry/power model.
    """
    x = _finite_vector(wavelength_nm, "wavelength_nm")
    y = _finite_vector(signal, "signal")
    if x.size != y.size:
        raise ValueError("wavelength_nm and signal must have equal lengths")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if not np.isfinite(laser_wavelength_nm) or laser_wavelength_nm <= 0:
        raise ValueError("laser_wavelength_nm must be positive and finite")
    if not np.isfinite(scattering_angle_rad) or not (0 < scattering_angle_rad <= np.pi):
        raise ValueError("scattering_angle_rad must be finite and in (0, pi]")
    if not np.isfinite(instrument_sigma_nm) or instrument_sigma_nm <= 0:
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

    peak_index = int(np.argmax(y))
    baseline = float(np.median(np.r_[y[: max(2, y.size // 10)], y[-max(2, y.size // 10):]]))
    amplitude0 = float(np.max(y) - baseline)
    if not np.isfinite(amplitude0) or amplitude0 <= 0:
        raise ValueError("signal must contain a positive peak above its edge baseline")
    center0 = float(x[peak_index])
    spacing = float(np.median(np.diff(x)))
    sigma0 = max(spacing, (float(x[-1] - x[0]) / 10.0))
    span = float(x[-1] - x[0])
    lower = [0.0, float(x[0]), max(spacing * 0.25, np.finfo(float).eps), -np.inf, -np.inf]
    upper = [np.inf, float(x[-1]), span, np.inf, np.inf]
    p0 = [amplitude0, center0, min(sigma0, span * 0.5), baseline, 0.0]

    params, covariance = curve_fit(
        _gaussian_linear_background,
        x,
        y,
        p0=p0,
        bounds=(lower, upper),
        sigma=uncertainty,
        absolute_sigma=uncertainty is not None,
        maxfev=50_000,
    )
    amplitude, center, sigma_fit, intercept, slope = [float(v) for v in params]
    if sigma_fit <= instrument_sigma_nm:
        raise ValueError(
            "Fitted width is not larger than the instrumental width; electron temperature is unresolved"
        )
    sigma_physical = float(np.sqrt(sigma_fit**2 - instrument_sigma_nm**2))
    angular_factor = float(np.sin(scattering_angle_rad / 2.0) ** 2)
    temperature_ev = float(
        (sigma_physical / laser_wavelength_nm) ** 2
        * (ELECTRON_REST_ENERGY_EV / 2.0)
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

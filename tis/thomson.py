"""Thomson-scattering spectral analysis for non-relativistic Maxwellian electrons.

This workflow fits a measured spectrum with a Gaussian Thomson feature plus a
linear background. It is intended for non-collective Thomson scattering where
the spectral response has already been deconvolved or is narrow compared with
the plasma feature. It does not replace an instrument-response convolution or
a collective-scattering model when those effects are important.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

ELECTRON_REST_ENERGY_EV = 510_998.95069


def _gaussian_linear_background(x, peak, center, sigma, offset, slope):
    return offset + slope * (x - center) + peak * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def analyze_thomson_spectrum(
    wavelength_nm,
    signal,
    *,
    laser_wavelength_nm: float,
    scattering_angle_deg: float,
    density_per_integrated_signal: float | None = None,
    wavelength_uncertainty_nm: float | None = None,
) -> dict:
    """Fit a Thomson-scattering spectrum and infer electron temperature.

    Parameters
    ----------
    wavelength_nm, signal:
        One-dimensional wavelength axis [nm] and background-containing signal.
        Signal units must be consistent across the spectrum.
    laser_wavelength_nm:
        Incident laser wavelength [nm]. Used as the expected spectral centre.
    scattering_angle_deg:
        Angle between incident and collected wave vectors [degrees], 0 < angle
        <= 180. The non-relativistic spectral-width relation is used.
    density_per_integrated_signal:
        Optional, externally established absolute calibration factor converting
        integrated fitted signal (signal-units * nm) to electron density [m^-3].
        No universal density calibration is assumed.
    wavelength_uncertainty_nm:
        Optional 1-sigma uncertainty for the wavelength axis [nm]. When supplied,
        it is reported but not propagated through a full instrument model.

    Returns a dictionary of fitted parameters, fit quality, and assumptions.
    """
    x = np.asarray(wavelength_nm, dtype=float).ravel()
    y = np.asarray(signal, dtype=float).ravel()
    if x.size != y.size or x.size < 8:
        raise ValueError("wavelength_nm and signal must have equal lengths of at least 8.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Spectrum values must be finite.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing without duplicates.")
    if not np.isfinite(laser_wavelength_nm) or laser_wavelength_nm <= 0:
        raise ValueError("laser_wavelength_nm must be positive and finite.")
    if not np.isfinite(scattering_angle_deg) or not 0 < scattering_angle_deg <= 180:
        raise ValueError("scattering_angle_deg must be in (0, 180].")
    if density_per_integrated_signal is not None and (
        not np.isfinite(density_per_integrated_signal) or density_per_integrated_signal <= 0
    ):
        raise ValueError("density_per_integrated_signal must be positive and finite.")
    if wavelength_uncertainty_nm is not None and (
        not np.isfinite(wavelength_uncertainty_nm) or wavelength_uncertainty_nm < 0
    ):
        raise ValueError("wavelength_uncertainty_nm must be finite and nonnegative.")

    dx = float(np.median(np.diff(x)))
    edge_n = max(2, x.size // 10)
    edge_values = np.r_[y[:edge_n], y[-edge_n:]]
    offset0 = float(np.median(edge_values))
    noise = float(1.4826 * np.median(np.abs(edge_values - np.median(edge_values))))
    peak0 = float(np.max(y) - offset0)
    if peak0 <= 0:
        peak0 = float(np.max(np.abs(y - offset0)))
    if peak0 <= 0 or not np.isfinite(peak0):
        raise ValueError("No positive spectral feature could be initialized above background.")

    center0 = float(x[np.argmax(y)])
    # Estimate a width from the positive background-subtracted second moment.
    weights = np.clip(y - offset0, 0, None)
    if weights.sum() > 0:
        sigma0 = float(np.sqrt(np.sum(weights * (x - np.average(x, weights=weights)) ** 2) / weights.sum()))
    else:
        sigma0 = dx * 2
    sigma0 = float(np.clip(sigma0, dx * 0.5, (x[-1] - x[0]) / 2.5))
    span = float(x[-1] - x[0])
    p0 = [peak0, center0, sigma0, offset0, 0.0]
    lower = [0.0, x[0], dx * 0.25, -np.inf, -np.inf]
    upper = [np.inf, x[-1], span, np.inf, np.inf]
    try:
        params, covariance = curve_fit(
            _gaussian_linear_background, x, y, p0=p0, bounds=(lower, upper),
            maxfev=50_000,
        )
    except (RuntimeError, ValueError, FloatingPointError) as exc:
        raise ValueError(f"Thomson spectrum fit failed: {exc}") from exc

    peak, center, sigma, offset, slope = [float(v) for v in params]
    fitted = _gaussian_linear_background(x, *params)
    residual = y - fitted
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y))**2))
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
    area = float(peak * sigma * np.sqrt(2 * np.pi))
    theta = np.deg2rad(scattering_angle_deg)
    sin_half_sq = np.sin(theta / 2) ** 2
    te_ev = float(ELECTRON_REST_ENERGY_EV * (sigma / laser_wavelength_nm) ** 2 / (2 * sin_half_sq))
    sigma_err = float(np.sqrt(max(covariance[2, 2], 0.0))) if covariance.shape == (5, 5) else float("nan")
    te_uncertainty_ev = (
        float(2 * te_ev * sigma_err / sigma)
        if np.isfinite(sigma_err) and sigma > 0 else None
    )
    result = {
        "diagnostic": "Thomson scattering",
        "fit_model": "Gaussian feature + linear background",
        "fit_parameters": {
            "peak_signal": peak, "center_nm": center, "sigma_nm": sigma,
            "fwhm_nm": float(2 * np.sqrt(2 * np.log(2)) * sigma),
            "background_at_center": offset, "background_slope_per_nm": slope,
            "integrated_signal_signal_nm": area,
        },
        "electron_temperature_eV": te_ev,
        "electron_temperature_fit_uncertainty_eV": te_uncertainty_ev,
        "electron_density_m3": (
            float(area * density_per_integrated_signal)
            if density_per_integrated_signal is not None else None
        ),
        "density_calibration_factor_per_integrated_signal": density_per_integrated_signal,
        "fit_quality": {
            "r_squared": r_squared,
            "residual_rms": float(np.sqrt(np.mean(residual**2))),
            "edge_noise_robust_sigma": noise,
            "peak_to_noise": float(peak / noise) if noise > 0 else None,
            "n_points": int(x.size),
        },
        "inputs": {
            "laser_wavelength_nm": float(laser_wavelength_nm),
            "scattering_angle_deg": float(scattering_angle_deg),
            "wavelength_uncertainty_nm": wavelength_uncertainty_nm,
        },
        "assumptions_and_limits": [
            "Non-collective, non-relativistic Thomson spectral-width approximation.",
            "Gaussian feature and linear background; use instrument-response convolution when needed.",
            "Electron density is returned only when an external absolute calibration factor is supplied.",
            "Fit covariance uncertainty excludes wavelength, calibration, background-model and instrument-response systematics.",
            "Do not use this simplified workflow for collective scattering or non-Maxwellian inference.",
        ],
    }
    return result

"""Computational workflow for calibrated Thomson-scattering profiles.

Input arrays are already calibrated profiles, not raw detector counts. Optional
pointwise uncertainties are propagated through linear finite-difference gradients,
normalized gradients, local scale lengths, and profile averages assuming independent
input samples.
"""
from __future__ import annotations

import numpy as np

from .thomson_scattering import build_profile, profile_summary


def _scale_length(value, gradient):
    out = np.full(value.shape, np.nan, dtype=float)
    threshold = np.finfo(float).eps * np.maximum(1.0, np.abs(value))
    valid = (value > 0) & (np.abs(gradient) > threshold)
    out[valid] = -value[valid] / gradient[valid]
    return out


def _gradient_uncertainty(derivative_matrix, uncertainty):
    if uncertainty is None:
        return None, None
    variance = (derivative_matrix**2) @ (uncertainty**2)
    standard_uncertainty = np.sqrt(np.maximum(variance, 0.0))
    covariance_value_gradient = np.diag(derivative_matrix) * uncertainty**2
    return standard_uncertainty, covariance_value_gradient


def _normalized_gradient_uncertainty(value, gradient, value_uncertainty, gradient_uncertainty, covariance):
    out = np.full(value.shape, np.nan, dtype=float)
    valid = value > 0
    if not np.any(valid):
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = (
            gradient_uncertainty[valid]**2 / value[valid]**2
            + gradient[valid]**2 * value_uncertainty[valid]**2 / value[valid]**4
            - 2.0 * gradient[valid] * covariance[valid] / value[valid]**3
        )
    out[valid] = np.sqrt(np.maximum(variance, 0.0))
    return out


def _scale_length_uncertainty(value, gradient, value_uncertainty, gradient_uncertainty, covariance):
    out = np.full(value.shape, np.nan, dtype=float)
    threshold = np.finfo(float).eps * np.maximum(1.0, np.abs(value))
    valid = (value > 0) & (np.abs(gradient) > threshold)
    if not np.any(valid):
        return out
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        variance = (
            value_uncertainty[valid]**2 / gradient[valid]**2
            + value[valid]**2 * gradient_uncertainty[valid]**2 / gradient[valid]**4
            - 2.0 * value[valid] * covariance[valid] / gradient[valid]**3
        )
    out[valid] = np.sqrt(np.maximum(variance, 0.0))
    return out


def analyze_thomson_profile(
    radius_m,
    density_m3,
    temperature_ev,
    *,
    density_uncertainty_m3=None,
    temperature_uncertainty_ev=None,
    major_radius_m=None,
):
    """Calculate profile summaries, gradients, scale lengths, and available uncertainties."""
    p = build_profile(
        radius_m, density_m3, temperature_ev,
        density_uncertainty_m3, temperature_uncertainty_ev,
    )
    r, ne, te = p.radius_m, p.density_m3, p.temperature_ev
    edge_order = 2 if r.size >= 3 else 1
    derivative_matrix = np.gradient(np.eye(r.size), r, axis=0, edge_order=edge_order)
    dne_dr = derivative_matrix @ ne
    dte_dr = derivative_matrix @ te
    density_normalized_gradient = np.divide(
        dne_dr, ne, out=np.full_like(dne_dr, np.nan), where=ne > 0
    )
    temperature_normalized_gradient = np.divide(
        dte_dr, te, out=np.full_like(dte_dr, np.nan), where=te > 0
    )
    result = {
        "radius_m": r.copy(),
        "density_m3": ne.copy(),
        "temperature_eV": te.copy(),
        "density_gradient_m-4": dne_dr,
        "temperature_gradient_eV_m-1": dte_dr,
        "density_normalized_gradient_m-1": density_normalized_gradient,
        "temperature_normalized_gradient_m-1": temperature_normalized_gradient,
        "density_scale_length_m": _scale_length(ne, dne_dr),
        "temperature_scale_length_m": _scale_length(te, dte_dr),
        "summary": profile_summary(p, major_radius_m=major_radius_m),
        "notes": [
            "Input must be calibrated Thomson-scattering profile data.",
            "Gradient and derived uncertainty propagation assumes independent pointwise errors.",
            "Volume averages, when requested, use circular large-aspect-ratio weighting.",
            "NaN scale lengths mark zero/near-zero gradients or profile values.",
            "Peak-location uncertainty is not estimated by this workflow.",
        ],
    }
    if p.density_uncertainty_m3 is not None:
        sne = p.density_uncertainty_m3
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, sne)
        result["density_uncertainty_m3"] = sne.copy()
        result["density_gradient_uncertainty_m-4"] = grad_unc
        result["density_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            ne, dne_dr, sne, grad_unc, cov
        )
        result["density_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            ne, dne_dr, sne, grad_unc, cov
        )
    if p.temperature_uncertainty_ev is not None:
        ste = p.temperature_uncertainty_ev
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, ste)
        result["temperature_uncertainty_eV"] = ste.copy()
        result["temperature_gradient_uncertainty_eV_m-1"] = grad_unc
        result["temperature_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            te, dte_dr, ste, grad_unc, cov
        )
        result["temperature_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            te, dte_dr, ste, grad_unc, cov
        )
    return result


def analyze_thomson_spectrum_and_density(
    wavelength_nm,
    spectral_power_density_W_nm,
    *,
    laser_wavelength_nm: float,
    scattering_angle_rad: float,
    instrument_sigma_nm: float,
    incident_power_W: float,
    interaction_length_m: float,
    collection_solid_angle_sr: float,
    optical_efficiency: float = 1.0,
    signal_uncertainty_W_nm=None,
    incident_power_uncertainty_W: float | None = None,
    interaction_length_uncertainty_m: float | None = None,
    scattering_angle_uncertainty_rad: float | None = None,
    collection_solid_angle_uncertainty_sr: float | None = None,
    optical_efficiency_uncertainty: float | None = None,
) -> dict:
    """Analyze a calibrated Thomson spectrum for electron temperature and density.

    This end-to-end path is intentionally restricted to a spectrum calibrated as
    collected spectral power density in W/nm, an approximately non-collective,
    non-relativistic Maxwellian line shape, a known Gaussian instrument width,
    and a compatible single-band optical-power model. The fitted Gaussian area
    estimates collected scattered power; the classical unpolarized differential
    Thomson cross-section then estimates density. Calibration/model discrepancy,
    uncertainty correlations, stray-light systematics and spectral-response
    errors are not inferred automatically and must be assessed separately.
    """
    from .thomson_scattering import density_from_thomson_scattered_power
    from .thomson_spectrum import fit_thomson_spectrum

    fit = fit_thomson_spectrum(
        wavelength_nm,
        spectral_power_density_W_nm,
        laser_wavelength_nm=laser_wavelength_nm,
        scattering_angle_rad=scattering_angle_rad,
        instrument_sigma_nm=instrument_sigma_nm,
        signal_uncertainty=signal_uncertainty_W_nm,
        spectral_power_density_calibrated=True,
    )
    density = density_from_thomson_scattered_power(
        fit["integrated_scattered_power_W"],
        incident_power_W=incident_power_W,
        interaction_length_m=interaction_length_m,
        scattering_angle_rad=scattering_angle_rad,
        collection_solid_angle_sr=collection_solid_angle_sr,
        optical_efficiency=optical_efficiency,
        scattered_power_uncertainty_W=fit["integrated_scattered_power_uncertainty_W"],
        incident_power_uncertainty_W=incident_power_uncertainty_W,
        interaction_length_uncertainty_m=interaction_length_uncertainty_m,
        scattering_angle_uncertainty_rad=scattering_angle_uncertainty_rad,
        collection_solid_angle_uncertainty_sr=collection_solid_angle_uncertainty_sr,
        optical_efficiency_uncertainty=optical_efficiency_uncertainty,
    )
    return {
        "electron_temperature_eV": fit["electron_temperature_eV"],
        "electron_temperature_uncertainty_eV": fit["electron_temperature_uncertainty_eV"],
        "electron_density_m3": density["electron_density_m3"],
        "electron_density_uncertainty_m3": density.get("electron_density_uncertainty_m3"),
        "integrated_scattered_power_W": fit["integrated_scattered_power_W"],
        "integrated_scattered_power_uncertainty_W": fit["integrated_scattered_power_uncertainty_W"],
        "fit": fit,
        "density": density,
        "assumptions": [
            "Input signal is calibrated collected spectral power density in W/nm.",
            "The spectral fit uses a non-collective, non-relativistic Maxwellian Gaussian model.",
            "The instrument line-spread function is Gaussian with known standard deviation.",
            "The density relation assumes optically thin single scattering and compatible band/time-gate calibration.",
            "Input uncertainty propagation assumes independent errors and excludes calibration/model discrepancy.",
        ],
        "limitations": [
            "Not valid for collective spectra, strongly relativistic or non-Maxwellian distributions, or non-Gaussian instrument response.",
            "Absolute density depends directly on laser-power, interaction-length, collection-solid-angle and optical-efficiency calibration.",
            "Stray-light subtraction, wavelength-dependent throughput, spectral-band completeness and calibration covariance are not estimated.",
            "Review fit residuals and uncertainty budget before interpreting results scientifically.",
        ],
    }

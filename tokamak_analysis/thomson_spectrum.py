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
    """Compatibility adapter to the canonical Thomson inference engine.

    The historical Gaussian-width API is retained, but no independent fit is
    performed here. A Gaussian line-spread response matrix is constructed and
    passed to Analysis 1's canonical multi-channel Maxwellian inference path.
    """
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")
    x = _finite_vector(wavelength_nm, "wavelength_nm")
    y = _finite_vector(signal, "signal")
    if x.size != y.size or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm and signal must have equal lengths and wavelength must be strictly increasing")
    if not _is_finite_real(instrument_sigma_nm) or instrument_sigma_nm <= 0:
        raise ValueError("instrument_sigma_nm must be positive and finite")
    if signal_uncertainty is not None:
        uncertainty = _finite_vector(signal_uncertainty, "signal_uncertainty")
        if uncertainty.shape != y.shape or np.any(uncertainty <= 0):
            raise ValueError("signal_uncertainty must be strictly positive and match signal")
    else:
        uncertainty = None
    # A Gaussian LSF matrix maps true wavelength bins to detector bins.
    delta = x[:, None] - x[None, :]
    response = np.exp(-0.5 * (delta / float(instrument_sigma_nm)) ** 2)
    response /= response.sum(axis=0, keepdims=True)
    from .thomson_inference import analyze_thomson_spectra
    try:
        result = analyze_thomson_spectra(
            x, y[None, :], [0.0],
        laser_wavelength_nm=laser_wavelength_nm,
        scattering_angle_rad=scattering_angle_rad,
            instrument_response=response,
            signal_uncertainty=None if uncertainty is None else uncertainty[None, :],
            model="non_collective_maxwellian",
            spectral_power_density_calibrated=spectral_power_density_calibrated,
        )["channels"][0]
    except ValueError as exc:
        if "Wavelength coverage must include" in str(exc):
            raise ValueError("Wavelength window must cover the fitted Thomson spectral feature and its intrinsic wings.") from exc
        raise
    intrinsic_sigma = float(result["intrinsic_sigma_nm"])
    intrinsic_uncertainty = float(np.sqrt(max(float(np.asarray(result["fit_covariance"])[2, 2]), 0.0)))
    if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0 or intrinsic_sigma <= intrinsic_uncertainty:
        raise ValueError("Fitted physical width is not resolved: it is not larger than the instrumental width uncertainty at the available spectral resolution.")
    fitted_sigma = float(np.sqrt(intrinsic_sigma**2 + float(instrument_sigma_nm)**2))
    residual = np.asarray(result["residual"], dtype=float)
    dof = max(1, y.size - 5)
    reduced_chi_square = result.get("fit_reduced_chi_square")
    chi_square = None if reduced_chi_square is None else float(reduced_chi_square * dof)
    ss = float(residual @ residual)
    total = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "electron_temperature_eV": result["electron_temperature_eV"],
        "electron_temperature_uncertainty_eV": result["electron_temperature_uncertainty_eV"],
        "amplitude": result["amplitude"],
        "center_wavelength_nm": result["fitted_center_nm"],
        "fitted_sigma_nm": fitted_sigma,
        "instrument_sigma_nm": float(instrument_sigma_nm),
        "physical_sigma_nm": intrinsic_sigma,
        "background_intercept": result["background_intercept"],
        "background_slope_per_nm": result["background_slope_per_nm"],
        "fitted_spectrum": result["fitted_spectrum"],
        "residual": residual,
        "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "residual_sum_squares": ss,
        "residual_variance": None if uncertainty is not None else float(ss / dof),
        "r_squared": float(1.0 - ss / total) if total > 0 else float("nan"),
        "degrees_of_freedom": int(dof),
        "fit_covariance": result["fit_covariance"],
        "spectral_power_density_calibrated": bool(spectral_power_density_calibrated),
        "integrated_scattered_power_W": result.get("integrated_scattered_power_W"),
        "integrated_scattered_power_uncertainty_W": result.get("integrated_scattered_power_uncertainty_W"),
        "model": "Canonical Thomson inference: non-collective Maxwellian convolved with a Gaussian response matrix",
        "assumptions": result["assumptions"] + ["Gaussian instrumental line-spread function represented on the measured wavelength grid"],
        "limitations": [
            "Does not infer electron density from spectral shape; all fitting and physical inference are delegated to Analysis 1's canonical engine.",
            "Absolute electron density still requires separately calibrated optical power and geometry.",
        ],
        "canonical_engine": "tokamak_analysis.thomson_inference.analyze_thomson_spectra",
        "fit_accepted": result["fit_accepted"],
        "fit_quality_status": result["fit_quality_status"],
    }

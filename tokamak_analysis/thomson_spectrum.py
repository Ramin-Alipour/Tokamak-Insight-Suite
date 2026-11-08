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
    instrument_sigma_nm: float | None = None,
    instrument_response_matrix=None,
    signal_uncertainty=None,
    spectral_power_density_calibrated: bool = False,
) -> dict:
    """Compatibility adapter to the canonical Thomson inference engine.

    The historical scalar-width argument is retained only as an optional
    measured width summary. A measured instrument-response matrix is mandatory;
    this adapter never synthesizes a Gaussian response from a scalar width.
    """
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")
    x = _finite_vector(wavelength_nm, "wavelength_nm")
    y = _finite_vector(signal, "signal")
    if x.size != y.size or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm and signal must have equal lengths and wavelength must be strictly increasing")
    if instrument_sigma_nm is not None and (not _is_finite_real(instrument_sigma_nm) or instrument_sigma_nm <= 0):
        raise ValueError("instrument_sigma_nm must be positive and finite when supplied")
    if signal_uncertainty is not None:
        uncertainty = _finite_vector(signal_uncertainty, "signal_uncertainty")
        if uncertainty.shape != y.shape or np.any(uncertainty <= 0):
            raise ValueError("signal_uncertainty must be strictly positive and match signal")
    else:
        uncertainty = None
    if instrument_response_matrix is None:
        raise ValueError("A measured instrument_response_matrix is required; a scalar Gaussian width is not converted into a synthetic response.")
    response = np.asarray(instrument_response_matrix, dtype=float)
    if response.shape != (x.size, x.size) or np.any(~np.isfinite(response)) or np.any(response < 0) or np.any(response.sum(axis=0) <= 0):
        raise ValueError("instrument_response_matrix must be a finite non-negative measured square matrix with positive column sums")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "measured instrument_response_matrix columns must already sum to one; "
            "the calibrated response operator is not renormalized"
        )
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
    if not np.isfinite(intrinsic_sigma) or intrinsic_sigma <= 0:
        raise ValueError("Fitted physical width is not positive and finite.")
    if intrinsic_sigma <= intrinsic_uncertainty:
        raise ValueError("Fitted physical width is not resolved relative to its fit uncertainty, independent of whether a scalar instrument-width summary was supplied.")
    # The canonical fit has already convolved the intrinsic physical spectrum
    # with the supplied measured response matrix. Derive the observed width from
    # that convolved fitted component; do not approximate an arbitrary response
    # matrix by adding an optional scalar width in quadrature.
    fitted_center = float(result["fitted_center_nm"])
    fitted_background = (
        float(result["background_intercept"])
        + float(result["background_slope_per_nm"]) * (x - fitted_center)
    )
    fitted_component = np.asarray(result["fitted_spectrum"], dtype=float) - fitted_background
    fitted_component = np.maximum(fitted_component, 0.0)
    # Use the same wavelength-bin quadrature as the forward response operator.
    # Trapezoidal integration is equivalent on uniform grids but differs on
    # nonuniform grids, where mixing the two conventions biases the width.
    bin_edges = np.empty(x.size + 1, dtype=float)
    bin_edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    bin_edges[0] = x[0] - 0.5 * (x[1] - x[0])
    bin_edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    bin_widths = np.diff(bin_edges)
    if np.any(bin_widths <= 0.0):
        raise ValueError("Response-convolved fitted Thomson feature requires positive wavelength-bin widths.")
    # The response operator acts on wavelength-bin integrated power. Use the
    # identical finite-volume measure for area, centroid and variance, including
    # non-uniform wavelength grids (not trapezoidal point-sample weights).
    fitted_area = float(np.sum(fitted_component * bin_widths))
    if not np.isfinite(fitted_area) or fitted_area <= 0.0:
        raise ValueError("Response-convolved fitted Thomson feature has no positive finite integrated area.")
    fitted_mean = float(np.sum(x * fitted_component * bin_widths) / fitted_area)
    fitted_variance = float(
        np.sum((x - fitted_mean) ** 2 * fitted_component * bin_widths) / fitted_area
    )
    if not np.isfinite(fitted_variance) or fitted_variance <= 0.0:
        raise ValueError("Response-convolved fitted Thomson feature has no positive finite spectral variance.")
    fitted_sigma = float(np.sqrt(fitted_variance))
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
        "response_convolved_sigma_nm": fitted_sigma,
        "instrument_sigma_nm": None if instrument_sigma_nm is None else float(instrument_sigma_nm),
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
        "model": "Canonical Thomson inference: non-collective Maxwellian convolved with the supplied measured response matrix",
        "assumptions": result["assumptions"] + ["Measured instrument-response matrix supplied by the caller; response-convolved width is computed from the fitted spectrum without adding a second Gaussian width"],
        "limitations": [
            "Does not infer electron density from spectral shape; all fitting and physical inference are delegated to Analysis 1's canonical engine.",
            "Absolute electron density still requires separately calibrated optical power and geometry.",
        ],
        "canonical_engine": "tokamak_analysis.thomson_inference.analyze_thomson_spectra",
        "fit_accepted": result["fit_accepted"],
        "fit_quality_status": result["fit_quality_status"],
    }

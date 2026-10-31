"""Model-independent diagnostics for calibrated Thomson-scattering spectra.

This module characterizes the measured spectral distribution after local linear
background removal and Gaussian instrument-width deconvolution. It does not
pretend that arbitrary spectral moments uniquely recover an electron velocity
distribution; physical distribution inference requires an experiment-appropriate
collective/non-collective forward model and instrument response.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit


def _vector(values, name, *, minimum=9):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum} samples")
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans or strings")
    arr = raw.astype(float, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _gaussian_background(x, amplitude, center, sigma, intercept, slope):
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2) + intercept + slope * (x - center)


def _trapz(y, x):
    integrate = getattr(np, "trapezoid", None)
    return integrate(y, x) if integrate is not None else np.trapz(y, x)


def _moment_statistics(
    x, y, instrument_sigma_nm, background_fraction, background_mask,
    background_signal=None,
):
    if background_signal is not None:
        background = np.asarray(background_signal, dtype=float)
        if background.shape != x.shape or not np.all(np.isfinite(background)):
            raise ValueError("background_signal must be a finite array matching wavelength_nm")
        background_source = "explicit_fitted_background_signal"
    else:
        if background_mask is None:
            edge_width = float(background_fraction) * float(x[-1] - x[0])
            mask = (x <= x[0] + edge_width) | (x >= x[-1] - edge_width)
            background_source = "spectral_edges_assumed_background_by_wavelength_span"
        else:
            mask = background_mask
            background_source = "user_supplied_background_mask"
        if np.count_nonzero(mask) < 3:
            raise ValueError("background_mask must select at least three wavelength samples")
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        background = slope * x + intercept
    corrected = y - background
    positive = np.maximum(corrected, 0.0)
    negative = np.minimum(corrected, 0.0)
    positive_area = float(_trapz(positive, x))
    negative_area = float(_trapz(negative, x))
    signed_area = float(_trapz(corrected, x))
    signal_scale = float(np.max(np.abs(y)))
    area_tolerance = 100.0 * np.finfo(float).eps * signal_scale * float(x[-1] - x[0])
    if not np.isfinite(signed_area) or signed_area <= area_tolerance:
        raise ValueError("No positive spectral area (signed integral) remains after background subtraction")
    quadrature = np.empty_like(x)
    quadrature[0] = 0.5 * (x[1] - x[0])
    quadrature[-1] = 0.5 * (x[-1] - x[-2])
    quadrature[1:-1] = 0.5 * (x[2:] - x[:-2])
    measure = corrected * quadrature
    measure_sum = float(np.sum(measure))
    normalized = measure / measure_sum
    center = float(np.sum(normalized * x))
    centered = x - center
    variance = float(np.sum(normalized * centered**2))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Signed spectral measure has non-positive observed variance; inspect background selection and signal uncertainty")
    sigma_observed = float(np.sqrt(variance))
    if instrument_sigma_nm is None:
        variance_physical = None
        resolved = None
        sigma_physical = None
    else:
        variance_physical = variance - float(instrument_sigma_nm)**2
        resolved = bool(variance_physical > 0)
        sigma_physical = float(np.sqrt(variance_physical)) if resolved else None
    skewness = float(np.sum(normalized * centered**3) / variance**1.5)
    kurtosis = float(np.sum(normalized * centered**4) / variance**2 - 3.0)
    total_absolute = positive_area + abs(negative_area)
    negative_fraction = abs(negative_area) / total_absolute if total_absolute > 0 else 0.0
    return {
        "background": background, "corrected": corrected, "positive": positive,
        "negative_area": negative_area, "positive_area": positive_area,
        "signed_area": signed_area, "quadrature": quadrature, "measure": measure,
        "center": center, "variance": variance, "sigma_observed": sigma_observed,
        "variance_physical": variance_physical, "resolved": resolved,
        "sigma_physical": sigma_physical, "skewness": skewness,
        "kurtosis": kurtosis, "negative_fraction": negative_fraction,
        "background_source": background_source,
    }


def analyze_thomson_spectral_distribution(
    wavelength_nm,
    signal,
    *,
    instrument_sigma_nm: float | None = None,
    instrument_sigma_uncertainty_nm: float | None = None,
    instrument_response_matrix=None,
    signal_uncertainty=None,
    signal_covariance=None,
    background_fraction: float = 0.1,
    background_mask=None,
    background_signal=None,
    moment_mc_samples: int = 200,
) -> dict:
    """Return signed spectral moments and instrument-aware Gaussian-fit diagnostics.

    Supply either a scalar Gaussian instrument sigma or a general measured response
    matrix. A general response matrix is used in the Gaussian forward fit; no scalar
    Gaussian deconvolution is then claimed. A caller-supplied background mask can
    replace the edge-background assumption. Signed, background-subtracted samples
    (including negative noise residuals) are used for moments; clipping is forbidden.
    """
    x = _vector(wavelength_nm, "wavelength_nm")
    y = _vector(signal, "signal")
    if x.size != y.size:
        raise ValueError("wavelength_nm and signal must have equal lengths")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if (instrument_sigma_nm is None) == (instrument_response_matrix is None):
        raise ValueError("Provide exactly one of instrument_sigma_nm or instrument_response_matrix")
    if instrument_sigma_nm is not None:
        if isinstance(instrument_sigma_nm, (bool, np.bool_)) or not isinstance(instrument_sigma_nm, (int, float, np.integer, np.floating)) or not np.isfinite(instrument_sigma_nm) or instrument_sigma_nm <= 0:
            raise ValueError("instrument_sigma_nm must be positive and finite (a real number)")
    if instrument_sigma_uncertainty_nm is not None:
        if instrument_sigma_nm is None:
            raise ValueError("instrument_sigma_uncertainty_nm requires scalar instrument_sigma_nm")
        if isinstance(instrument_sigma_uncertainty_nm, (bool, np.bool_)) or not isinstance(instrument_sigma_uncertainty_nm, (int, float, np.integer, np.floating)) or not np.isfinite(instrument_sigma_uncertainty_nm) or instrument_sigma_uncertainty_nm < 0:
            raise ValueError("instrument_sigma_uncertainty_nm must be finite and non-negative")
    if isinstance(background_fraction, (bool, np.bool_)) or not isinstance(background_fraction, (int, float, np.integer, np.floating)) or not np.isfinite(background_fraction) or not (0.02 <= background_fraction <= 0.4):
        raise ValueError("background_fraction must be a finite real number in [0.02, 0.4]")
    if isinstance(moment_mc_samples, (bool, np.bool_)) or not isinstance(moment_mc_samples, (int, np.integer)) or moment_mc_samples < 20:
        raise ValueError("moment_mc_samples must be an integer >= 20")

    response = None
    if instrument_response_matrix is not None:
        response = np.asarray(instrument_response_matrix, dtype=float)
        if response.shape != (x.size, x.size) or not np.all(np.isfinite(response)) or np.any(response < 0):
            raise ValueError("instrument_response_matrix must be a finite non-negative square matrix on the wavelength grid")
        column_sums = np.sum(response, axis=0)
        if np.any(column_sums <= 0):
            raise ValueError("instrument_response_matrix must have positive column sums")
        response = response / column_sums[None, :]

    mask = None
    if background_mask is not None:
        raw_mask = np.asarray(background_mask)
        if raw_mask.dtype.kind != "b" or raw_mask.shape != x.shape:
            raise ValueError("background_mask must be a boolean array matching wavelength_nm")
        if np.count_nonzero(raw_mask) < 3:
            raise ValueError("background_mask must select at least three samples")
        mask = raw_mask.astype(bool, copy=True)
    explicit_background = None
    if background_signal is not None:
        if mask is not None:
            raise ValueError("Supply background_signal or background_mask, not both.")
        explicit_background = _vector(background_signal, "background_signal")
        if explicit_background.size != x.size:
            raise ValueError("background_signal must have the same length as wavelength_nm")

    uncertainty = None
    covariance = None
    if signal_uncertainty is not None and signal_covariance is not None:
        raise ValueError("Supply signal_uncertainty or signal_covariance, not both.")
    if signal_uncertainty is not None:
        uncertainty = _vector(signal_uncertainty, "signal_uncertainty")
        if uncertainty.size != y.size:
            raise ValueError("signal_uncertainty must have the same length as signal")
        if np.any(uncertainty <= 0):
            raise ValueError("signal_uncertainty must be strictly positive")
    if signal_covariance is not None:
        covariance_raw = np.asarray(signal_covariance)
        if covariance_raw.dtype.kind not in "iuf":
            raise ValueError("signal_covariance must contain real numeric values")
        covariance = covariance_raw.astype(float, copy=False)
        if covariance.shape != (y.size, y.size) or not np.all(np.isfinite(covariance)):
            raise ValueError("signal_covariance must be a finite square matrix matching signal")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError("signal_covariance must be symmetric")
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("signal_covariance must be positive definite") from exc

    moments = _moment_statistics(x, y, instrument_sigma_nm, background_fraction, mask, explicit_background)
    background = moments["background"]
    corrected = moments["corrected"]
    weights = moments["positive"]
    area = moments["positive_area"]
    negative_area = moments["negative_area"]
    center = moments["center"]
    variance_observed = moments["variance"]
    sigma_observed = moments["sigma_observed"]
    variance_physical = moments["variance_physical"]
    resolved = moments["resolved"]
    sigma_physical = moments["sigma_physical"]
    skewness = moments["skewness"]
    excess_kurtosis = moments["kurtosis"]
    quadrature = moments["quadrature"]
    spectral_measure = moments["measure"]

    peak = int(np.argmax(corrected))
    amplitude0 = float(max(corrected[peak], np.finfo(float).tiny))
    spacing = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    p0 = [amplitude0, float(x[peak]), max(sigma_observed, spacing), float(np.interp(x[peak], x, background)), float(np.polyfit(x[mask] if mask is not None else np.r_[x[:max(3,int(np.ceil(x.size*background_fraction)))],x[-max(3,int(np.ceil(x.size*background_fraction))):]], y[mask] if mask is not None else np.r_[y[:max(3,int(np.ceil(x.size*background_fraction)))],y[-max(3,int(np.ceil(x.size*background_fraction))):]], 1)[0])]
    lower = [0.0, float(x[0]), max(spacing * 0.25, np.finfo(float).eps), -np.inf, -np.inf]
    upper = [np.inf, float(x[-1]), span, np.inf, np.inf]
    fit_scale = float(np.max(np.abs(y)))
    if fit_scale <= 0:
        raise ValueError("Signal must have nonzero scale for Gaussian fitting")
    fit_y = y / fit_scale
    fit_sigma = (
        uncertainty / fit_scale if uncertainty is not None
        else covariance / fit_scale**2 if covariance is not None
        else None
    )
    fit_p0 = [p0[0] / fit_scale, p0[1], p0[2], p0[3] / fit_scale, p0[4] / fit_scale]
    fit_lower = [lower[0] / fit_scale, lower[1], lower[2], lower[3], lower[4] / fit_scale]
    fit_upper = [upper[0] / fit_scale, upper[1], upper[2], upper[3], upper[4] / fit_scale]

    if response is None:
        fit_model = _gaussian_background
    else:
        widths = quadrature.copy()
        def fit_model(wavelength, amplitude, fitted_center, sigma, intercept, slope):
            intrinsic = amplitude * np.exp(-0.5 * ((wavelength - fitted_center) / sigma) ** 2)
            convolved = (response @ (intrinsic * widths)) / widths
            return convolved + intercept + slope * (wavelength - fitted_center)

    try:
        scaled_params, scaled_covariance = curve_fit(
            fit_model, x, fit_y, p0=fit_p0, bounds=(fit_lower, fit_upper),
            sigma=fit_sigma, absolute_sigma=(uncertainty is not None or covariance is not None), maxfev=50000,
        )
        parameter_scales = np.array([fit_scale, 1.0, 1.0, fit_scale, fit_scale])
        params = scaled_params * parameter_scales
        covariance = scaled_covariance * np.outer(parameter_scales, parameter_scales)
        fitted = fit_model(x, *params)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        raise ValueError(f"Gaussian spectral-shape fit failed: {exc}") from exc
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(fitted)):
        raise ValueError("Gaussian spectral-shape fit returned non-finite results")
    residual = y - fitted
    dof = max(1, x.size - len(params))
    chi_square_raw = None if uncertainty is None else float(np.sum((residual / uncertainty) ** 2))
    chi_square = chi_square_raw if chi_square_raw is not None and np.isfinite(chi_square_raw) else None
    reduced_chi_square = None if chi_square is None else float(chi_square / dof)
    ss_total = float(np.sum((y - np.mean(y))**2))
    r_squared = float(1.0 - np.sum(residual**2) / ss_total) if ss_total > 0 else None
    parameter_uncertainties = [float(np.sqrt(max(float(v),0.0))) if np.isfinite(v) else None for v in np.diag(covariance)]

    moment_uncertainties = {}
    successful_draws = 0
    if uncertainty is not None or covariance is not None:
        rng = np.random.default_rng(58123)
        draws = []
        for _ in range(int(moment_mc_samples)):
            sampled_signal = (
                y + rng.normal(0.0, uncertainty)
                if uncertainty is not None
                else rng.multivariate_normal(y, covariance, check_valid="raise")
            )
            sampled_sigma = instrument_sigma_nm
            if instrument_sigma_nm is not None and instrument_sigma_uncertainty_nm is not None and instrument_sigma_uncertainty_nm > 0:
                sampled_sigma = max(np.finfo(float).tiny, float(rng.normal(instrument_sigma_nm, instrument_sigma_uncertainty_nm)))
            try:
                sample = _moment_statistics(x, sampled_signal, sampled_sigma, background_fraction, mask, explicit_background)
                draws.append(sample)
            except ValueError:
                continue
        successful_draws = len(draws)
        minimum_draws = max(20, int(np.ceil(0.8 * moment_mc_samples)))
        if successful_draws >= minimum_draws:
            for key, out_key in (
                ("center", "spectral_centroid_uncertainty_nm"),
                ("variance", "observed_variance_uncertainty_nm2"),
                ("sigma_observed", "observed_sigma_uncertainty_nm"),
                ("skewness", "spectral_skewness_uncertainty"),
                ("kurtosis", "spectral_excess_kurtosis_uncertainty"),
            ):
                moment_uncertainties[out_key] = float(np.std([item[key] for item in draws], ddof=1))
            if instrument_sigma_nm is not None and moments["resolved"] and all(item["resolved"] for item in draws):
                moment_uncertainties["instrument_corrected_sigma_uncertainty_nm"] = float(np.std([item["sigma_physical"] for item in draws], ddof=1))
        else:
            moment_uncertainties["moment_uncertainty_warning"] = "Too few valid signal/background Monte Carlo draws to estimate moment uncertainty."
    elif instrument_sigma_uncertainty_nm is not None and instrument_sigma_uncertainty_nm > 0 and resolved:
        moment_uncertainties["instrument_corrected_sigma_uncertainty_nm"] = float(
            instrument_sigma_nm / sigma_physical * instrument_sigma_uncertainty_nm
        )

    covariance_valid = bool(np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) > 0))
    covariance_condition = None
    if covariance_valid:
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        covariance_condition = float(np.linalg.cond(correlation))
        covariance_valid = bool(np.isfinite(covariance_condition) and covariance_condition <= 1e14)
    return {
        "wavelength_nm": x.copy(), "signal": y.copy(), "background": background,
        "background_subtracted_signal": corrected,
        "positive_spectral_weights": weights,
        "wavelength_quadrature_weights_nm": quadrature,
        "integrated_spectral_measure": spectral_measure,
        "integrated_positive_spectral_signal": area,
        "integrated_signed_spectral_signal": float(moments["signed_area"]),
        "integrated_negative_background_subtracted_signal": negative_area,
        "negative_area_fraction": float(moments["negative_fraction"]),
        "negative_area_fraction_removed_by_clipping": 0.0,
        "moment_weighting_method": "signed_background_subtracted_spectral_measure",
        "moment_clipping_bias_warning": "No clipping is applied; signed moments require positive integrated area and variance and remain sensitive to the supplied background model.",
        "background_estimation_method": moments["background_source"],
        "background_mask": mask,
        "spectral_centroid_nm": center,
        "observed_sigma_nm": sigma_observed,
        "observed_variance_nm2": variance_observed,
        "instrument_sigma_nm": None if instrument_sigma_nm is None else float(instrument_sigma_nm),
        "instrument_sigma_uncertainty_nm": None if instrument_sigma_uncertainty_nm is None else float(instrument_sigma_uncertainty_nm),
        "instrument_corrected_sigma_nm": sigma_physical,
        "instrument_corrected_variance_nm2": float(variance_physical) if variance_physical is not None and resolved else None,
        "instrument_width_resolved": resolved,
        "instrument_response_model": "general_measured_response_matrix" if response is not None else "gaussian_scalar_width",
        "spectral_skewness": float(skewness),
        "spectral_excess_kurtosis": float(excess_kurtosis),
        **moment_uncertainties,
        "moment_uncertainty_method": (
            "parametric Monte Carlo from full within-channel signal covariance and optional instrument-width uncertainty"
            if covariance is not None else
            "parametric Monte Carlo from independent pointwise signal uncertainty and optional instrument-width uncertainty"
            if uncertainty is not None else
            "instrument-width-only propagation"
            if instrument_sigma_uncertainty_nm is not None else
            "not_available"
        ),
        "signal_uncertainty_model": (
            "full_within_channel_spectral_covariance"
            if covariance is not None else
            "independent_pointwise_standard_uncertainties"
            if uncertainty is not None else
            "not_supplied"
        ),
        "moment_uncertainty_successful_draws": int(successful_draws),
        "gaussian_fit_parameters": {
            "amplitude": float(params[0]), "center_nm": float(params[1]),
            "sigma_nm": float(params[2]), "background_intercept": float(params[3]),
            "background_slope_per_nm": float(params[4]),
        },
        "gaussian_fit_parameter_uncertainties": {
            "amplitude": parameter_uncertainties[0], "center_nm": parameter_uncertainties[1],
            "sigma_nm": parameter_uncertainties[2], "background_intercept": parameter_uncertainties[3],
            "background_slope_per_nm": parameter_uncertainties[4],
        },
        "gaussian_fit_covariance_valid": covariance_valid,
        "gaussian_fit_covariance_correlation_condition_number": covariance_condition,
        "fitted_spectrum": fitted, "residual": residual,
        "degrees_of_freedom": int(dof), "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square, "r_squared": r_squared,
        "model": "signed spectral moments after background subtraction; Gaussian forward fit for shape diagnostics",
        "assumptions": [
            "Calibrated, strictly increasing wavelength axis",
            "A supplied Gaussian width or measured instrument-response matrix describes the spectral response",
            "A user-supplied background mask is preferred; spectral-edge estimation is used only when no mask is supplied",
            "Signed background-subtracted signal is used for moments without clipping negative residuals",
        ],
        "limitations": [
            "Spectral moments characterize measured wavelength distribution and are not a unique electron velocity distribution",
            "Signed moments can become unstable when the background-subtracted integrated area is small or the variance is non-positive",
            "A scalar Gaussian deconvolution is not applied to a general response matrix",
            "The edge-background fallback can remove real signal if the physical spectrum extends into both edges; supply background_mask to identify background-only samples",
            "No electron temperature or density is inferred by this generic descriptive helper",
            "Uncertainty estimates are Monte Carlo sampling uncertainties under the supplied pointwise noise model; shared systematic and correlated wavelength-bin errors are not included",
        ],
    }

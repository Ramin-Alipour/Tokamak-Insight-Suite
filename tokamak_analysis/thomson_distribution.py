"""Spectral-shape diagnostics linked to the canonical calibrated Thomson fit.

This module characterizes the measured spectral distribution after an explicit
background specification. A measured response matrix is used as a forward
operator in the Gaussian shape fit. Reported moments and widths describe the
measured spectrum; no scalar-width deconvolution, instrument-width subtraction,
or response correction is performed. The module does not infer an arbitrary
electron velocity distribution from moments; physical plasma parameters are
extracted by the canonical physical forward model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .thomson_identifiability import spectral_distribution_identifiability


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


def _wavelength_bin_widths(x):
    axis = np.asarray(x, dtype=float)
    if axis.ndim != 1 or axis.size < 2 or np.any(~np.isfinite(axis)) or np.any(np.diff(axis) <= 0.0):
        raise ValueError("wavelength axis must be finite, one-dimensional and strictly increasing")
    edges = np.empty(axis.size+1, dtype=float)
    edges[1:-1] = 0.5*(axis[:-1]+axis[1:])
    edges[0] = axis[0]-0.5*(axis[1]-axis[0])
    edges[-1] = axis[-1]+0.5*(axis[-1]-axis[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength axis must define positive bin widths")
    return widths


def _apply_response_matrix(response, spectrum, wavelength_nm):
    """Apply a bin-probability response while conserving integrated spectral power."""
    axis = np.asarray(wavelength_nm, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    matrix = np.asarray(response, dtype=float)
    if values.shape != axis.shape or matrix.shape != (axis.size, axis.size):
        raise ValueError("response matrix, spectrum and wavelength axis must share a square grid")
    widths = _wavelength_bin_widths(axis)
    return (matrix @ (values*widths))/widths


def _moment_statistics(
    x, y, background_fraction, background_mask, background_signal=None,
):
    if background_signal is not None:
        background = np.asarray(background_signal, dtype=float)
        if background.shape != x.shape or not np.all(np.isfinite(background)):
            raise ValueError("background_signal must be a finite array matching wavelength_nm")
        background_source = "explicit_fitted_background_signal"
    else:
        if background_mask is None:
            if background_fraction is None:
                raise ValueError("Specify background_signal, background_mask, or an explicit background_fraction; no spectral-edge background is assumed.")
            edge_width = float(background_fraction) * float(x[-1] - x[0])
            mask = (x <= x[0] + edge_width) | (x >= x[-1] - edge_width)
            background_source = "spectral_edges_assumed_background_by_wavelength_span"
        else:
            mask = background_mask
            background_source = "user_supplied_background_mask"
        minimum_background_samples = 2 if background_mask is None else 3
        if np.count_nonzero(mask) < minimum_background_samples:
            raise ValueError(f"background selection must include at least {minimum_background_samples} wavelength samples")
        slope, intercept = np.polyfit(x[mask], y[mask], 1)
        background = slope * x + intercept
    corrected = y - background
    positive = np.maximum(corrected, 0.0)
    negative = np.minimum(corrected, 0.0)
    # Use one finite-volume wavelength-bin measure consistently for areas
    # and all moments. Mixing trapezoidal areas with midpoint-bin moments
    # biases non-uniform calibrated wavelength grids.
    quadrature = _wavelength_bin_widths(x)
    positive_area = float(np.sum(positive * quadrature))
    negative_area = float(np.sum(negative * quadrature))
    signed_area = float(np.sum(corrected * quadrature))
    signal_scale = float(np.max(np.abs(y)))
    area_tolerance = 100.0 * np.finfo(float).eps * signal_scale * float(np.sum(quadrature))
    if not np.isfinite(signed_area) or signed_area <= area_tolerance:
        raise ValueError("No positive spectral area (signed integral) remains after background subtraction")
    measure = corrected * quadrature
    measure_sum = float(np.sum(measure))
    normalized = measure / measure_sum
    center = float(np.sum(normalized * x))
    centered = x - center
    variance = float(np.sum(normalized * centered**2))
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("Signed spectral measure has non-positive observed variance; inspect background selection and signal uncertainty")
    sigma_observed = float(np.sqrt(variance))
    skewness = float(np.sum(normalized * centered**3) / variance**1.5)
    kurtosis = float(np.sum(normalized * centered**4) / variance**2 - 3.0)
    total_absolute = positive_area + abs(negative_area)
    negative_fraction = abs(negative_area) / total_absolute if total_absolute > 0 else 0.0
    return {
        "background": background, "corrected": corrected, "positive": positive,
        "negative_area": negative_area, "positive_area": positive_area,
        "signed_area": signed_area, "quadrature": quadrature, "measure": measure,
        "center": center, "variance": variance, "sigma_observed": sigma_observed,
        "skewness": skewness,
        "kurtosis": kurtosis, "negative_fraction": negative_fraction,
        "background_source": background_source,
    }


def analyze_thomson_spectral_distribution(
    wavelength_nm,
    signal,
    *,
    instrument_response_matrix=None,
    signal_uncertainty=None,
    signal_covariance=None,
    background_fraction: float | None = None,
    background_mask=None,
    background_signal=None,
    background_signal_uncertainty=None,
    background_signal_covariance=None,
    signal_background_cross_covariance=None,
    moment_mc_samples: int = 200,
    physical_model: str | None = None,
    laser_wavelength_nm: float | None = None,
    scattering_angle_rad: float | None = None,
    laser_wavelength_uncertainty_nm: float | None = None,
    scattering_angle_uncertainty_rad: float | None = None,
    laser_angle_covariance_nm_rad: float = 0.0,
    physical_fit_options: dict | None = None,
    projected_evdf_options: dict | None = None,
) -> dict:
    """Characterize the measured, calibrated Thomson spectrum without refitting physics.

    This subanalysis reports observed wavelength-space moments and a Gaussian
    shape fit. Physical plasma parameters are extracted by the canonical Analysis 1
    forward model from the same measured spectrum; they are not accepted here as
    externally supplied inputs. The measured response matrix is used only as a
    forward operator in the shape fit; the measured spectrum is never deconvolved
    or otherwise corrected. Signed background-subtracted samples, including
    negative noise residuals, are retained for moment calculations.
    """
    x = _vector(wavelength_nm, "wavelength_nm")
    y = _vector(signal, "signal")
    if x.size != y.size:
        raise ValueError("wavelength_nm and signal must have equal lengths")
    if np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if instrument_response_matrix is None:
        raise ValueError("A measured instrument_response_matrix is required for the forward shape fit; no scalar-width correction is supported.")
    if background_fraction is not None and (isinstance(background_fraction, (bool, np.bool_)) or not isinstance(background_fraction, (int, float, np.integer, np.floating)) or not np.isfinite(background_fraction) or not (0.0 < background_fraction < 0.5)):
        raise ValueError("background_fraction must be None or an explicitly selected finite real number strictly between 0 and 0.5")
    if isinstance(moment_mc_samples, (bool, np.bool_)) or not isinstance(moment_mc_samples, (int, np.integer)) or moment_mc_samples < 20:
        raise ValueError("moment_mc_samples must be an integer >= 20")

    response = None
    if instrument_response_matrix is not None:
        raw_response = np.asarray(instrument_response_matrix)
        if raw_response.dtype.kind not in "iuf":
            raise ValueError("instrument_response_matrix must contain real numeric values, not booleans, strings or complex values")
        response = raw_response.astype(float, copy=False)
        if response.shape != (x.size, x.size) or not np.all(np.isfinite(response)) or np.any(response < 0):
            raise ValueError("instrument_response_matrix must be a finite non-negative square matrix on the wavelength grid")
        column_sums = np.sum(response, axis=0)
        if np.any(column_sums <= 0):
            raise ValueError("instrument_response_matrix must have positive column sums")
        if not np.allclose(column_sums, 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError(
                "measured instrument_response_matrix columns must already sum to one; "
                "the calibrated response operator is not renormalized"
            )

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
        explicit_background = _vector(background_signal, "background_signal")
        if explicit_background.size != x.size:
            raise ValueError("background_signal must have the same length as wavelength_nm")

    if background_signal_uncertainty is not None and background_signal_covariance is not None:
        raise ValueError("Supply background_signal_uncertainty or background_signal_covariance, not both.")
    background_uncertainty = None
    background_covariance_matrix = None
    background_covariance_cholesky = None
    if background_signal_uncertainty is not None or background_signal_covariance is not None:
        if explicit_background is None:
            raise ValueError("Background uncertainty is supported only with an explicitly supplied background_signal; mask/fraction backgrounds are refitted within each signal draw.")
        if background_signal_uncertainty is not None:
            background_uncertainty = _vector(background_signal_uncertainty, "background_signal_uncertainty")
            if background_uncertainty.size != x.size or np.any(background_uncertainty <= 0.0):
                raise ValueError("background_signal_uncertainty must be strictly positive and match wavelength_nm")
        else:
            raw_background_covariance = np.asarray(background_signal_covariance)
            if raw_background_covariance.dtype.kind not in "iuf":
                raise ValueError("background_signal_covariance must contain real numeric values")
            background_covariance_matrix = raw_background_covariance.astype(float, copy=False)
            if background_covariance_matrix.shape != (x.size, x.size) or not np.all(np.isfinite(background_covariance_matrix)):
                raise ValueError("background_signal_covariance must be a finite square matrix matching wavelength_nm")
            if not np.allclose(background_covariance_matrix, background_covariance_matrix.T, rtol=1e-10, atol=0.0):
                raise ValueError("background_signal_covariance must be symmetric")
            try:
                background_covariance_cholesky = np.linalg.cholesky(background_covariance_matrix)
            except np.linalg.LinAlgError as exc:
                raise ValueError("background_signal_covariance must be positive definite") from exc

    background_selection_count = sum(
        value is not None for value in (explicit_background, mask, background_fraction)
    )
    if background_selection_count == 0:
        raise ValueError(
            "Background is not inferred automatically. Supply background_signal, "
            "background_mask, or explicitly select background_fraction."
        )
    if background_selection_count > 1:
        raise ValueError(
            "Supply exactly one background specification: background_signal, "
            "background_mask, or background_fraction."
        )

    uncertainty = None
    signal_covariance_matrix = None
    covariance_cholesky = None
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
        signal_covariance_matrix = covariance_raw.astype(float, copy=False)
        if signal_covariance_matrix.shape != (y.size, y.size) or not np.all(np.isfinite(signal_covariance_matrix)):
            raise ValueError("signal_covariance must be a finite square matrix matching signal")
        if not np.allclose(signal_covariance_matrix, signal_covariance_matrix.T, rtol=1e-10, atol=0.0):
            raise ValueError("signal_covariance must be symmetric")
        try:
            covariance_cholesky = np.linalg.cholesky(signal_covariance_matrix)
        except np.linalg.LinAlgError as exc:
            raise ValueError("signal_covariance must be positive definite") from exc

    joint_signal_background_cholesky = None
    if signal_background_cross_covariance is not None:
        if explicit_background is None:
            raise ValueError(
                "signal_background_cross_covariance requires an explicitly supplied background_signal."
            )
        if signal_covariance_matrix is None and uncertainty is None:
            raise ValueError(
                "signal_background_cross_covariance requires signal_uncertainty or signal_covariance."
            )
        if background_covariance_matrix is None and background_uncertainty is None:
            raise ValueError(
                "signal_background_cross_covariance requires background_signal_uncertainty or background_signal_covariance."
            )
        raw_cross_covariance = np.asarray(signal_background_cross_covariance)
        if raw_cross_covariance.dtype.kind not in "iuf":
            raise ValueError("signal_background_cross_covariance must contain real numeric values")
        cross_covariance = raw_cross_covariance.astype(float, copy=False)
        if cross_covariance.shape != (x.size, x.size) or not np.all(np.isfinite(cross_covariance)):
            raise ValueError(
                "signal_background_cross_covariance must be a finite square wavelength-space matrix."
            )
        signal_covariance_for_joint = (
            signal_covariance_matrix if signal_covariance_matrix is not None
            else np.diag(uncertainty**2)
        )
        background_covariance_for_joint = (
            background_covariance_matrix if background_covariance_matrix is not None
            else np.diag(background_uncertainty**2)
        )
        joint_covariance = np.block([
            [signal_covariance_for_joint, cross_covariance],
            [cross_covariance.T, background_covariance_for_joint],
        ])
        joint_covariance = 0.5 * (joint_covariance + joint_covariance.T)
        try:
            joint_signal_background_cholesky = np.linalg.cholesky(joint_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "The joint signal/background covariance, including signal_background_cross_covariance, must be positive definite."
            ) from exc

    moments = _moment_statistics(x, y, background_fraction, mask, explicit_background)
    background = moments["background"]
    corrected = moments["corrected"]
    weights = moments["positive"]
    area = moments["positive_area"]
    negative_area = moments["negative_area"]
    center = moments["center"]
    variance_observed = moments["variance"]
    sigma_observed = moments["sigma_observed"]
    skewness = moments["skewness"]
    excess_kurtosis = moments["kurtosis"]
    quadrature = moments["quadrature"]
    spectral_measure = moments["measure"]

    peak = int(np.argmax(corrected))
    amplitude0 = float(max(corrected[peak], np.finfo(float).tiny))
    spacing = float(np.median(np.diff(x)))
    span = float(x[-1] - x[0])
    if explicit_background is not None:
        initial_background_slope = float(np.polyfit(x, background, 1)[0])
    elif mask is not None:
        initial_background_slope = float(np.polyfit(x[mask], y[mask], 1)[0])
    else:
        edge_count = max(3, int(np.ceil(x.size * background_fraction)))
        edge_indices = np.r_[np.arange(edge_count), np.arange(x.size-edge_count, x.size)]
        initial_background_slope = float(np.polyfit(
            x[edge_indices], y[edge_indices], 1
        )[0])
    p0 = [
        amplitude0, float(x[peak]), max(sigma_observed, spacing),
        float(np.interp(x[peak], x, background)), initial_background_slope,
    ]
    lower = [0.0, float(x[0]), max(spacing * 0.25, np.finfo(float).eps), -np.inf, -np.inf]
    upper = [np.inf, float(x[-1]), span, np.inf, np.inf]
    fit_scale = float(np.max(np.abs(y)))
    if fit_scale <= 0:
        raise ValueError("Signal must have nonzero scale for Gaussian fitting")
    fit_y = y / fit_scale
    fit_sigma = (
        uncertainty / fit_scale if uncertainty is not None
        else signal_covariance_matrix / fit_scale**2 if signal_covariance_matrix is not None
        else None
    )
    fit_p0 = [p0[0] / fit_scale, p0[1], p0[2], p0[3] / fit_scale, p0[4] / fit_scale]
    fit_lower = [lower[0] / fit_scale, lower[1], lower[2], lower[3], lower[4] / fit_scale]
    fit_upper = [upper[0] / fit_scale, upper[1], upper[2], upper[3], upper[4] / fit_scale]

    if response is None:
        fit_model = _gaussian_background
    else:
        def fit_model(wavelength, amplitude, fitted_center, sigma, intercept, slope):
            intrinsic = amplitude * np.exp(-0.5 * ((wavelength - fitted_center) / sigma) ** 2)
            convolved = _apply_response_matrix(response, intrinsic, x)
            return convolved + intercept + slope * (wavelength - fitted_center)

    try:
        scaled_params, scaled_covariance = curve_fit(
            fit_model, x, fit_y, p0=fit_p0, bounds=(fit_lower, fit_upper),
            sigma=fit_sigma, absolute_sigma=(uncertainty is not None or signal_covariance_matrix is not None), maxfev=50000,
        )
        # fit_model is evaluated in normalized-intensity space. Keep the
        # optimizer parameters for model evaluation and expose the converted
        # measured-unit parameters/covariance separately.
        parameter_scales = np.array([fit_scale, 1.0, 1.0, fit_scale, fit_scale])
        params = scaled_params * parameter_scales
        covariance = scaled_covariance * np.outer(parameter_scales, parameter_scales)
        fitted = fit_scale * fit_model(x, *scaled_params)
    except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
        raise ValueError(f"Gaussian spectral-shape fit failed: {exc}") from exc
    if not np.all(np.isfinite(params)) or not np.all(np.isfinite(fitted)):
        raise ValueError("Gaussian spectral-shape fit returned non-finite results")
    residual = y - fitted
    dof = max(1, x.size - len(params))
    if uncertainty is not None:
        chi_square_raw = float(np.sum((residual / uncertainty) ** 2))
    elif signal_covariance_matrix is not None:
        whitened_residual = np.linalg.solve(covariance_cholesky, residual)
        chi_square_raw = float(whitened_residual @ whitened_residual)
    else:
        chi_square_raw = None
    chi_square = chi_square_raw if chi_square_raw is not None and np.isfinite(chi_square_raw) else None
    reduced_chi_square = None if chi_square is None else float(chi_square / dof)
    ss_total = float(np.sum((y - np.mean(y))**2))
    r_squared = float(1.0 - np.sum(residual**2) / ss_total) if ss_total > 0 else None
    parameter_uncertainties = [float(np.sqrt(max(float(v),0.0))) if np.isfinite(v) else None for v in np.diag(covariance)]

    moment_uncertainties = {}
    successful_draws = 0
    if (uncertainty is not None or signal_covariance_matrix is not None
            or background_uncertainty is not None or background_covariance_matrix is not None):
        rng = np.random.default_rng(58123)
        draws = []
        for _ in range(int(moment_mc_samples)):
            sampled_background = explicit_background
            if joint_signal_background_cholesky is not None:
                joint_noise = joint_signal_background_cholesky @ rng.normal(size=2 * x.size)
                sampled_signal = y + joint_noise[:x.size]
                sampled_background = explicit_background + joint_noise[x.size:]
            else:
                sampled_signal = (
                    y + rng.normal(0.0, uncertainty)
                    if uncertainty is not None else
                    y + covariance_cholesky @ rng.normal(size=y.size)
                    if signal_covariance_matrix is not None else y.copy()
                )
                if background_uncertainty is not None:
                    sampled_background = explicit_background + rng.normal(0.0, background_uncertainty)
                elif background_covariance_matrix is not None:
                    sampled_background = explicit_background + background_covariance_cholesky @ rng.normal(size=x.size)
            try:
                sample = _moment_statistics(x, sampled_signal, background_fraction, mask, sampled_background)
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
        else:
            moment_uncertainties["moment_uncertainty_warning"] = "Too few valid signal/background Monte Carlo draws to estimate moment uncertainty."
    covariance_valid = bool(np.all(np.isfinite(covariance)) and np.all(np.diag(covariance) > 0))
    covariance_condition = None
    if covariance_valid:
        scale = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(scale, scale)
        covariance_condition = float(np.linalg.cond(correlation))
        covariance_valid = bool(np.isfinite(covariance_condition) and covariance_condition <= 1e14)
    result = {
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
        "background_identifiability_status": (
            "researcher_supplied_background"
            if explicit_background is not None or mask is not None
            else "edge_fraction_assumption_not_independently_identified"
        ),
        "background_confidence_note": (
            "Supplied background uncertainty is propagated with the supplied signal/background cross-covariance when present; otherwise signal and background errors are treated as independent. Possible signal contamination is not independently identified."
            if background_uncertainty is not None or background_covariance_matrix is not None
            else "The supplied background is treated as fixed; its uncertainty and possible signal contamination are not independently identified by this subanalysis."
        ),
        "background_uncertainty_model": (
            "independent_pointwise_background_signal_uncertainty" if background_uncertainty is not None else
            "joint_signal_background_covariance" if joint_signal_background_cholesky is not None else
            "independent_full_background_signal_covariance" if background_covariance_matrix is not None else
            "fixed_background_no_background_uncertainty_supplied"
        ),
        "background_uncertainty_supplied": bool(background_uncertainty is not None or background_covariance_matrix is not None),
        "background_mask": mask,
        "spectral_centroid_nm": center,
        "observed_sigma_nm": sigma_observed,
        "gaussian_width_interpretation": "Observed wavelength-space shape descriptor only; it is not a unique electron temperature or an EVDF width.",
        "gaussian_width_is_physical_temperature": False,
        "physical_temperature_source": "The separately selected canonical Thomson physical forward-model fit to the calibrated measured spectrum; not this Gaussian shape fit.",
        "observed_variance_nm2": variance_observed,
        "instrument_response_model": "measured_response_matrix_used_in_forward_shape_fit",
        "gaussian_fit_uncertainty_scope": "conditional_on_fixed_supplied_measured_response_matrix",
        "spectral_skewness": float(skewness),
        "spectral_excess_kurtosis": float(excess_kurtosis),
        **moment_uncertainties,
        "moment_uncertainty_method": (
            "parametric Monte Carlo from the correlated joint signal/background block covariance"
            if joint_signal_background_cholesky is not None else
            "parametric Monte Carlo from signal and independent fitted-background uncertainty"
            if (signal_covariance_matrix is not None or uncertainty is not None or background_uncertainty is not None or background_covariance_matrix is not None)
            else "not_available"
        ),
        "signal_uncertainty_model": (
            "full_within_channel_spectral_covariance"
            if signal_covariance_matrix is not None else
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
        "gaussian_fit_statistical_weighting_status": (
            "weighted_by_full_within_channel_spectral_covariance"
            if signal_covariance_matrix is not None else
            "weighted_by_independent_pointwise_measurement_uncertainty"
            if uncertainty is not None else
            "unweighted_least_squares_no_measurement_uncertainty_supplied"
        ),
        "gaussian_fit_covariance_interpretation": (
            "conditional_local_covariance_for_fixed_background_and_fixed_measured_response"
            if uncertainty is not None or signal_covariance_matrix is not None else
            "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"
        ),
        "moment_interpretation": (
            "Moments use signed background-subtracted spectral measure; negative residuals are retained, not clipped. They describe the measured wavelength distribution and are not an electron velocity distribution."
        ),
        "gaussian_fit_covariance_correlation_condition_number": covariance_condition,
        "fitted_spectrum": fitted, "residual": residual,
        "degrees_of_freedom": int(dof), "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square, "r_squared": r_squared,
        "model": "observed Thomson spectral moments and response-convolved Gaussian shape diagnostics",
        "parameter_identifiability": spectral_distribution_identifiability(),
        "assumptions": [
            "Calibrated, strictly increasing wavelength axis",
            "A measured instrument-response matrix is used only in the forward shape fit; measured input data are not corrected",
            "Background moments use an explicitly supplied fitted background, a user mask, or spectral-edge estimation in that order",
            "Signed background-subtracted signal is used for moments without clipping negative residuals",
        ],
        "limitations": [
            "Spectral moments characterize measured wavelength distribution and are not a unique electron velocity distribution",
            "Signed moments can become unstable when the background-subtracted integrated area is small or the variance is non-positive",
            "No instrument-width deconvolution or correction of the measured spectrum is performed",
            "The edge-background fallback can remove real signal if the physical spectrum extends into both edges; supply background_mask to identify background-only samples",
            "Physical plasma parameters are extracted separately by the canonical Analysis 1 forward fit from the same calibrated measured spectrum; moments are not used as a surrogate electron velocity distribution",
            "Pointwise-noise Monte Carlo treats wavelength samples as independent; supplied full within-channel covariance is sampled jointly. Signal/background cross-covariance is sampled jointly when supplied as a positive-definite block covariance; without it, the two are treated as independent. Possible signal contamination in the supplied background remains non-identifiable.",
        ],
    }

    if physical_model not in {None, "non_collective_maxwellian", "kappa", "relativistic_maxwell_juttner"}:
        raise ValueError("physical_model must be None, non_collective_maxwellian, kappa, or relativistic_maxwell_juttner")
    if physical_fit_options is not None and physical_model not in {
        "non_collective_maxwellian", "kappa", "relativistic_maxwell_juttner"
    }:
        raise ValueError(
            "physical_fit_options requires an explicitly selected physical_model"
        )

    # Kappa and relativistic Maxwell-Juttner require their own physical
    # forward models. Never map a Gaussian width to these non-Gaussian models.
    if physical_model in {"kappa", "relativistic_maxwell_juttner"}:
        if laser_wavelength_nm is None or scattering_angle_rad is None:
            raise ValueError("laser_wavelength_nm and scattering_angle_rad are required for kinetic physical inference")
        if not isinstance(physical_fit_options, dict):
            raise ValueError("physical_fit_options must provide physical_initial and the selected model's nuisance/uncertainty inputs")
        options = dict(physical_fit_options)
        reserved = {
            "physical_model", "instrument_response", "laser_wavelength_nm",
            "scattering_angle_rad", "wavelength_nm", "measured_spectrum",
        }
        conflicts = sorted(reserved.intersection(options))
        if conflicts:
            raise ValueError(
                "physical_fit_options cannot override the selected model, measured spectrum, "
                f"geometry or instrument response: {conflicts}"
            )

        # Preserve the uncertainty supplied to the spectral-distribution API
        # when handing the *same measured samples* to the native Kappa/MJ
        # physical forward fitter.  The fitter uses these as measurement
        # covariance inputs before adding any separately supplied discrepancy
        # or systematic covariance components.
        if uncertainty is not None:
            if "measurement_uncertainty" in options or "measurement_covariance" in options:
                raise ValueError(
                    "Top-level signal_uncertainty cannot be combined with measurement_uncertainty "
                    "or measurement_covariance in physical_fit_options."
                )
            options["measurement_uncertainty"] = uncertainty.copy()
        elif signal_covariance_matrix is not None:
            if "measurement_uncertainty" in options or "measurement_covariance" in options:
                raise ValueError(
                    "Top-level signal_covariance cannot be combined with measurement_uncertainty "
                    "or measurement_covariance in physical_fit_options."
                )
            options["measurement_covariance"] = signal_covariance_matrix.copy()

        from .thomson_kinetic_background import fit_noncollective_kinetic_thomson_with_background
        kinetic_fit = fit_noncollective_kinetic_thomson_with_background(
            x, y, physical_model=physical_model,
            laser_wavelength_nm=laser_wavelength_nm,
            scattering_angle_rad=scattering_angle_rad,
            instrument_response=response,
            **options,
        )
        result["physical_inference"] = {
            "selected_model": physical_model,
            "status": kinetic_fit.get("status", "completed_with_solver_status"),
            "reportable_as_primary_inference": kinetic_fit.get("primary_result_status") == "identified_from_measurement",
            "fit": kinetic_fit,
            "source_spectrum": "same_calibrated_measured_spectrum_as_spectral_distribution",
            "forward_model": "native_finite_bin_kinetic_thomson_model_with_measured_response_and_joint_detector_background",
            "gaussian_width_used_as_temperature": False,
            "limitations": [
                "Kappa is the finite-variance isotropic projected Student-t model, not a general anisotropic or relativistic Kappa EVDF.",
                "Relativistic Maxwell-Juttner uses the selected Thomson/Compton recoil contract and supplied polarization; it is not an unrestricted relativistic EVDF inverse.",
                "Identifiability and covariance are conditional on the supplied response, background basis, and uncertainty model.",
            ],
        }

    # Optional physical interpretation: only an explicitly selected
    # non-relativistic Maxwellian model may map the intrinsic Gaussian
    # forward-fit width to Te. The measured response is already included in
    # fit_model; no scalar instrument-width subtraction is performed.
    if physical_model == "non_collective_maxwellian":
        if laser_wavelength_nm is None or scattering_angle_rad is None:
            raise ValueError("laser_wavelength_nm and scattering_angle_rad are required for Maxwellian physical interpretation")
        laser = float(laser_wavelength_nm)
        angle = float(scattering_angle_rad)
        laser_unc = 0.0 if laser_wavelength_uncertainty_nm is None else float(laser_wavelength_uncertainty_nm)
        angle_unc = 0.0 if scattering_angle_uncertainty_rad is None else float(scattering_angle_uncertainty_rad)
        if not np.isfinite(laser) or laser <= 0.0:
            raise ValueError("laser_wavelength_nm must be positive and finite")
        if not np.isfinite(angle) or not 0.0 < angle < np.pi:
            raise ValueError("scattering_angle_rad must lie strictly between 0 and pi")
        if not np.isfinite(laser_unc) or laser_unc < 0.0 or not np.isfinite(angle_unc) or angle_unc < 0.0:
            raise ValueError("laser/angle uncertainties must be finite and non-negative")
        geometry_cross_covariance = float(laser_angle_covariance_nm_rad)
        if not np.isfinite(geometry_cross_covariance):
            raise ValueError("laser_angle_covariance_nm_rad must be finite")
        geometry_covariance = np.array([[laser_unc**2, geometry_cross_covariance], [geometry_cross_covariance, angle_unc**2]])
        geometry_eigenvalues = np.linalg.eigvalsh(geometry_covariance)
        geometry_scale = max(float(np.max(np.abs(geometry_eigenvalues))), np.finfo(float).tiny)
        if geometry_eigenvalues[0] < -1e-12 * geometry_scale:
            raise ValueError("laser/angle calibration covariance must be positive semidefinite; require |covariance| <= laser_uncertainty * angle_uncertainty")
        q = 2.0 * np.sin(angle / 2.0)
        light_speed = 299792458.0
        electron_rest_energy_eV = 510998.95069
        intrinsic_sigma = float(params[2])
        fitted_center = float(params[1])
        thermal_ratio = intrinsic_sigma / (laser * q)
        temperature_candidate = electron_rest_energy_eV * thermal_ratio**2
        flow_candidate = light_speed * (1.0 - fitted_center / laser) / q
        theta_e = temperature_candidate / electron_rest_energy_eV
        beta_flow = abs(flow_candidate) / light_speed
        # Propagate the fitted (center,sigma) covariance and independent
        # laser/angle calibration errors through the analytic Jacobian.
        physical_covariance = None
        physical_uncertainties = None
        if covariance_valid and np.all(np.isfinite(covariance[np.ix_([1, 2], [1, 2])])):
            cot_half = 1.0 / np.tan(angle / 2.0)
            grad_temperature = np.array([
                0.0, 2.0 * temperature_candidate / intrinsic_sigma,
                -2.0 * temperature_candidate / laser,
                -temperature_candidate * cot_half,
            ])
            grad_flow = np.array([
                -light_speed / (laser * q), 0.0,
                light_speed * fitted_center / (laser**2 * q),
                -0.5 * flow_candidate * cot_half,
            ])
            input_covariance = np.zeros((4, 4), dtype=float)
            input_covariance[:2, :2] = covariance[np.ix_([1, 2], [1, 2])]
            input_covariance[2, 2] = laser_unc**2
            input_covariance[3, 3] = angle_unc**2
            input_covariance[2, 3] = geometry_cross_covariance
            input_covariance[3, 2] = geometry_cross_covariance
            jacobian = np.vstack((grad_temperature, grad_flow))
            physical_covariance = jacobian @ input_covariance @ jacobian.T
            physical_covariance = 0.5 * (physical_covariance + physical_covariance.T)
            physical_uncertainties = {
                "electron_temperature_eV": float(np.sqrt(max(physical_covariance[0, 0], 0.0))),
                "electron_velocity_projection_m_s": float(np.sqrt(max(physical_covariance[1, 1], 0.0))),
            }
        valid_domain = bool(theta_e < 0.01 and beta_flow < 0.1)
        result["physical_inference"] = {
            "selected_model": physical_model,
            "electron_temperature_candidate_eV": float(temperature_candidate) if valid_domain else None,
            "electron_velocity_projection_candidate_m_s": float(flow_candidate) if valid_domain else None,
            "parameter_covariance_order": ["electron_temperature_eV", "electron_velocity_projection_m_s"],
            "parameter_covariance": physical_covariance if valid_domain else None,
            "parameter_standard_uncertainties": physical_uncertainties if valid_domain else None,
            "validity_parameters": {"theta_e": float(theta_e), "abs_beta_flow": float(beta_flow)},
            "validity_limits": {"theta_e_strictly_less_than": 0.01, "abs_beta_flow_strictly_less_than": 0.1},
            "status": "conditional_gaussian_maxwellian_candidate" if valid_domain else "rejected_outside_declared_nonrelativistic_validity_domain",
            "reportable_as_primary_inference": False,
            "formula_temperature_eV": "T_e = (m_e*c^2/e) * (sigma_lambda_intrinsic/(lambda_0*2*sin(theta/2)))^2",
            "formula_velocity_projection_m_s": "v_q = c*(1 - lambda_centroid/lambda_0)/(2*sin(theta/2))",
            "uncertainty_method": "analytic_Jacobian_propagation_from_intrinsic_Gaussian_fit_covariance_and_independent_laser/angle_calibration_uncertainties",
            "assumptions": [
                "explicit non-collective Maxwellian selection",
                "intrinsic Gaussian width from forward convolution with the supplied measured response matrix",
                "non-relativistic elastic Thomson wavelength mapping",
                "laser wavelength and scattering-angle calibration errors are independent of fitted spectral parameters",
                "laser wavelength/scattering-angle calibration cross-covariance is included when supplied",
                "no model-discrepancy or background-parameter cross-covariance is included in this conditional conversion",
            ],
            "limitations": [
                "This is a conditional diagnostic candidate, not a replacement for the canonical joint physical Thomson inference.",
                "Gaussian spectral shape alone cannot identify Kappa index, anisotropy, multiple populations or a general EVDF.",
                "No primary physical result is reported outside the declared asymptotic validity screens.",
            ],
        }
    # The Maxwellian option must reach the canonical measured-spectrum
    # forward fitter when the caller supplies the spatial coordinate and fit
    # controls. The Gaussian-width conversion above remains a clearly labelled
    # diagnostic candidate; it is never promoted to the primary measurement.
    if physical_model == "non_collective_maxwellian" and physical_fit_options is not None:
        if laser_wavelength_nm is None or scattering_angle_rad is None:
            raise ValueError(
                "laser_wavelength_nm and scattering_angle_rad are required for canonical Maxwellian inference"
            )
        if not isinstance(physical_fit_options, dict):
            raise ValueError("physical_fit_options must be a mapping for canonical Maxwellian inference")
        options = dict(physical_fit_options)
        if "radius_m" not in options:
            raise ValueError(
                "physical_fit_options must include radius_m for the canonical measured-spectrum Maxwellian fit"
            )
        radius = options.pop("radius_m")
        reserved = {
            "wavelength_nm", "spectra", "laser_wavelength_nm",
            "scattering_angle_rad", "instrument_response", "model",
            "signal_uncertainty", "joint_spectral_covariance",
        }
        conflicts = sorted(reserved.intersection(options))
        if conflicts:
            raise ValueError(
                "physical_fit_options cannot override measured spectrum, selected model, geometry, "
                f"response or top-level uncertainty inputs: {conflicts}"
            )
        radius_array = np.asarray(radius)
        if radius_array.dtype.kind not in "iuf" or radius_array.ndim != 0:
            raise ValueError("physical_fit_options.radius_m must be one finite real scalar for the single spectrum")
        radius_value = float(radius_array)
        if not np.isfinite(radius_value):
            raise ValueError("physical_fit_options.radius_m must be finite")
        if uncertainty is not None and signal_covariance_matrix is not None:
            raise ValueError("internal error: signal uncertainty and covariance are mutually exclusive")
        from .thomson_inference import analyze_thomson_spectra as canonical_fit
        canonical_kwargs = dict(
            laser_wavelength_nm=float(laser_wavelength_nm),
            scattering_angle_rad=float(scattering_angle_rad),
            instrument_response=response,
            model="non_collective_maxwellian",
            **options,
        )
        if uncertainty is not None:
            canonical_kwargs["signal_uncertainty"] = uncertainty.reshape(1, -1)
        elif signal_covariance_matrix is not None:
            canonical_kwargs["joint_spectral_covariance"] = signal_covariance_matrix
        canonical_result = canonical_fit(
            x, y.reshape(1, -1), np.asarray([radius_value]), **canonical_kwargs
        )
        channel = canonical_result.get("channels", [{}])[0]
        primary_parameters = {
            item.get("parameter")
            for item in canonical_result.get("result_classification", {})
            .get("channels", [{}])[0].get("primary", [])
        }
        required_primary = {
            "electron_temperature_eV", "electron_velocity_projection_m_s"
        }
        result["physical_inference"] = {
            "selected_model": physical_model,
            "status": (
                "identified_from_measured_spectrum"
                if required_primary.issubset(primary_parameters)
                else "canonical_fit_completed_with_conditional_or_nonidentifiable_parameters"
            ),
            "reportable_as_primary_inference": required_primary.issubset(primary_parameters),
            "fit": canonical_result,
            "channel_fit": channel,
            "source_spectrum": "same_calibrated_measured_spectrum_as_spectral_distribution",
            "forward_model": "canonical_multichannel_optical_thomson_maxwellian_forward_fit",
            "gaussian_width_used_as_temperature": False,
            "gaussian_width_role": "independent_descriptive_cross_check_only",
            "fit_configuration": {
                "radius_m": radius_value,
                "background_model": canonical_kwargs.get("background_model", "linear"),
                "measurement_uncertainty_source": (
                    "pointwise_signal_uncertainty" if uncertainty is not None else
                    "full_within_channel_spectral_covariance" if signal_covariance_matrix is not None else
                    "not_supplied"
                ),
            },
            "limitations": [
                "The canonical physical fit uses the measured spectrum and selected detector-background nuisance basis; it does not use Gaussian moments or width as a substitute likelihood.",
                "Absolute electron density remains conditional on absolute scattered-power calibration and complete optical geometry.",
                "A single scattering view identifies only the q-parallel flow projection, not an unrestricted three-dimensional EVDF.",
            ],
        }

    elif physical_model is None:
        result["physical_inference"] = {
            "selected_model": None,
            "status": "not_requested",
            "reportable_as_primary_inference": False,
            "reason": "Spectral moments and Gaussian width remain descriptive unless a physical model and scattering geometry are explicitly selected.",
        }

    # SD-02: connect the same measured spectrum to the native response-aware
    # projected-EVDF inverse. This solves y = R A_q f_q + b, with f_q >= 0
    # and integral(f_q dv) = 1. A single view identifies only the q-parallel
    # marginal, never an unrestricted three-dimensional EVDF.
    if projected_evdf_options is not None:
        if not isinstance(projected_evdf_options, dict):
            raise ValueError("projected_evdf_options must be a mapping")
        if laser_wavelength_nm is None or scattering_angle_rad is None:
            raise ValueError(
                "laser_wavelength_nm and scattering_angle_rad are required for projected EVDF inversion"
            )
        options = dict(projected_evdf_options)
        reserved = {
            "wavelength_nm", "signal", "laser_wavelength_nm",
            "scattering_angle_rad", "instrument_response_matrix",
            "background_signal", "signal_uncertainty", "signal_covariance",
            "background_uncertainty", "background_covariance",
            "signal_background_cross_covariance",
        }
        conflicts = sorted(reserved.intersection(options))
        if conflicts:
            raise ValueError(
                "projected_evdf_options cannot override the measured spectrum, geometry, "
                f"response, background or top-level covariance inputs: {conflicts}"
            )
        from .thomson_evdf import reconstruct_thomson_evdf
        evdf_kwargs = {
            "wavelength_nm": x,
            "signal": y,
            "laser_wavelength_nm": float(laser_wavelength_nm),
            "scattering_angle_rad": float(scattering_angle_rad),
            "instrument_response_matrix": response,
            "background_signal": background.copy(),
            **options,
        }
        if uncertainty is not None:
            evdf_kwargs["signal_uncertainty"] = uncertainty.copy()
        elif signal_covariance_matrix is not None:
            evdf_kwargs["signal_covariance"] = signal_covariance_matrix.copy()
        if background_uncertainty is not None:
            evdf_kwargs["background_uncertainty"] = background_uncertainty.copy()
        elif background_covariance_matrix is not None:
            evdf_kwargs["background_covariance"] = background_covariance_matrix.copy()
        if signal_background_cross_covariance is not None:
            evdf_kwargs["signal_background_cross_covariance"] = cross_covariance.copy()
        result["projected_evdf_inference"] = reconstruct_thomson_evdf(**evdf_kwargs)
        result["projected_evdf_inference"]["source_spectrum"] = (
            "same_calibrated_measured_spectrum_as_spectral_distribution"
        )
        result["projected_evdf_inference"]["inference_scope"] = (
            "one_dimensional_q_parallel_velocity_marginal_only"
        )
        result["projected_evdf_inference"]["gaussian_width_used_as_evdf"] = False
        # Cross-check the reconstructed q-parallel EVDF against measured spectral moments.
        # In the non-relativistic Thomson limit: lambda=lambda0*(1-q*v_parallel/c).
        # This is a forward-model consistency diagnostic, never a Gaussian Te estimator.
        inverse = result["projected_evdf_inference"]
        velocity_grid = np.asarray(inverse["velocity_parallel_m_s"], dtype=float)
        projected_pdf = np.asarray(inverse["evdf_parallel_s_per_m"], dtype=float)
        from scipy.integrate import trapezoid
        if (velocity_grid.ndim != 1 or projected_pdf.shape != velocity_grid.shape
                or not np.all(np.isfinite(velocity_grid))
                or not np.all(np.isfinite(projected_pdf)) or np.any(projected_pdf < 0.0)):
            raise ValueError("Projected EVDF returned an invalid probability density")
        evdf_norm = float(trapezoid(projected_pdf, velocity_grid))
        if not np.isfinite(evdf_norm) or evdf_norm <= 0.0:
            raise ValueError("Projected EVDF has non-positive integrated probability")
        normalized_pdf = projected_pdf / evdf_norm
        evdf_mean = float(trapezoid(velocity_grid * normalized_pdf, velocity_grid))
        evdf_variance = float(trapezoid((velocity_grid - evdf_mean)**2 * normalized_pdf, velocity_grid))
        q = 2.0 * np.sin(float(scattering_angle_rad) / 2.0)
        c_light = 299792458.0
        if float(np.max(np.abs(velocity_grid))) >= 0.1 * c_light:
            closure = {"status": "outside_nonrelativistic_doppler_closure_domain",
                       "comparison_performed": False, "evdf_mean_velocity_parallel_m_s": evdf_mean,
                       "evdf_variance_velocity_parallel_m2_s2": evdf_variance}
        else:
            predicted_centroid = float(laser_wavelength_nm) * (1.0 - q * evdf_mean / c_light)
            predicted_variance = (float(laser_wavelength_nm) * q / c_light)**2 * evdf_variance
            closure = {
                "status": "conditional_moment_closure",
                "formula": "lambda=lambda_0*(1-q*v_parallel/c), q=2*sin(theta/2)",
                "evdf_mean_velocity_parallel_m_s": evdf_mean,
                "evdf_variance_velocity_parallel_m2_s2": evdf_variance,
                "evdf_predicted_spectral_centroid_nm": predicted_centroid,
                "evdf_predicted_spectral_sigma_nm": float(np.sqrt(max(predicted_variance, 0.0))),
                "measured_spectral_centroid_nm": float(center),
                "measured_spectral_sigma_nm": float(sigma_observed),
                "centroid_closure_residual_nm": float(center - predicted_centroid),
                "variance_closure_residual_nm2": float(variance_observed - predicted_variance),
                "comparison_performed": True,
                "interpretation": "Compare conditionally after accounting for background, finite spectral window, instrument response and regularization; disagreement is a model/data consistency warning, not grounds to force a Gaussian temperature.",
            }
        inverse["spectral_moment_closure"] = closure

    return result

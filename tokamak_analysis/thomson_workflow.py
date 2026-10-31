"""Thomson spectral-inference entry point and downstream calibrated-profile utilities.

The primary Analysis 1 input is measured, calibrated multi-channel Thomson spectra.
The legacy profile-summary function remains available only for downstream use when
profiles have been obtained independently.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import PchipInterpolator, make_smoothing_spline

from .thomson_scattering import (
    build_profile,
    profile_summary,
    _covariance_is_positive_semidefinite,
)


def _gradient_roundoff_threshold(derivative_matrix, values):
    """Estimate a pointwise floating-point cancellation bound for each derivative.

    The bound follows the absolute terms in the local finite-difference dot
    product. It is therefore local to each coordinate and profile scale, rather
    than being controlled by the largest gradient elsewhere in the profile.
    """
    absolute_terms = np.abs(derivative_matrix) @ np.abs(values)
    eps = np.finfo(float).eps
    sample_count = values.size
    gamma = (sample_count * eps) / (1.0 - sample_count * eps)
    return gamma * absolute_terms


def _scale_length(value, gradient, gradient_threshold):
    """Return -value/gradient where the local derivative exceeds round-off."""
    out = np.full(value.shape, np.nan, dtype=float)
    valid = (
        (value > 0)
        & np.isfinite(gradient)
        & np.isfinite(gradient_threshold)
        & (np.abs(gradient) > gradient_threshold)
    )
    out[valid] = -value[valid] / gradient[valid]
    return out


def _gradient_uncertainty(derivative_matrix, uncertainty, covariance=None):
    if uncertainty is None:
        return None, None
    input_covariance = np.diag(np.square(uncertainty)) if covariance is None else np.asarray(covariance, dtype=float)
    gradient_covariance = derivative_matrix @ input_covariance @ derivative_matrix.T
    variance = np.diag(gradient_covariance)
    standard_uncertainty = np.sqrt(np.maximum(variance, 0.0))
    covariance_value_gradient = np.diag(input_covariance @ derivative_matrix.T)
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


def _scale_length_uncertainty(
    value, gradient, value_uncertainty, gradient_uncertainty, covariance,
    gradient_threshold,
):
    out = np.full(value.shape, np.nan, dtype=float)
    valid = (
        (value > 0)
        & np.isfinite(gradient)
        & np.isfinite(gradient_threshold)
        & (np.abs(gradient) > gradient_threshold)
    )
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
    density_covariance_m3_2=None,
    temperature_covariance_eV2=None,
):
    """Calculate profile summaries and gradients with pointwise or full covariance.

    Full covariance matrices are ordered like the supplied profile points and are
    reordered together with the radius/density/temperature arrays.
    """
    radius_raw = np.asarray(radius_m)
    if radius_raw.ndim != 1 or radius_raw.dtype.kind not in "iuf":
        raise ValueError("radius_m must be a one-dimensional real numeric array")
    radius_input = radius_raw.astype(float, copy=False)
    if not np.all(np.isfinite(radius_input)):
        raise ValueError("radius_m must contain only finite values")
    order = np.argsort(radius_input)

    def validate_profile_covariance(value, name, supplied_uncertainty):
        if value is None:
            return None, supplied_uncertainty
        covariance_raw = np.asarray(value)
        if covariance_raw.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values, not booleans or strings")
        covariance = covariance_raw.astype(float, copy=False)
        expected = (radius_input.size, radius_input.size)
        if covariance.shape != expected or not np.all(np.isfinite(covariance)):
            raise ValueError(f"{name} must be a finite square covariance matrix matching the profile length")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError(f"{name} must be symmetric")
        if not _covariance_is_positive_semidefinite(covariance):
            raise ValueError(f"{name} must be positive semidefinite")
        diagonal_sigma = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        if supplied_uncertainty is not None:
            supplied = np.asarray(supplied_uncertainty, dtype=float)
            if supplied.shape != diagonal_sigma.shape or not np.allclose(
                supplied**2, np.diag(covariance), rtol=1e-6, atol=1e-24
            ):
                raise ValueError(f"Pointwise uncertainty must match the diagonal of {name}")
        else:
            supplied_uncertainty = diagonal_sigma
        return covariance[np.ix_(order, order)], supplied_uncertainty

    density_covariance, density_uncertainty_m3 = validate_profile_covariance(
        density_covariance_m3_2, "density_covariance_m3_2", density_uncertainty_m3
    )
    temperature_covariance, temperature_uncertainty_ev = validate_profile_covariance(
        temperature_covariance_eV2, "temperature_covariance_eV2", temperature_uncertainty_ev
    )
    p = build_profile(
        radius_m, density_m3, temperature_ev,
        density_uncertainty_m3, temperature_uncertainty_ev,
    )
    r, ne, te = p.radius_m, p.density_m3, p.temperature_ev
    edge_order = 2 if r.size >= 3 else 1
    derivative_matrix = np.gradient(np.eye(r.size), r, axis=0, edge_order=edge_order)
    dne_dr = derivative_matrix @ ne
    dte_dr = derivative_matrix @ te
    density_gradient_threshold = _gradient_roundoff_threshold(derivative_matrix, ne)
    temperature_gradient_threshold = _gradient_roundoff_threshold(derivative_matrix, te)
    if p.density_uncertainty_m3 is not None:
        density_gradient_sigma, _ = _gradient_uncertainty(
            derivative_matrix, p.density_uncertainty_m3, density_covariance
        )
        density_gradient_threshold = np.maximum(density_gradient_threshold, 2.0 * density_gradient_sigma)
    if p.temperature_uncertainty_ev is not None:
        temperature_gradient_sigma, _ = _gradient_uncertainty(
            derivative_matrix, p.temperature_uncertainty_ev, temperature_covariance
        )
        temperature_gradient_threshold = np.maximum(temperature_gradient_threshold, 2.0 * temperature_gradient_sigma)
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
        "density_scale_length_m": _scale_length(ne, dne_dr, density_gradient_threshold),
        "temperature_scale_length_m": _scale_length(te, dte_dr, temperature_gradient_threshold),
        "summary": profile_summary(p),
        "scale_length_gradient_significance_threshold_sigma": 2.0,
        "notes": [
            "Input must be calibrated Thomson-scattering profile data.",
            "When pointwise uncertainties are supplied, a scale length is reported only where the local gradient exceeds twice its propagated 1-sigma uncertainty; otherwise the numerical round-off threshold is used.",
            "Full density and temperature covariance matrices are propagated through the gradient operator when supplied; otherwise pointwise uncertainties are treated as independent.",
            "NaN scale lengths mark zero, unresolved or statistically insignificant gradients or profile values.",
            "Peak-location uncertainty is not estimated by this workflow.",
        ],
    }
    if p.density_uncertainty_m3 is not None:
        sne = p.density_uncertainty_m3
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, sne, density_covariance)
        result["density_uncertainty_m3"] = sne.copy()
        result["density_input_covariance_m3_2"] = (
            density_covariance.copy() if density_covariance is not None else np.diag(sne**2)
        )
        result["density_gradient_uncertainty_m-4"] = grad_unc
        density_input_covariance = density_covariance if density_covariance is not None else np.diag(sne**2)
        result["density_gradient_covariance_m-8"] = (
            derivative_matrix @ density_input_covariance @ derivative_matrix.T
        )
        result["density_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            ne, dne_dr, sne, grad_unc, cov
        )
        result["density_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            ne, dne_dr, sne, grad_unc, cov, density_gradient_threshold
        )
    if p.temperature_uncertainty_ev is not None:
        ste = p.temperature_uncertainty_ev
        grad_unc, cov = _gradient_uncertainty(derivative_matrix, ste, temperature_covariance)
        result["temperature_uncertainty_eV"] = ste.copy()
        result["temperature_input_covariance_eV2"] = (
            temperature_covariance.copy() if temperature_covariance is not None else np.diag(ste**2)
        )
        result["temperature_gradient_uncertainty_eV_m-1"] = grad_unc
        temperature_input_covariance = temperature_covariance if temperature_covariance is not None else np.diag(ste**2)
        result["temperature_gradient_covariance_eV2_m-2"] = (
            derivative_matrix @ temperature_input_covariance @ derivative_matrix.T
        )
        result["temperature_normalized_gradient_uncertainty_m-1"] = _normalized_gradient_uncertainty(
            te, dte_dr, ste, grad_unc, cov
        )
        result["temperature_scale_length_uncertainty_m"] = _scale_length_uncertainty(
            te, dte_dr, ste, grad_unc, cov, temperature_gradient_threshold
        )
    return result


def analyze_thomson_spectra(*args, **kwargs):
    """Primary Analysis 1 entry point: infer plasma parameters from measured spectra.

    The spectral forward model is maintained in thomson_inference to keep the
    legacy calibrated-profile summary utilities below available as downstream
    operations without confusing their input contract with the primary analysis.
    """
    from .thomson_inference import analyze_thomson_spectra as _infer
    return _infer(*args, **kwargs)



def _reconstruct_inferred_parameter_profile(
    radius_m, values, *, uncertainty=None, covariance=None, positive=False,
    method="both", points=200, mc_samples=200, random_seed=18473,
):
    """Reconstruct one spectrum-inferred physical profile without extrapolation.

    PCHIP preserves local shape; a GCV smoothing spline supplies a smooth
    alternative. Supplied cross-channel covariance is sampled jointly. When only
    pointwise uncertainties exist, independent channel errors are an explicit
    fallback rather than an implied full covariance model.
    """
    radius = np.asarray(radius_m, dtype=float)
    value = np.asarray(values, dtype=float)
    if radius.ndim != 1 or value.ndim != 1 or radius.size != value.size or radius.size < 2:
        raise ValueError("A profile requires at least two paired one-dimensional channel values.")
    if not np.all(np.isfinite(radius)) or not np.all(np.isfinite(value)):
        raise ValueError("Profile coordinates and inferred values must be finite.")
    order = np.argsort(radius)
    radius, value = radius[order], value[order]
    if np.any(np.diff(radius) <= 0):
        raise ValueError("Spatial profile reconstruction requires unique, strictly increasing channel coordinates.")
    if positive and np.any(value <= 0):
        raise ValueError("Positive physical profile values must be strictly positive.")
    if method not in {"both", "pchip", "smoothing_spline"}:
        raise ValueError("profile_reconstruction_method must be 'both', 'pchip' or 'smoothing_spline'.")
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or not 20 <= points <= 5000:
        raise ValueError("profile_reconstruction_points must be an integer between 20 and 5000.")
    if isinstance(mc_samples, (bool, np.bool_)) or not isinstance(mc_samples, (int, np.integer)) or mc_samples < 20:
        raise ValueError("profile_reconstruction_mc_samples must be an integer >= 20.")

    sigma = None
    if uncertainty is not None:
        sigma = np.asarray(uncertainty, dtype=float)
        if sigma.shape != value.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("Profile point uncertainties must be finite, non-negative and match the channel count.")
        sigma = sigma[order]
    cov = None
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (value.size, value.size) or not np.all(np.isfinite(cov)):
            raise ValueError("Profile covariance must be a finite square matrix matching the channel count.")
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=0.0) or not _covariance_is_positive_semidefinite(cov):
            raise ValueError("Profile covariance must be symmetric and positive semidefinite.")
        cov = cov[np.ix_(order, order)]
        diagonal_sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if sigma is not None and not np.allclose(sigma**2, diagonal_sigma**2, rtol=1e-6, atol=1e-24):
            raise ValueError("Point uncertainties must match the covariance diagonal.")
        sigma = diagonal_sigma
    elif sigma is not None:
        cov = np.diag(sigma**2)

    grid = np.linspace(float(radius[0]), float(radius[-1]), int(points))
    selected = ("pchip", "smoothing_spline") if method == "both" else (method,)
    output = {
        "input_radius_m": radius.copy(),
        "input_values": value.copy(),
        "reconstructed_radius_m": grid,
        "methods": {},
        "covariance_model": (
            "full_supplied_cross_channel_covariance"
            if covariance is not None
            else "independent_pointwise_uncertainties"
            if sigma is not None
            else "not_available"
        ),
        "extrapolation": "forbidden; reconstruction is restricted to the measured channel-coordinate interval",
    }

    def fit_profile(sample, selected_method):
        if selected_method == "pchip":
            fit = PchipInterpolator(radius, sample, extrapolate=False)
        else:
            if sigma is None or not np.any(sigma > 0):
                weights = None
            else:
                sigma_floor = max(
                    float(np.max(sigma)) * 1e-12,
                    np.sqrt(np.finfo(float).tiny),
                )
                weights = 1.0 / np.maximum(sigma, sigma_floor)
            fit = make_smoothing_spline(radius, sample, w=weights, lam=None)
        return np.asarray(fit(grid), dtype=float), np.asarray(fit.derivative()(grid), dtype=float)

    rng = np.random.default_rng(random_seed)
    draws = None
    if cov is not None and np.any(np.diag(cov) > 0):
        try:
            draws = rng.multivariate_normal(value, cov, size=int(mc_samples), check_valid="raise")
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Profile uncertainty covariance could not be sampled: {exc}") from exc

    for selected_method in selected:
        if selected_method == "smoothing_spline" and radius.size < 5:
            output["methods"][selected_method] = {
                "status": "insufficient_spatial_channels",
                "required_minimum_channels": 5,
                "available_channels": int(radius.size),
            }
            continue
        reconstructed, gradient = fit_profile(value, selected_method)
        item = {
            "status": "completed",
            "values": reconstructed,
            "gradient_per_m": gradient,
            "fit_method": (
                "shape-preserving piecewise cubic Hermite interpolation (PCHIP)"
                if selected_method == "pchip"
                else "generalized-cross-validation smoothing spline"
            ),
            "uncertainty_available": False,
        }
        if draws is not None:
            sampled_profiles, sampled_gradients = [], []
            for sample in draws:
                if positive and np.any(sample <= 0):
                    continue
                try:
                    sample_profile, sample_gradient = fit_profile(sample, selected_method)
                    if np.all(np.isfinite(sample_profile)) and np.all(np.isfinite(sample_gradient)):
                        sampled_profiles.append(sample_profile)
                        sampled_gradients.append(sample_gradient)
                except (ValueError, RuntimeError, FloatingPointError):
                    continue
            minimum = max(20, int(np.ceil(0.8 * mc_samples)))
            item["uncertainty_mc_successful_samples"] = len(sampled_profiles)
            item["uncertainty_mc_requested_samples"] = int(mc_samples)
            if len(sampled_profiles) >= minimum:
                profile_draws = np.asarray(sampled_profiles)
                gradient_draws = np.asarray(sampled_gradients)
                item.update({
                    "standard_uncertainty": np.std(profile_draws, axis=0, ddof=1),
                    "lower_95_percentile": np.percentile(profile_draws, 2.5, axis=0),
                    "upper_95_percentile": np.percentile(profile_draws, 97.5, axis=0),
                    "gradient_standard_uncertainty_per_m": np.std(gradient_draws, axis=0, ddof=1),
                    "uncertainty_available": True,
                })
            else:
                item["uncertainty_warning"] = "Too few valid covariance draws to estimate reconstructed-profile uncertainty."
        output["methods"][selected_method] = item
    return output


def reconstruct_thomson_profile_from_spectra(
    *args, profile_reconstruction_method="both",
    profile_reconstruction_points=200, profile_reconstruction_mc_samples=200,
    **kwargs,
):
    """Infer physical parameters from calibrated spectra, then reconstruct profiles.

    All physical values originate from the measured spectral forward fit. The
    reconstruction never accepts pre-derived profile input and never extrapolates
    beyond the measured channel coordinates.
    """
    result = analyze_thomson_spectra(*args, **kwargs)
    radii = np.asarray(result["radius_m"], dtype=float)
    n_channels = radii.size
    external_covariance = result.get("collective_external_parameter_covariance")
    external_order = result.get("collective_external_parameter_covariance_order") or []
    joint_summary = result.get("joint_inference") or {}
    joint_physical_covariance = joint_summary.get(
        "total_physical_parameter_covariance",
        joint_summary.get("physical_parameter_covariance"),
    )
    density_covariance = result.get("electron_density_joint_covariance_m6")
    if density_covariance is None:
        density_covariance = result.get("electron_density_external_parameter_covariance_m6")

    def covariance_for(labels, fallback=None):
        if external_covariance is not None and external_order:
            indices = [external_order.index(label) if label in external_order else None for label in labels]
            if all(index is not None for index in indices):
                matrix = np.asarray(external_covariance, dtype=float)
                return matrix[np.ix_(indices, indices)]
        return fallback

    profiles = {}
    te_covariance = None
    flow_covariance = None
    if joint_physical_covariance is not None:
        joint = np.asarray(joint_physical_covariance, dtype=float)
        if joint.shape == (2*n_channels, 2*n_channels):
            te_indices = np.arange(0, 2*n_channels, 2)
            flow_indices = te_indices + 1
            te_covariance = joint[np.ix_(te_indices, te_indices)]
            flow_covariance = joint[np.ix_(flow_indices, flow_indices)]
    te_labels = [f"channel_{i}_electron_temperature_eV" for i in range(n_channels)]
    flow_labels = [f"channel_{i}_electron_velocity_projection_m_s" for i in range(n_channels)]
    te_covariance = covariance_for(te_labels, te_covariance)
    flow_covariance = covariance_for(flow_labels, flow_covariance)
    if density_covariance is None:
        density_covariance = covariance_for(
            [f"channel_{i}_electron_density_m3" for i in range(n_channels)]
        )

    candidates = [
        ("electron_temperature_eV", "electron_temperature_profile_eV",
         "electron_temperature_uncertainty_profile_eV", te_covariance, True),
        ("electron_density_m3", "electron_density_profile_m3",
         "electron_density_uncertainty_profile_m3", density_covariance, True),
        ("electron_velocity_projection_m_s", "electron_velocity_projection_profile_m_s",
         "electron_velocity_projection_uncertainty_profile_m_s", flow_covariance, False),
    ]
    if result.get("model") in {"non_collective_kappa", "collective_kappa"}:
        kappa_values = np.asarray([channel.get("kappa_index", np.nan) for channel in result["channels"]], dtype=float)
        kappa_uncertainties = np.asarray([
            channel.get("kappa_index_uncertainty", np.nan) for channel in result["channels"]
        ], dtype=float)
        kappa_covariance = (
            np.diag(np.square(kappa_uncertainties))
            if np.all(np.isfinite(kappa_uncertainties)) else None
        )
        if result.get("model") == "collective_kappa":
            kappa_covariance = covariance_for(
                [f"channel_{i}_kappa_index" for i in range(n_channels)],
                kappa_covariance,
            )
        candidates.append((
            "electron_kappa_index", None, None, kappa_covariance, False,
        ))
    if result.get("model") in {"collective_maxwellian", "collective_kappa"}:
        alpha_values = np.asarray([
            channel.get("scattering_parameter_alpha", np.nan) for channel in result["channels"]
        ], dtype=float)
        candidates.append(("scattering_parameter_alpha", None, None, None, True))
    spectral_profile_inputs = {}
    power_values = np.asarray([
        channel.get(
            "integrated_scattered_power_W",
            channel.get("integrated_spectral_signal_input_units_nm", np.nan),
        ) for channel in result["channels"]
    ], dtype=float)
    if np.all(np.isfinite(power_values)):
        power_sigma = np.asarray([
            channel.get(
                "integrated_scattered_power_uncertainty_W",
                channel.get("integrated_spectral_signal_uncertainty_input_units_nm", np.nan),
            ) for channel in result["channels"]
        ], dtype=float)
        power_profile_name = (
            "integrated_scattered_power_W"
            if result.get("absolute_scattered_power_available")
            else "integrated_spectral_signal_input_units_nm"
        )
        candidates.append((
            power_profile_name, None, None,
            np.diag(power_sigma**2) if np.all(np.isfinite(power_sigma)) else None,
            True,
        ))
    # Spatially resolved spectral-shape descriptors are also useful outputs.
    # They remain explicitly wavelength-space diagnostics, not substitutes for
    # the physical temperature/flow inversion.
    distribution_keys = (
        ("spectral_centroid_nm", "spectral_centroid_nm", "spectral_centroid_uncertainty_nm", False),
        ("observed_sigma_nm", "observed_sigma_nm", "observed_sigma_uncertainty_nm", True),
        ("spectral_skewness", "spectral_skewness", "spectral_skewness_uncertainty", False),
        ("spectral_excess_kurtosis", "spectral_excess_kurtosis", "spectral_excess_kurtosis_uncertainty", False),
    )
    for profile_name, metric_key, uncertainty_key, positive in distribution_keys:
        metric_values = np.asarray([
            channel.get("spectral_distribution", {}).get(metric_key, np.nan)
            for channel in result["channels"]
        ], dtype=float)
        metric_uncertainties = np.asarray([
            channel.get("spectral_distribution", {}).get(uncertainty_key, np.nan)
            for channel in result["channels"]
        ], dtype=float)
        candidates.append((
            profile_name, None, None, None, positive,
        ))
        # Keep the channel vectors in a private lookup used by the common
        # reconstruction loop below.
        spectral_profile_inputs[profile_name] = (
            metric_values,
            metric_uncertainties if np.all(np.isfinite(metric_uncertainties)) else None,
        )
    for name, value_key, uncertainty_key, covariance, positive in candidates:
        if name == "electron_kappa_index":
            values = kappa_values
            uncertainty_values = np.sqrt(np.maximum(np.diag(covariance), 0.0)) if covariance is not None else None
        elif name == "scattering_parameter_alpha":
            values = alpha_values
            uncertainty_values = None
        elif name in {"integrated_scattered_power_W", "integrated_spectral_signal_input_units_nm"}:
            values = power_values
            uncertainty_values = power_sigma if np.all(np.isfinite(power_sigma)) else None
        elif name in spectral_profile_inputs:
            values, uncertainty_values = spectral_profile_inputs[name]
        else:
            values = np.asarray(result[value_key], dtype=float)
            uncertainty_values = np.asarray(result.get(uncertainty_key), dtype=float)
        if values.shape != (n_channels,) or not np.all(np.isfinite(values)):
            profiles[name] = {
                "status": "not_identifiable_for_all_channels",
                "reason": "The selected physical model did not identify this parameter at every measured channel.",
            }
            continue
        if uncertainty_values is not None and (
            uncertainty_values.shape != (n_channels,) or not np.all(np.isfinite(uncertainty_values))
        ):
            uncertainty_values = None
        try:
            profiles[name] = _reconstruct_inferred_parameter_profile(
                radii, values, uncertainty=uncertainty_values, covariance=covariance,
                positive=positive, method=profile_reconstruction_method,
                points=profile_reconstruction_points,
                mc_samples=profile_reconstruction_mc_samples,
                random_seed=18473 + len(profiles),
            )
        except ValueError as exc:
            profiles[name] = {"status": "not_reconstructed", "reason": str(exc)}

    if result.get("model") in {"collective_maxwellian", "collective_kappa"}:
        species_count = len(result["channels"][0].get("ion_velocity_projection_m_s", []))
        for species_index in range(species_count):
            name = f"ion_{species_index}_velocity_projection_m_s"
            values = np.asarray([
                channel["ion_velocity_projection_m_s"][species_index]
                for channel in result["channels"]
            ], dtype=float)
            labels = [f"channel_{i}_ion_{species_index}_velocity_projection_m_s" for i in range(n_channels)]
            covariance = covariance_for(labels)
            uncertainties = [
                channel.get("ion_velocity_projection_uncertainty_m_s", [None] * species_count)[species_index]
                for channel in result["channels"]
            ]
            uncertainty_values = None if any(v is None for v in uncertainties) else np.asarray(uncertainties, dtype=float)
            try:
                profiles[name] = _reconstruct_inferred_parameter_profile(
                    radii, values, uncertainty=uncertainty_values, covariance=covariance,
                    positive=False, method=profile_reconstruction_method,
                    points=profile_reconstruction_points,
                    mc_samples=profile_reconstruction_mc_samples,
                    random_seed=18473 + 10 + species_index,
                )
            except ValueError as exc:
                profiles[name] = {"status": "not_reconstructed", "reason": str(exc)}
    result["spatial_profile_reconstruction"] = {
        "input_origin": "physical parameters inferred from measured calibrated Thomson spectra",
        "profiles": profiles,
        "channel_order": "sorted by calibrated radius within each reconstructed profile",
        "methods_requested": profile_reconstruction_method,
        "no_extrapolation": True,
        "limitations": [
            "Reconstruction is a downstream spatial representation of parameters first inferred from the measured spectra.",
            "PCHIP and GCV smoothing splines are numerical profile representations, not additional plasma-physics inference models.",
            "Uncertainty bands are conditional on supplied/inferred channel covariance; when only pointwise uncertainties exist, channel independence is assumed.",
            "No equilibrium mapping, flux-surface mapping or tomographic inversion is inferred from radius alone.",
        ],
    }
    result["registered_subanalysis"] = "Thomson spatial profile reconstruction from calibrated spectra"
    return result

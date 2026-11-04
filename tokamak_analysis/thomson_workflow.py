"""Canonical Thomson measured-spectrum-to-physics workflows."""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator, make_smoothing_spline

from .thomson_scattering import _covariance_is_positive_semidefinite
from .formula_library import thomson_velocity_projection_relativistic_exact

def analyze_thomson_spectra(*args, **kwargs):
    """Infer physical plasma parameters directly from measured calibrated spectra.

    All public Thomson workflows start from measured spectra and the selected
    physical forward model. Pre-derived density/temperature profile inputs are
    not accepted by this module.
    """
    from .thomson_inference import analyze_thomson_spectra as _infer
    return _infer(*args, **kwargs)



def _generalized_least_squares_smoothing_spline(
    radius_m, values, covariance, evaluation_radius_m,
):
    """Fit a natural cubic smoothing spline with a full correlated-error GLS objective.

    The supplied covariance is used as-is; no correlations are inferred. The
    smoothing parameter is selected by generalized cross-validation using the
    GLS hat-matrix effective degrees of freedom.
    """
    radius = np.asarray(radius_m, dtype=float)
    observed = np.asarray(values, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    scale = max(float(np.max(np.abs(observed))), np.finfo(float).tiny)
    y = observed / scale
    cov = covariance / scale**2
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (cov + cov.T))
    tolerance = np.finfo(float).eps * max(cov.shape) * float(eigenvalues[-1])
    if np.any(eigenvalues <= tolerance):
        raise ValueError(
            "Full-covariance GLS smoothing requires a positive-definite supplied "
            "cross-channel covariance; singular covariance is not silently regularized."
        )
    inverse_covariance = (eigenvectors / eigenvalues) @ eigenvectors.T

    n = radius.size
    basis = CubicSpline(radius, np.eye(n), axis=0, bc_type="natural")
    # Integrate the exact piecewise-polynomial roughness integrand with
    # Gauss-Legendre quadrature on every measured-coordinate interval.
    nodes, weights = np.polynomial.legendre.leggauss(max(8, n))
    penalty = np.zeros((n, n), dtype=float)
    for left, right in zip(radius[:-1], radius[1:]):
        points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        second_derivatives = np.asarray(basis(points, 2), dtype=float)
        penalty += 0.5 * (right - left) * (
            second_derivatives.T @ (weights[:, None] * second_derivatives)
        )
    penalty = 0.5 * (penalty + penalty.T)

    covariance_scale = float(np.median(eigenvalues))
    span = float(radius[-1] - radius[0])
    lambda_scale = span**3 / covariance_scale
    candidates = lambda_scale * np.logspace(-8.0, 8.0, 81)
    best = None
    for smoothing_parameter in candidates:
        system = inverse_covariance + smoothing_parameter * penalty
        try:
            coefficients = np.linalg.solve(system, inverse_covariance @ y)
            hat_matrix = np.linalg.solve(system, inverse_covariance)
        except np.linalg.LinAlgError:
            continue
        residual = y - coefficients
        effective_df = float(np.trace(hat_matrix))
        denominator = 1.0 - effective_df / n
        if denominator <= 1e-10:
            continue
        generalized_residual = float(residual @ inverse_covariance @ residual)
        score = (generalized_residual / n) / denominator**2
        if np.isfinite(score) and (best is None or score < best["score"]):
            best = {
                "score": score,
                "smoothing_parameter": float(smoothing_parameter),
                "effective_degrees_of_freedom": effective_df,
                "generalized_residual_sum_squares": generalized_residual,
                "system_condition_number": float(np.linalg.cond(system)),
                "coefficients": coefficients,
            }
    if best is None:
        raise ValueError("GLS smoothing-parameter selection failed for the supplied covariance.")
    coefficients = best.pop("coefficients")
    fitted = np.asarray(basis(evaluation_radius_m), dtype=float) @ coefficients
    gradient = np.asarray(basis(evaluation_radius_m, 1), dtype=float) @ coefficients
    diagnostics = {
        "selection_method": "generalized_cross_validation_with_full_GLS_covariance",
        "covariance_model": "researcher_supplied_full_cross_channel_covariance",
        "smoothing_parameter": best["smoothing_parameter"],
        "gcv_score": best["score"],
        "effective_degrees_of_freedom": best["effective_degrees_of_freedom"],
        "generalized_residual_sum_squares": best["generalized_residual_sum_squares"],
        "system_condition_number": best["system_condition_number"],
        "covariance_regularization": "none; non-positive-definite covariance is rejected",
    }
    return fitted * scale, gradient * scale, diagnostics


def _reconstruct_inferred_parameter_profile(
    radius_m, values, *, uncertainty=None, covariance=None, coordinate_covariance=None,
    coordinate_value_cross_covariance=None,
    positive=False, method="both", points=200, mc_samples=200, random_seed=18473,
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
    if np.any(np.diff(radius) <= 0):
        raise ValueError(
            "Spatial profile reconstruction requires the supplied channel coordinates "
            "to be unique and strictly increasing; input order is preserved and never repaired."
        )
    if positive and np.any(value <= 0):
        raise ValueError("Positive physical profile values must be strictly positive.")
    if method not in {"both", "pchip", "smoothing_spline"}:
        raise ValueError("profile_reconstruction_method must be 'both', 'pchip' or 'smoothing_spline'.")
    if isinstance(points, (bool, np.bool_)) or not isinstance(points, (int, np.integer)) or not 20 <= points <= 5000:
        raise ValueError("profile_reconstruction_points must be an integer between 20 and 5000.")
    if isinstance(mc_samples, (bool, np.bool_)) or not isinstance(mc_samples, (int, np.integer)) or mc_samples < 20:
        raise ValueError("profile_reconstruction_mc_samples must be an integer >= 20.")

    full_covariance_supplied = covariance is not None
    sigma = None
    if uncertainty is not None:
        sigma = np.asarray(uncertainty, dtype=float)
        if sigma.shape != value.shape or not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError("Profile point uncertainties must be finite, non-negative and match the channel count.")
    cov = None
    if covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (value.size, value.size) or not np.all(np.isfinite(cov)):
            raise ValueError("Profile covariance must be a finite square matrix matching the channel count.")
        if not np.allclose(cov, cov.T, rtol=1e-10, atol=0.0) or not _covariance_is_positive_semidefinite(cov):
            raise ValueError("Profile covariance must be symmetric and positive semidefinite.")
        diagonal_sigma = np.sqrt(np.maximum(np.diag(cov), 0.0))
        if sigma is not None and not np.allclose(sigma**2, diagonal_sigma**2, rtol=1e-6, atol=1e-24):
            raise ValueError("Point uncertainties must match the covariance diagonal.")
        sigma = diagonal_sigma
    elif sigma is not None:
        cov = np.diag(sigma**2)

    coordinate_cov = None
    if coordinate_covariance is not None:
        coordinate_cov = np.asarray(coordinate_covariance, dtype=float)
        if (coordinate_cov.shape != (radius.size, radius.size)
                or not np.all(np.isfinite(coordinate_cov))
                or not np.allclose(coordinate_cov, coordinate_cov.T, rtol=1e-10, atol=0.0)
                or not _covariance_is_positive_semidefinite(coordinate_cov)):
            raise ValueError("Coordinate covariance must be finite, symmetric, positive semidefinite and match the channel count.")
    coordinate_value_cross = None
    if coordinate_value_cross_covariance is not None:
        if coordinate_cov is None or cov is None:
            raise ValueError(
                "Coordinate/value cross-covariance requires both profile_coordinate_covariance_m2 "
                "and a profile-value covariance (or pointwise uncertainty)."
            )
        coordinate_value_cross = np.asarray(coordinate_value_cross_covariance, dtype=float)
        if (coordinate_value_cross.shape != (radius.size, value.size)
                or not np.all(np.isfinite(coordinate_value_cross))):
            raise ValueError("Coordinate/value cross-covariance must be finite with shape (n_channels,n_channels).")
        joint_covariance = np.block([
            [coordinate_cov, coordinate_value_cross],
            [coordinate_value_cross.T, cov],
        ])
        if (not np.allclose(joint_covariance, joint_covariance.T, rtol=1e-10, atol=0.0)
                or not _covariance_is_positive_semidefinite(joint_covariance)):
            raise ValueError(
                "The joint coordinate/profile-value covariance block must be positive semidefinite."
            )
    else:
        joint_covariance = None

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
        "coordinate_uncertainty_model": (
            "joint_researcher_supplied_coordinate_and_profile_value_covariance"
            if coordinate_value_cross is not None
            else "researcher_supplied_full_coordinate_covariance_independent_of_profile_values"
            if coordinate_cov is not None else "coordinates_treated_as_exact"
        ),
        "coordinate_value_cross_covariance_used": coordinate_value_cross is not None,
        "extrapolation": "forbidden; reconstruction is restricted to the measured channel-coordinate interval",
    }

    spline_diagnostics = {}

    def fit_profile(sample, selected_method, sample_radius=None):
        sample_radius = radius if sample_radius is None else np.asarray(sample_radius, dtype=float)
        if sample_radius.shape != radius.shape or np.any(np.diff(sample_radius) <= 0):
            raise ValueError("Sampled spatial coordinates must remain strictly increasing.")
        if grid[0] < sample_radius[0] or grid[-1] > sample_radius[-1]:
            raise ValueError("Sampled spatial coordinates do not cover the fixed reconstruction grid.")
        if selected_method == "pchip":
            fit = PchipInterpolator(sample_radius, sample, extrapolate=False)
            reconstructed = np.asarray(fit(grid), dtype=float)
            gradient = np.asarray(fit.derivative()(grid), dtype=float)
        elif selected_method == "smoothing_spline" and full_covariance_supplied:
            if positive:
                if np.any(sample <= 0.0):
                    raise ValueError("Positive profile samples must remain strictly positive.")
                log_covariance = cov / np.outer(value, value)
                log_profile, log_gradient, diagnostics = _generalized_least_squares_smoothing_spline(
                    sample_radius, np.log(sample), log_covariance, grid,
                )
                reconstructed = np.exp(log_profile)
                gradient = reconstructed * log_gradient
            else:
                reconstructed, gradient, diagnostics = _generalized_least_squares_smoothing_spline(
                    sample_radius, sample, cov, grid,
                )
            spline_diagnostics[selected_method] = diagnostics
        elif positive:
            # An unconstrained smoothing spline in physical coordinates can
            # overshoot below zero between positive measured channels. Fit the
            # logarithm instead so density/temperature-like profiles remain
            # strictly positive throughout the reconstructed interval.
            if np.any(sample <= 0.0):
                raise ValueError("Positive profile samples must remain strictly positive.")
            log_sample = np.log(sample)
            if sigma is None or not np.any(sigma > 0):
                weights = None
            else:
                relative_sigma = sigma / sample
                sigma_floor = max(
                    float(np.max(relative_sigma)) * 1e-12,
                    np.sqrt(np.finfo(float).tiny),
                )
                weights = 1.0 / np.maximum(relative_sigma, sigma_floor)
            fit = make_smoothing_spline(sample_radius, log_sample, w=weights, lam=None)
            log_profile = np.asarray(fit(grid), dtype=float)
            reconstructed = np.exp(log_profile)
            gradient = reconstructed * np.asarray(fit.derivative()(grid), dtype=float)
        else:
            if sigma is None or not np.any(sigma > 0):
                weights = None
            else:
                sigma_floor = max(
                    float(np.max(sigma)) * 1e-12,
                    np.sqrt(np.finfo(float).tiny),
                )
                weights = 1.0 / np.maximum(sigma, sigma_floor)
            fit = make_smoothing_spline(sample_radius, sample, w=weights, lam=None)
            reconstructed = np.asarray(fit(grid), dtype=float)
            gradient = np.asarray(fit.derivative()(grid), dtype=float)
        return reconstructed, gradient

    rng = np.random.default_rng(random_seed)
    draws = None
    coordinate_draws = None
    has_profile_variance = cov is not None and np.any(np.diag(cov) > 0)
    has_coordinate_variance = coordinate_cov is not None and np.any(np.diag(coordinate_cov) > 0)
    has_cross_covariance = coordinate_value_cross is not None and np.any(coordinate_value_cross != 0)
    if has_profile_variance or has_coordinate_variance or has_cross_covariance:
        try:
            if joint_covariance is not None:
                joint_mean = np.r_[radius, value]
                joint_draws = rng.multivariate_normal(
                    joint_mean, joint_covariance, size=int(mc_samples), check_valid="raise"
                )
                coordinate_draws = joint_draws[:, :radius.size]
                draws = joint_draws[:, radius.size:]
            else:
                draws = (
                    rng.multivariate_normal(value, cov, size=int(mc_samples), check_valid="raise")
                    if has_profile_variance
                    else np.repeat(value[None, :], int(mc_samples), axis=0)
                )
                coordinate_draws = (
                    rng.multivariate_normal(radius, coordinate_cov, size=int(mc_samples), check_valid="raise")
                    if has_coordinate_variance
                    else np.repeat(radius[None, :], int(mc_samples), axis=0)
                )
        except (ValueError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"Joint profile/coordinate uncertainty covariance could not be sampled: {exc}") from exc

    for selected_method in selected:
        if selected_method == "smoothing_spline" and radius.size < 5:
            output["methods"][selected_method] = {
                "status": "insufficient_spatial_channels",
                "required_minimum_channels": 5,
                "available_channels": int(radius.size),
            }
            continue
        reconstructed, gradient = fit_profile(value, selected_method, radius)
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
        if selected_method in spline_diagnostics:
            item["smoothing_diagnostics"] = spline_diagnostics[selected_method]
        if draws is not None:
            sampled_profiles, sampled_gradients = [], []
            for sample, sample_radius in zip(draws, coordinate_draws):
                if positive and np.any(sample <= 0):
                    continue
                if np.any(np.diff(sample_radius) <= 0):
                    continue
                try:
                    sample_profile, sample_gradient = fit_profile(sample, selected_method, sample_radius)
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
    profile_coordinate_covariance_m2=None,
    profile_coordinate_value_cross_covariance=None,
    joint_spatial_spectral=False, spatial_mixing_matrix=None,
    spatial_mixing_covariance=None, spatial_mixing_mc_samples=200,
    spatial_mixing_mc_seed=84622,
    spatial_regularization_strength=0.0, profile_grid_radius_m=None,
    spatial_background_model="linear", spatial_background_degree=2,
    spatial_background_interior_knots=3, spatial_background_templates=None,
    **kwargs,
):
    """Infer physical parameters from calibrated spectra, then reconstruct profiles.

    All physical values originate from the measured spectral forward fit. The
    reconstruction never accepts pre-derived profile input and never extrapolates
    beyond the measured channel coordinates.
    """
    selected_model = kwargs.get("model")
    if spatial_mixing_covariance is not None and spatial_mixing_matrix is None:
        raise ValueError("spatial_mixing_covariance requires spatial_mixing_matrix")
    if joint_spatial_spectral and (profile_coordinate_covariance_m2 is not None
                                  or profile_coordinate_value_cross_covariance is not None):
        raise ValueError(
            "profile_coordinate_covariance_m2 is supported by downstream profile "
            "reconstruction, not by the raw joint spatial-spectral optimizer"
        )
    if joint_spatial_spectral:
        if spatial_mixing_matrix is None:
            raise ValueError("spatial_mixing_matrix is required when joint_spatial_spectral=True")
        if kwargs.get("instrument_response") is None:
            raise ValueError(
                "instrument_response is required when joint_spatial_spectral=True; "
                "supply the measured response operator or explicitly supply an identity "
                "matrix only for spectra already corrected for the instrument response."
            )
        if selected_model not in (None, "non_collective_maxwellian", "non_collective_kappa", "non_collective_relativistic_maxwell_juttner"):
            raise ValueError("joint spatial-spectral reconstruction requires a registered non-collective optical Thomson model")
        if len(args) < 3:
            raise ValueError("joint spatial-spectral reconstruction requires wavelength_nm, measured spectra and radius_m")
        allowed_joint = {"model", "laser_wavelength_nm", "scattering_angle_rad",
                         "instrument_response", "laser_spectrum_wavelength_nm",
                         "laser_spectrum_density_per_nm", "signal_uncertainty", "joint_spectral_covariance",
                         "model_discrepancy_covariance",
                         "regularization_sensitivity_strengths",
                         "kappa_index", "fit_kappa_index", "incident_polarization",
                         "radial_order", "angular_order", "maximum_recoil_correction_fraction",
                         "include_compton_recoil"}
        unsupported = set(kwargs)-allowed_joint
        if unsupported:
            raise ValueError(f"unsupported joint spatial-spectral inputs: {sorted(unsupported)}")
        from .thomson_joint_spatial_spectral import reconstruct_joint_thomson_spatial_spectra
        fit_kwargs = {
            "laser_wavelength_nm": kwargs["laser_wavelength_nm"],
            "scattering_angle_rad": kwargs["scattering_angle_rad"],
            "instrument_response": kwargs.get("instrument_response"),
            "laser_spectrum_wavelength_nm": kwargs.get("laser_spectrum_wavelength_nm"),
            "laser_spectrum_density_per_nm": kwargs.get("laser_spectrum_density_per_nm"),
            "signal_uncertainty": kwargs.get("signal_uncertainty"),
            "joint_spectral_covariance": kwargs.get("joint_spectral_covariance"),
            "model_discrepancy_covariance": kwargs.get("model_discrepancy_covariance"),
            "regularization_sensitivity_strengths": kwargs.get("regularization_sensitivity_strengths"),
            "regularization_strength": spatial_regularization_strength,
            "background_model": spatial_background_model,
            "background_degree": spatial_background_degree,
            "background_interior_knots": spatial_background_interior_knots,
            "background_templates": spatial_background_templates,
            "physical_model": {
                "non_collective_kappa": "kappa",
                "non_collective_relativistic_maxwell_juttner": "relativistic_maxwell_juttner",
            }.get(selected_model, "maxwellian"),
            "kappa_index": kwargs.get("kappa_index"),
            "fit_kappa_index": kwargs.get("fit_kappa_index", False),
            "incident_polarization": kwargs.get("incident_polarization"),
            "radial_order": kwargs.get("radial_order", 24),
            "angular_order": kwargs.get("angular_order", 12),
            "maximum_recoil_correction_fraction": kwargs.get("maximum_recoil_correction_fraction", 0.01),
            "include_compton_recoil": kwargs.get("include_compton_recoil", False),
        }
        result = reconstruct_joint_thomson_spatial_spectra(
            args[0], args[1], args[2], spatial_mixing_matrix, **fit_kwargs
        )
        if spatial_mixing_covariance is None:
            return result

        # Propagate uncertainty in the supplied spatial forward operator by
        # complete nonlinear refits. This is an external-operator contribution,
        # not a calibration or an attempt to infer the PSF from the plasma data.
        raw_h = np.asarray(spatial_mixing_matrix)
        if raw_h.dtype.kind not in "iuf" or raw_h.ndim != 2:
            raise ValueError("spatial_mixing_matrix must be a real two-dimensional array")
        h = raw_h.astype(float, copy=False)
        raw_cov = np.asarray(spatial_mixing_covariance)
        if raw_cov.dtype.kind not in "iuf":
            raise ValueError("spatial_mixing_covariance must contain real numeric values")
        mixing_cov = raw_cov.astype(float, copy=False)
        expected_shape = (h.size, h.size)
        if mixing_cov.shape != expected_shape or not np.all(np.isfinite(mixing_cov)):
            raise ValueError(f"spatial_mixing_covariance must be finite with shape {expected_shape}")
        scale = max(float(np.max(np.abs(mixing_cov))), np.finfo(float).tiny)
        if not np.allclose(mixing_cov, mixing_cov.T, rtol=1e-10, atol=1e-12 * scale):
            raise ValueError("spatial_mixing_covariance must be symmetric")
        mixing_cov = 0.5 * (mixing_cov + mixing_cov.T)
        eigenvalues = np.linalg.eigvalsh(mixing_cov)
        eig_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if eigenvalues[0] < -1e-10 * eig_scale:
            raise ValueError("spatial_mixing_covariance must be positive semidefinite")
        if (isinstance(spatial_mixing_mc_samples, (bool, np.bool_))
                or not isinstance(spatial_mixing_mc_samples, (int, np.integer))
                or spatial_mixing_mc_samples < 20):
            raise ValueError("spatial_mixing_mc_samples must be an integer >= 20")
        if (isinstance(spatial_mixing_mc_seed, (bool, np.bool_))
                or not isinstance(spatial_mixing_mc_seed, (int, np.integer))):
            raise ValueError("spatial_mixing_mc_seed must be an integer")
        base_covariance = result.get("parameter_covariance")
        if (not result.get("primary_outputs_reportable_as_identified")
                or base_covariance is None):
            result["spatial_mixing_uncertainty"] = {
                "supplied": True, "propagated": False,
                "status": "not_propagated_nominal_solution_not_data_identified",
                "covariance_shape": list(mixing_cov.shape),
                "requested_samples": int(spatial_mixing_mc_samples),
                "successful_samples": 0,
                "failed_samples": 0,
                "warning": "Spatial-operator uncertainty is not added to a non-identifiable nominal physical result.",
            }
            return result

        rng = np.random.default_rng(int(spatial_mixing_mc_seed))
        parameter_draws = []
        failures = {"negative_or_rank_deficient_operator": 0, "fit_or_identifiability_failure": 0}
        for _ in range(int(spatial_mixing_mc_samples)):
            try:
                sampled_h = rng.multivariate_normal(
                    h.reshape(-1), mixing_cov, check_valid="raise"
                ).reshape(h.shape)
                if np.any(sampled_h < 0) or np.linalg.matrix_rank(sampled_h) < min(sampled_h.shape):
                    failures["negative_or_rank_deficient_operator"] += 1
                    continue
                sampled_result = reconstruct_joint_thomson_spatial_spectra(
                    args[0], args[1], args[2], sampled_h,
                    **{**fit_kwargs, "regularization_sensitivity_strengths": None},
                )
                if (not sampled_result.get("primary_outputs_reportable_as_identified")
                        or sampled_result.get("parameter_covariance") is None
                        or sampled_result.get("parameter_order") != result.get("parameter_order")):
                    failures["fit_or_identifiability_failure"] += 1
                    continue
                candidate = sampled_result["optimizer_candidate"]
                vector_parts = [
                    np.asarray(candidate["relative_spectral_amplitude_profile"], dtype=float),
                    np.asarray(candidate["electron_temperature_profile_eV"], dtype=float),
                    np.asarray(candidate["electron_velocity_projection_profile_m_s"], dtype=float),
                ]
                if "kappa_index" in sampled_result["parameter_order"]:
                    vector_parts.append(np.asarray([candidate["kappa_index"]], dtype=float))
                vector_parts.append(
                    np.asarray(sampled_result["detector_background_coefficients"], dtype=float).reshape(-1)
                )
                vector = np.concatenate(vector_parts)
                if vector.size != len(result["parameter_order"]) or not np.all(np.isfinite(vector)):
                    failures["fit_or_identifiability_failure"] += 1
                    continue
                parameter_draws.append(vector)
            except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                failures["fit_or_identifiability_failure"] += 1

        minimum_success = max(20, int(np.ceil(0.8 * int(spatial_mixing_mc_samples))))
        metadata = {
            "supplied": True,
            "propagated": False,
            "status": "insufficient_successful_refits",
            "covariance_shape": list(mixing_cov.shape),
            "covariance_parameter_order": [
                f"spatial_mixing_matrix_row_major_element_{i}" for i in range(h.size)
            ],
            "requested_samples": int(spatial_mixing_mc_samples),
            "successful_samples": len(parameter_draws),
            "failed_samples": int(spatial_mixing_mc_samples) - len(parameter_draws),
            "failure_reasons": failures,
            "minimum_successful_samples": minimum_success,
            "seed": int(spatial_mixing_mc_seed),
            "method": "nonlinear_forward_refit_monte_carlo",
            "covariance_semantics": "external_spatial_operator_parameter_covariance_added_to_data_only_GLS_covariance",
        }
        if len(parameter_draws) >= minimum_success:
            operator_contribution = np.cov(np.asarray(parameter_draws), rowvar=False, ddof=1)
            total_covariance = np.asarray(base_covariance, dtype=float) + operator_contribution
            total_covariance = 0.5 * (total_covariance + total_covariance.T)
            n_nodes = len(result["radius_m"])
            diagonal = np.maximum(np.diag(total_covariance), 0.0)
            result["parameter_covariance"] = total_covariance
            result["profile_parameter_covariance"] = total_covariance[:3*n_nodes, :3*n_nodes].copy()
            result["relative_spectral_amplitude_uncertainty"] = np.sqrt(diagonal[:n_nodes])
            result["electron_temperature_uncertainty_profile_eV"] = np.sqrt(diagonal[n_nodes:2*n_nodes])
            result["electron_temperature_uncertainty_eV"] = result["electron_temperature_uncertainty_profile_eV"]
            result["electron_velocity_projection_uncertainty_profile_m_s"] = np.sqrt(diagonal[2*n_nodes:3*n_nodes])
            result["electron_velocity_projection_uncertainty_m_s"] = result["electron_velocity_projection_uncertainty_profile_m_s"]
            metadata.update({
                "propagated": True,
                "status": "propagated_by_nonlinear_refits",
                "parameter_covariance_contribution": operator_contribution,
                "total_parameter_covariance": total_covariance,
            })
            result["spatial_mixing_operator"] = "researcher_supplied_nominal_forward_operator_with_external_covariance_propagated"
            result["covariance_decomposition"]["spatial_mixing_covariance"] = "nonlinear_refit_external_operator_contribution"
            result["parameter_identifiability"]["uncertainty_contributions"] = [
                "data_only_spectral_GLS",
                "independently_sampled_spatial_mixing_operator_refits",
            ]
            result["uncertainty_semantics"] += (
                " The reported covariance additionally includes the independently supplied "
                "spatial-mixing covariance propagated by nonlinear forward refits."
            )
        else:
            metadata["warning"] = (
                "Too few valid, data-identifiable spatial-operator refits; no spatial-mixing "
                "uncertainty contribution was added to the reported covariance."
            )
        result["spatial_mixing_uncertainty"] = metadata
        return result
    if selected_model not in {
        "non_collective_maxwellian",
        "non_collective_kappa",
        "non_collective_relativistic_maxwell_juttner",
    }:
        raise ValueError(
            "Thomson spatial profile reconstruction supports only the three registered "
            "non-collective optical Thomson models; microwave CTS has its own subanalysis."
        )
    result = analyze_thomson_spectra(*args, **kwargs)
    radii = np.asarray(result["radius_m"], dtype=float)
    n_channels = radii.size
    external_covariance = result.get("collective_external_parameter_covariance")
    external_order = result.get("collective_external_parameter_covariance_order") or []
    shared_response_covariance = result.get("shared_instrument_response_parameter_covariance")
    shared_response_order = result.get("shared_instrument_response_parameter_covariance_order") or []
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

    def shared_response_covariance_for(labels):
        if shared_response_covariance is None or not shared_response_order:
            return None
        indices = [
            shared_response_order.index(label) if label in shared_response_order else None
            for label in labels
        ]
        if not all(index is not None for index in indices):
            return None
        matrix = np.asarray(shared_response_covariance, dtype=float)
        return matrix[np.ix_(indices, indices)]

    def merge_shared_response_covariance(covariance, labels, marginal_uncertainty):
        block = shared_response_covariance_for(labels)
        if block is None:
            return covariance
        if covariance is None:
            sigma = np.asarray(marginal_uncertainty, dtype=float)
            if sigma.shape != (n_channels,) or not np.all(np.isfinite(sigma)):
                return None
            combined = np.diag(np.square(sigma))
            off_diagonal = block.copy()
            np.fill_diagonal(off_diagonal, 0.0)
            return combined + off_diagonal
        combined = np.asarray(covariance, dtype=float).copy()
        if combined.shape != (n_channels, n_channels):
            raise ValueError("Shared-response covariance does not match the number of Thomson channels.")
        return combined + block

    def add_collective_local_fit_covariance(covariance, parameter_index):
        """Combine external-input MC covariance with each channel's local fit covariance.

        The reported per-channel uncertainties already combine these independent
        contributions. Profile reconstruction must use the same total covariance,
        otherwise its covariance-diagonal consistency check fails and/or its bands
        omit the spectral-fit contribution.
        """
        if result.get("model") not in {"collective_maxwellian", "collective_kappa"}:
            return covariance
        local_variances = []
        for channel in result["channels"]:
            local = channel.get(
                "total_fit_covariance_including_discrepancy",
                channel.get("fit_covariance"),
            )
            if local is None:
                return covariance
            local = np.asarray(local, dtype=float)
            if (local.ndim != 2 or local.shape[0] != local.shape[1]
                    or local.shape[0] <= parameter_index
                    or not np.all(np.isfinite(local))):
                return covariance
            local_variances.append(float(local[parameter_index, parameter_index]))
        if len(local_variances) != n_channels or np.any(np.asarray(local_variances) < 0):
            return covariance
        combined = (
            np.zeros((n_channels, n_channels), dtype=float)
            if covariance is None
            else np.asarray(covariance, dtype=float).copy()
        )
        if combined.shape != (n_channels, n_channels):
            raise ValueError("Collective cross-channel covariance does not match the number of Thomson channels.")
        combined[np.diag_indices(n_channels)] += np.asarray(local_variances, dtype=float)
        return combined

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
    te_covariance = merge_shared_response_covariance(
        te_covariance, te_labels, result.get("electron_temperature_uncertainty_profile_eV")
    )
    flow_covariance = merge_shared_response_covariance(
        flow_covariance, flow_labels, result.get("electron_velocity_projection_uncertainty_profile_m_s")
    )
    if result.get("model") in {"collective_maxwellian", "collective_kappa"}:
        te_covariance = add_collective_local_fit_covariance(te_covariance, 1)
    if density_covariance is None:
        density_covariance = covariance_for(
            [f"channel_{i}_electron_density_m3" for i in range(n_channels)]
        )
    density_uncertainty_profile = result.get("electron_density_uncertainty_profile_m3")
    if density_uncertainty_profile is not None:
        if density_covariance is None:
            density_covariance = np.diag(np.square(np.asarray(density_uncertainty_profile, dtype=float)))
        else:
            density_covariance = np.asarray(density_covariance, dtype=float).copy()
        if result.get("electron_density_joint_covariance_m6") is None:
            for covariance_key in (
                "electron_density_angle_covariance_m6",
                "electron_density_shared_response_covariance_m6",
            ):
                additional_covariance = result.get(covariance_key)
                if additional_covariance is None:
                    continue
                additional_covariance = np.asarray(additional_covariance, dtype=float)
                if additional_covariance.shape == (n_channels, n_channels):
                    off_diagonal = additional_covariance.copy()
                    np.fill_diagonal(off_diagonal, 0.0)
                    density_covariance += np.nan_to_num(off_diagonal, nan=0.0)
    if result.get("model") in {"collective_maxwellian", "collective_kappa"}:
        density_covariance = add_collective_local_fit_covariance(density_covariance, 0)

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
            kappa_covariance = add_collective_local_fit_covariance(kappa_covariance, 2)
        elif joint_summary.get("total_kappa_parameter_covariance") is not None:
            kappa_covariance = np.asarray(
                joint_summary["total_kappa_parameter_covariance"], dtype=float
            )
        elif joint_summary.get("kappa_parameter_covariance") is not None:
            kappa_covariance = np.asarray(
                joint_summary["kappa_parameter_covariance"], dtype=float
            )
        candidates.append((
            "electron_kappa_index", None, None, kappa_covariance, False,
        ))
    if result.get("model") in {"collective_maxwellian", "collective_kappa"}:
        alpha_values = np.asarray([
            channel.get("scattering_parameter_alpha", np.nan) for channel in result["channels"]
        ], dtype=float)
        alpha_uncertainties = np.asarray([
            channel.get("scattering_parameter_alpha_uncertainty", np.nan)
            for channel in result["channels"]
        ], dtype=float)
        alpha_covariance = (
            np.diag(np.square(alpha_uncertainties))
            if np.all(np.isfinite(alpha_uncertainties)) else None
        )
        if alpha_covariance is not None:
            for contribution_key in (
                "scattering_parameter_alpha_laser_uncertainty_contribution",
                "scattering_parameter_alpha_angle_uncertainty_contribution",
            ):
                shared_contribution = np.asarray([
                    channel.get(contribution_key, np.nan) for channel in result["channels"]
                ], dtype=float)
                if np.all(np.isfinite(shared_contribution)):
                    shared_source_covariance = np.outer(
                        shared_contribution, shared_contribution
                    )
                    np.fill_diagonal(shared_source_covariance, 0.0)
                    alpha_covariance += shared_source_covariance
        candidates.append(("scattering_parameter_alpha", None, None, alpha_covariance, True))
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
        power_profile_covariance = (
            np.diag(power_sigma**2) if np.all(np.isfinite(power_sigma)) else None
        )
        if power_profile_covariance is not None:
            area_labels = [
                f"channel_{i}_integrated_scattered_power_W" for i in range(n_channels)
            ]
            shared_area_covariance = shared_response_covariance_for(area_labels)
            if shared_area_covariance is not None:
                off_diagonal = shared_area_covariance.copy()
                np.fill_diagonal(off_diagonal, 0.0)
                power_profile_covariance += off_diagonal
        candidates.append((
            power_profile_name, None, None, power_profile_covariance, True,
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
            uncertainty_values = (
                alpha_uncertainties if np.all(np.isfinite(alpha_uncertainties)) else None
            )
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
                coordinate_covariance=profile_coordinate_covariance_m2,
                coordinate_value_cross_covariance=profile_coordinate_value_cross_covariance,
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
                    coordinate_covariance=profile_coordinate_covariance_m2,
                    coordinate_value_cross_covariance=profile_coordinate_value_cross_covariance,
                )
            except ValueError as exc:
                profiles[name] = {"status": "not_reconstructed", "reason": str(exc)}
    result["spatial_profile_reconstruction"] = {
        "input_origin": "physical parameters inferred from measured calibrated Thomson spectra",
        "profiles": profiles,
        "channel_order": "supplied channel order; input coordinates must already be unique and strictly increasing; no sorting or reordering is performed",
        "methods_requested": profile_reconstruction_method,
        "no_extrapolation": True,
        "limitations": [
            "Reconstruction is a downstream spatial representation of parameters first inferred from the measured spectra.",
            "PCHIP and GCV smoothing splines are numerical profile representations, not additional plasma-physics inference models.",
            "Uncertainty bands are conditional on supplied/inferred channel covariance; when only pointwise uncertainties exist, channel independence is assumed.",
            "No equilibrium mapping, flux-surface mapping or tomographic inversion is inferred from radius alone.",
        ],
    }
    from .thomson_identifiability import spatial_profile_identifiability
    result["parameter_identifiability"] = spatial_profile_identifiability()
    result["registered_subanalysis"] = "Thomson spatial profile reconstruction from calibrated spectra"

    # Optional downstream PSF-aware profile inversion. This path is distinct
    # from the joint spatial-spectral fit above: it acts on physical parameters
    # already inferred per channel and never estimates/calibrates the mixing
    # operator. A profile grid is mandatory to prevent an implicit resolution
    # or interpolation choice.
    if spatial_mixing_matrix is not None:
        if profile_grid_radius_m is None:
            raise ValueError(
                "profile_grid_radius_m is required when spatial_mixing_matrix is supplied "
                "for downstream PSF-aware profile reconstruction."
            )
        from .thomson_spatial_mixing import reconstruct_profile_with_spatial_mixing
        profile_grid = np.asarray(profile_grid_radius_m, dtype=float)
        mixing = np.asarray(spatial_mixing_matrix, dtype=float)
        if (profile_grid.ndim != 1 or profile_grid.size < 2
                or not np.all(np.isfinite(profile_grid))
                or np.any(np.diff(profile_grid) <= 0)):
            raise ValueError("profile_grid_radius_m must be a finite, strictly increasing vector with at least two nodes.")
        if (mixing.ndim != 2 or mixing.shape != (n_channels, profile_grid.size)
                or not np.all(np.isfinite(mixing))):
            raise ValueError(
                "spatial_mixing_matrix must be finite with shape "
                "(n_measured_channels, n_profile_grid_nodes)."
            )
        if spatial_mixing_covariance is not None:
            covariance_shape = (mixing.size, mixing.size)
            raw_mixing_covariance = np.asarray(spatial_mixing_covariance)
            if (raw_mixing_covariance.dtype.kind not in "iuf"
                    or raw_mixing_covariance.shape != covariance_shape
                    or not np.all(np.isfinite(raw_mixing_covariance))
                    or not np.allclose(raw_mixing_covariance, raw_mixing_covariance.T,
                                        rtol=1e-10, atol=1e-14)):
                raise ValueError(
                    "spatial_mixing_covariance must be a finite symmetric matrix "
                    "with shape (spatial_mixing_matrix.size, spatial_mixing_matrix.size)"
                )
        elif spatial_mixing_mc_samples != 200:
            raise ValueError(
                "spatial_mixing_mc_samples is meaningful only when spatial_mixing_covariance is supplied"
            )
        psf_profiles = {}
        profile_inputs = (
            ("electron_temperature", result.get("electron_temperature_profile_eV"),
             result.get("electron_temperature_uncertainty_profile_eV"), te_covariance, True, "eV"),
            ("electron_density", result.get("electron_density_profile_m3"),
             result.get("electron_density_uncertainty_profile_m3"), density_covariance, True, "m^-3"),
            ("electron_velocity_projection", result.get("electron_velocity_projection_profile_m_s"),
             result.get("electron_velocity_projection_uncertainty_profile_m_s"), flow_covariance, False, "m s^-1"),
        )
        for name, values, sigma, parameter_covariance, positive_parameter, unit in profile_inputs:
            if values is None:
                psf_profiles[name] = {
                    "status": "not_available",
                    "reason": "The upstream spectral inference did not identify this parameter profile.",
                }
                continue
            values = np.asarray(values, dtype=float)
            if values.shape != (n_channels,) or not np.all(np.isfinite(values)):
                psf_profiles[name] = {
                    "status": "not_available",
                    "reason": "The upstream parameter profile is incomplete or non-finite.",
                }
                continue
            try:
                psf_profiles[name] = reconstruct_profile_with_spatial_mixing(
                    values, mixing, profile_grid, uncertainty=sigma,
                    covariance=parameter_covariance, positive=positive_parameter,
                    parameter_name=name, parameter_unit=unit,
                    regularization_strength=spatial_regularization_strength,
                    spatial_mixing_covariance=spatial_mixing_covariance,
                    mixing_mc_samples=spatial_mixing_mc_samples,
                    mixing_mc_seed=spatial_mixing_mc_seed,
                )
            except ValueError as exc:
                psf_profiles[name] = {"status": "not_reconstructed", "reason": str(exc)}
        result["psf_aware_profile_reconstruction"] = {
            "status": "completed",
            "profile_grid_radius_m": profile_grid,
            "spatial_mixing_operator": "researcher_supplied_fixed_forward_operator",
            "spatial_mixing_matrix": mixing,
            "spatial_mixing_covariance": (
                None if spatial_mixing_covariance is None
                else np.asarray(spatial_mixing_covariance, dtype=float).copy()
            ),
            "spatial_mixing_covariance_vectorization": "row-major C order over spatial_mixing_matrix entries",
            "profiles": psf_profiles,
            "interpretation": (
                "Downstream inversion of channel-inferred physical parameters through a fixed supplied "
                "spatial mixing/PSF operator. This is not a joint fit to raw spectra and does not estimate "
                "the PSF, instrument response, or unresolved spatial structure."
            ),
            "resolution_limit": (
                "Profile components in the null space or below the spatial resolution of the supplied "
                "mixing operator remain non-identifiable."
            ),
        }
    return result



def reconstruct_thomson_joint_spatial_spectra(*args, **kwargs):
    """Run joint wavelength-resolved spatial Thomson inversion with fixed mixing."""
    from .thomson_joint_spatial_spectral import reconstruct_joint_thomson_spatial_spectra
    return reconstruct_joint_thomson_spatial_spectra(*args, **kwargs)


# PSF-aware inverse profile reconstruction is distinct from PCHIP/smoothing interpolation.
from .thomson_gap_closure import reconstruct_psf_aware_thomson_profile



def fit_thomson_spectrum_with_structured_background(*args, **kwargs):
    """Jointly fit any supplied physical Thomson forward model and detector background.

    This lower-level public entry point is intended for relativistic MJ, Kappa,
    or custom kinetic forward models when their native fit API does not yet
    expose the structured background nuisance block.
    """
    from .thomson_inference_closure import fit_thomson_with_structured_background
    return fit_thomson_with_structured_background(*args, **kwargs)


def reconstruct_joint_ne_te_ti_profiles(*args, **kwargs):
    """Joint measured-spectrum inversion for ne(r), Te(r), and Ti(r).

    The caller must supply a physically appropriate channel-resolved forward
    model; ion temperature is not reported as identifiable unless that model
    contains an ion-sensitive spectral response.
    """
    from .thomson_inference_closure import reconstruct_joint_thomson_profiles
    return reconstruct_joint_thomson_profiles(*args, **kwargs)


def thomson_compton_recoil_contract(*args, **kwargs):
    """Return the explicit recoil/cross-section validity contract."""
    from .thomson_inference_closure import compton_recoil_contract
    return compton_recoil_contract(*args, **kwargs)


def propagate_full_em_cts_ray_geometry_uncertainty(*args, **kwargs):
    """Propagate equilibrium, ray, beam-covariance and scattering-volume uncertainty.

    Every Monte Carlo realization must be regenerated by the caller's geometry
    sampler and is passed through a complete Full-EM spectral refit. This is
    intentionally not approximated by adding a final-parameter covariance.
    """
    from .cts_geometry_uncertainty import propagate_full_em_cts_ray_geometry_uncertainty as _propagate
    return _propagate(*args, **kwargs)


def fit_noncollective_kinetic_thomson_with_background(*args, **kwargs):
    """Fit Kappa/MJ optical Thomson spectra with simultaneous structured background.

    The returned joint covariance retains physical, amplitude and background
    coefficient cross terms. This lower-level API does not infer absolute density.
    """
    from .thomson_kinetic_background import fit_noncollective_kinetic_thomson_with_background as _fit
    return _fit(*args, **kwargs)


def collective_tabulated_electron_thomson_spectrum(*args, **kwargs):
    """Evaluate electrostatic collective Thomson response for a supplied 1-D EVDF."""
    from .thomson_generalized_dispersion import collective_tabulated_electron_spectral_density
    return collective_tabulated_electron_spectral_density(*args, **kwargs)


def generalized_thomson_landau_integral(*args, **kwargs):
    """Evaluate the piecewise-linear first-order generalized Landau integral."""
    from .thomson_generalized_dispersion import generalized_landau_integral
    return generalized_landau_integral(*args, **kwargs)


def fit_n_population_relativistic_thomson_spectrum(*args, **kwargs):
    """Fit a finite N-population relativistic optical Thomson model to measured spectra."""
    from .thomson_n_population import (
        fit_n_population_relativistic_thomson_spectrum as _fit,
    )
    return _fit(*args, **kwargs)

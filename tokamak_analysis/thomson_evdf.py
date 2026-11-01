"""Non-parametric 1-D projected electron-velocity reconstruction from optical Thomson spectra.

This is an inverse problem on calibrated measured spectra, not a calibration or
measurement-correction routine. A single scattering geometry identifies only
the marginal along the scattering vector, never an unrestricted 3-D EVDF.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear, least_squares


_C = 299_792_458.0


def _finite_vector(values, name, minimum=9):
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size < minimum or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real one-dimensional array with at least {minimum} samples")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _response_matrix(values, n):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError("instrument_response_matrix must contain real numeric values")
    response = raw.astype(float, copy=False)
    if response.shape != (n, n) or not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("instrument_response_matrix must be finite, non-negative and square on the measured wavelength grid")
    sums = response.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError("every instrument-response column must have positive integrated response")
    return response / sums[None, :]


def reconstruct_thomson_evdf(
    wavelength_nm,
    signal,
    *,
    laser_wavelength_nm,
    scattering_angle_rad,
    instrument_response_matrix,
    background_signal,
    regularization_strength,
    velocity_grid_m_s=None,
    velocity_limit_m_s=8.0e6,
    velocity_points=241,
    signal_uncertainty=None,
    signal_covariance=None,
    max_iter=3000,
):
    """Reconstruct the non-negative 1-D electron-velocity marginal along q.

    The Doppler mapping uses the non-relativistic single-particle relation
    delta_omega = q dot v, with |q|/|k_i| = 2 sin(theta/2). The latent spectrum
    is convolved with the supplied measured response matrix before fitting.
    Background and regularization strength are explicit inputs. The result is
    shape-normalized; absolute density is not inferred by this inverse routine.
    """
    wavelength = _finite_vector(wavelength_nm, "wavelength_nm")
    measured = _finite_vector(signal, "signal")
    background = _finite_vector(background_signal, "background_signal")
    if measured.size != wavelength.size or background.size != wavelength.size:
        raise ValueError("wavelength_nm, signal and background_signal must have equal lengths")
    if np.any(np.diff(wavelength) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing; samples are never sorted or repaired")
    if isinstance(laser_wavelength_nm, (bool, np.bool_)) or not np.isfinite(laser_wavelength_nm) or laser_wavelength_nm <= 0:
        raise ValueError("laser_wavelength_nm must be finite and positive")
    if isinstance(scattering_angle_rad, (bool, np.bool_)) or not np.isfinite(scattering_angle_rad) or not 0 < scattering_angle_rad < np.pi:
        raise ValueError("scattering_angle_rad must lie strictly between 0 and pi")
    if isinstance(regularization_strength, (bool, np.bool_)) or not np.isfinite(regularization_strength) or regularization_strength <= 0:
        raise ValueError("regularization_strength must be finite and positive; choose and report it explicitly")
    if isinstance(velocity_points, (bool, np.bool_)) or not isinstance(velocity_points, (int, np.integer)) or velocity_points < 41 or velocity_points > 2001:
        raise ValueError("velocity_points must be an integer from 41 to 2001")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 100:
        raise ValueError("max_iter must be an integer >= 100")

    response = _response_matrix(instrument_response_matrix, wavelength.size)
    theta = float(scattering_angle_rad)
    q_over_ki = 2.0 * np.sin(theta / 2.0)
    if velocity_grid_m_s is None:
        if isinstance(velocity_limit_m_s, (bool, np.bool_)) or not np.isfinite(velocity_limit_m_s) or velocity_limit_m_s <= 0:
            raise ValueError("velocity_limit_m_s must be finite and positive")
        velocity = np.linspace(-float(velocity_limit_m_s), float(velocity_limit_m_s), int(velocity_points))
    else:
        velocity = _finite_vector(velocity_grid_m_s, "velocity_grid_m_s", minimum=41)
        if np.any(np.diff(velocity) <= 0):
            raise ValueError("velocity_grid_m_s must be strictly increasing")

    if np.max(np.abs(velocity)) >= 0.1 * _C:
        raise ValueError("The current Doppler inversion is non-relativistic and requires every represented |v_parallel| < 0.1 c; use a relativistic forward model otherwise")

    # omega_s = omega_i + q.v; first-order wavelength mapping.
    lambda_centres = float(laser_wavelength_nm) * (1.0 - q_over_ki * velocity / _C)
    if np.any(lambda_centres < wavelength[0]) or np.any(lambda_centres > wavelength[-1]):
        raise ValueError("velocity grid maps outside the measured wavelength interval; narrow the velocity grid or widen the measured spectral window")

    widths = np.empty(wavelength.size)
    widths[1:-1] = 0.5 * (wavelength[2:] - wavelength[:-2])
    widths[0] = wavelength[1] - wavelength[0]
    widths[-1] = wavelength[-1] - wavelength[-2]

    # Each velocity node contributes unit integrated spectral power. Linear
    # deposition avoids pretending that a sample is a finite-width line shape.
    latent = np.zeros((wavelength.size, velocity.size), dtype=float)
    for j, centre in enumerate(lambda_centres):
        right = int(np.searchsorted(wavelength, centre, side="right"))
        if right == 0:
            latent[0, j] = 1.0 / widths[0]
        elif right >= wavelength.size:
            latent[-1, j] = 1.0 / widths[-1]
        else:
            left = right - 1
            fraction = (centre - wavelength[left]) / (wavelength[right] - wavelength[left])
            latent[left, j] = (1.0 - fraction) / widths[left]
            latent[right, j] = fraction / widths[right]
    forward = response @ (latent * widths[:, None])
    forward = forward / widths[:, None]

    residual_data = measured - background
    if signal_uncertainty is not None and signal_covariance is not None:
        raise ValueError("Supply signal_uncertainty or signal_covariance, not both")
    if signal_uncertainty is not None:
        sigma = _finite_vector(signal_uncertainty, "signal_uncertainty")
        if sigma.size != measured.size or np.any(sigma <= 0):
            raise ValueError("signal_uncertainty must be strictly positive and match signal")
        design = forward / sigma[:, None]
        target = residual_data / sigma
        uncertainty_model = "independent_pointwise_measurement_uncertainty"
    elif signal_covariance is not None:
        covariance = np.asarray(signal_covariance, dtype=float)
        if covariance.shape != (measured.size, measured.size) or not np.all(np.isfinite(covariance)) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=0.0):
            raise ValueError("signal_covariance must be a finite symmetric square matrix matching signal")
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("signal_covariance must be positive definite") from exc
        design = np.linalg.solve(chol, forward)
        target = np.linalg.solve(chol, residual_data)
        uncertainty_model = "full_within_spectrum_covariance"
    else:
        scale = max(float(np.max(np.abs(residual_data))), np.finfo(float).tiny)
        design = forward / scale
        target = residual_data / scale
        uncertainty_model = "not_supplied_unweighted_fit"

    dv = np.diff(velocity)
    if not np.allclose(dv, dv[0], rtol=1e-6, atol=1e-12):
        raise ValueError("Non-uniform velocity_grid_m_s is not supported by the current regularizer")
    spacing = float(dv[0])
    # Second-difference Tikhonov penalty suppresses oscillatory, unresolved modes.
    smooth = np.diff(np.eye(velocity.size), n=2, axis=0) / (spacing * spacing)
    smooth_scale = np.linalg.norm(design, ord="fro") / max(np.linalg.norm(smooth, ord="fro"), np.finfo(float).tiny)
    augmented_design = np.vstack([design, np.sqrt(float(regularization_strength)) * smooth_scale * smooth])
    augmented_target = np.concatenate([target, np.zeros(smooth.shape[0])])
    fit = lsq_linear(augmented_design, augmented_target, bounds=(0.0, np.inf), method="trf", max_iter=int(max_iter), tol=1e-10)
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"EVDF non-negative regularized inversion failed: {fit.message}")
    coefficients = np.maximum(fit.x, 0.0)
    distribution = coefficients / spacing
    norm = float(np.trapz(distribution, velocity))
    if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
        raise ValueError("EVDF inversion returned no positive integrated distribution")
    distribution /= norm

    fitted = forward @ coefficients
    residual = residual_data - fitted
    integral = lambda values: float(np.trapz(values, velocity))
    mean_v = integral(velocity * distribution)
    centred = velocity - mean_v
    variance = integral(centred**2 * distribution)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Reconstructed EVDF has non-positive variance")
    sigma_v = float(np.sqrt(variance))
    skewness = integral(centred**3 * distribution) / sigma_v**3
    excess_kurtosis = integral(centred**4 * distribution) / variance**2 - 3.0
    electron_mass = 9.1093837139e-31
    boltzmann = 1.380649e-23
    directional_temperature_ev = electron_mass * variance / (boltzmann * 1.602176634e-19)

    singular_values = np.linalg.svd(design, compute_uv=False)
    threshold = np.finfo(float).eps * max(design.shape) * singular_values[0]
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else None

    # Compare three compact projected-marginal families against the same
    # measured-spectrum forward operator. These are candidate models, not
    # assumptions imposed on the non-parametric reconstruction.
    def projected_pdf(model_name, parameters):
        mean = float(parameters[0])
        scale = float(parameters[1])
        z = (velocity - mean) / scale
        if model_name == "maxwellian":
            raw_pdf = np.exp(-0.5 * z**2)
        elif model_name == "kappa_student_t":
            nu = float(parameters[2])
            raw_pdf = (1.0 + z**2 / nu) ** (-(nu + 1.0) / 2.0)
        else:
            exponent = float(parameters[2])
            raw_pdf = np.exp(-np.abs(z) ** exponent)
        area_pdf = float(np.trapz(raw_pdf, velocity))
        if not np.isfinite(area_pdf) or area_pdf <= 0:
            return np.zeros_like(velocity)
        return raw_pdf / area_pdf

    parametric_comparison = {}
    for model_name in ("maxwellian", "kappa_student_t", "super_gaussian"):
        if model_name == "maxwellian":
            initial = [float(np.max(np.abs(target))), mean_v, max(sigma_v, spacing)]
            lower_bounds = [0.0, float(velocity[0]), spacing]
            upper_bounds = [np.inf, float(velocity[-1]), float(velocity[-1]-velocity[0])]
        elif model_name == "kappa_student_t":
            initial = [float(np.max(np.abs(target))), mean_v, max(sigma_v, spacing), 5.0]
            lower_bounds = [0.0, float(velocity[0]), spacing, 2.05]
            upper_bounds = [np.inf, float(velocity[-1]), float(velocity[-1]-velocity[0]), 100.0]
        else:
            initial = [float(np.max(np.abs(target))), mean_v, max(sigma_v, spacing), 2.0]
            lower_bounds = [0.0, float(velocity[0]), spacing, 1.2]
            upper_bounds = [np.inf, float(velocity[-1]), float(velocity[-1]-velocity[0]), 8.0]

        def model_residual(parameters):
            pdf = projected_pdf(model_name, parameters[1:])
            probability = pdf * spacing
            probability /= max(float(np.sum(probability)), np.finfo(float).tiny)
            return design @ (probability * parameters[0]) - target

        try:
            model_fit = least_squares(
                model_residual, initial, bounds=(lower_bounds, upper_bounds),
                max_nfev=4000, xtol=1e-10, ftol=1e-10, gtol=1e-10,
            )
            model_resid = model_residual(model_fit.x)
            rss = float(model_resid @ model_resid)
            sample_count = int(target.size)
            parameter_count = int(model_fit.x.size)
            safe_rss = max(rss, np.finfo(float).tiny)
            parametric_comparison[model_name] = {
                "parameters": {
                    "amplitude_scaled": float(model_fit.x[0]),
                    "mean_velocity_parallel_m_s": float(model_fit.x[1]),
                    "scale_velocity_m_s": float(model_fit.x[2]),
                    **({"kappa_degrees_of_freedom": float(model_fit.x[3])} if model_name == "kappa_student_t" else {}),
                    **({"super_gaussian_exponent": float(model_fit.x[3])} if model_name == "super_gaussian" else {}),
                },
                "weighted_residual_sum_squares": rss,
                "aic": float(sample_count * np.log(safe_rss / sample_count) + 2 * parameter_count),
                "bic": float(sample_count * np.log(safe_rss / sample_count) + parameter_count * np.log(sample_count)),
                "success": bool(model_fit.success),
                "message": str(model_fit.message),
                "uncertainty_intervals": None,
            }
        except (ValueError, FloatingPointError) as exc:
            parametric_comparison[model_name] = {
                "success": False, "message": str(exc),
                "weighted_residual_sum_squares": None, "aic": None, "bic": None,
                "uncertainty_intervals": None,
            }

    return {
        "velocity_parallel_m_s": velocity,
        "evdf_parallel_s_per_m": distribution,
        "evdf_normalization": integral(distribution),
        "mean_velocity_parallel_m_s": mean_v,
        "velocity_parallel_standard_deviation_m_s": sigma_v,
        "directional_equivalent_temperature_eV": directional_temperature_ev,
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
        "fitted_spectrum": fitted,
        "residual": residual,
        "residual_sum_squares": float(np.sum(residual**2)),
        "regularization_strength": float(regularization_strength),
        "regularization": "non-negative second-difference Tikhonov",
        "measurement_uncertainty_model": uncertainty_model,
        "forward_model": "non-relativistic single-particle Doppler projection with supplied measured response matrix",
        "scattering_vector_projection_only": True,
        "absolute_density_identified": False,
        "numerical_design_rank": numerical_rank,
        "velocity_grid_points": int(velocity.size),
        "design_matrix_condition_number": condition,
        "parametric_model_comparison": parametric_comparison,
        "parametric_models": ["maxwellian", "kappa_student_t", "super_gaussian"],
        "fit_success": bool(fit.success),
        "fit_status": int(fit.status),
        "fit_message": str(fit.message),
        "identifiability": {
            "primary": ["normalized one-dimensional electron-velocity marginal along the scattering vector, conditional on the selected velocity grid, response matrix, background and regularization"],
            "conditional": ["uncertainty intervals require supplied measurement uncertainty/covariance and propagation through the constrained inverse problem", "absolute density requires independent calibrated Thomson power and complete optical collection geometry"],
            "non_identifiable": ["unrestricted three-dimensional EVDF from one scattering geometry", "perpendicular velocity components and anisotropy without independent viewing geometries or explicit physical constraints", "absolute electron density from this normalized inversion"],
        },
        "limitations": [
            "This is a regularized inverse estimate; velocity-space detail finer than the singular-value-supported resolution is not identifiable.",
            "The result depends on the declared velocity grid, regularization strength, supplied background and measured response matrix; sensitivity analysis over these choices is required for scientific interpretation.",
            "Only the one-dimensional marginal along the scattering vector is reconstructed. No isotropy assumption is silently imposed.",
            "The non-relativistic Doppler mapping is not suitable for relativistic electron populations.",
            "No posterior intervals are reported by this deterministic solver; provide a measurement covariance and use a future/bootstrap or Bayesian uncertainty workflow for calibrated intervals.",
            "The directional equivalent temperature is m_e Var(v_parallel)/k_B in eV; it is not a scalar electron temperature unless isotropy is independently justified.",
            "The routine does not infer absolute density or correct/calibrate measured spectra.",
        ],
    }

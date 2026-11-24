"""Poisson-likelihood inversion of a projected optical-Thomson EVDF.

The input is integer detector-bin counts, not calibrated spectral power. The
forward model integrates the Doppler kernel over finite wavelength bins,
applies the measured detector response, and fits a non-negative normalized
q-parallel velocity marginal with a separately fitted count-amplitude nuisance
parameter. A single view identifies only the q-parallel marginal.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import null_space
from scipy.special import softmax

from .thomson_evdf import (
    _finite_vector,
    _response_matrix,
    _projected_evdf_forward_matrix,
    _nonuniform_second_derivative_matrix,
    _velocity_quadrature_weights,
    _piecewise_linear_evdf_moments,
    _C,
)


def reconstruct_thomson_evdf_poisson(
    wavelength_nm,
    observed_counts,
    *,
    laser_wavelength_nm,
    scattering_angle_rad,
    instrument_response_matrix,
    background_expected_counts,
    velocity_grid_m_s,
    regularization_strength,
    initial_total_counts=None,
    max_iter=3000,
    electron_density_m3=None,
    electron_density_uncertainty_m3=None,
    electron_density_mu3_cross_covariance_s_minus3=None,
):
    """Infer normalized f_q(v) from independent Poisson detector-bin counts.

    Model:
      mu_i = A_tot * Delta_lambda_i * [R K f_q]_i + b_i
      L = sum_i (mu_i - y_i log(mu_i)) + lambda/2 ||D2 f_q||^2
      f_q >= 0, sum_j w_j f_q,j = 1, A_tot > 0.

    A_tot is the pre-window detected-count amplitude. Background counts are
    fixed explicit expected counts. The frequentist covariance is the
    fixed-active-set Poisson sandwich H^-1 F H^-1, projected onto the exact
    normalization tangent space; H includes Tikhonov curvature and F is the
    expected Poisson information. It is conditional on the selected
    regularization, velocity grid, response and fixed background.
    """
    wavelength = _finite_vector(wavelength_nm, "wavelength_nm")
    counts_raw = np.asarray(observed_counts)
    if counts_raw.ndim != 1 or counts_raw.size != wavelength.size or counts_raw.dtype.kind not in "iu":
        raise ValueError("observed_counts must be a one-dimensional integer-count array matching wavelength_nm")
    counts = counts_raw.astype(float)
    if np.any(counts < 0):
        raise ValueError("observed_counts must be non-negative")
    background = _finite_vector(background_expected_counts, "background_expected_counts")
    if background.shape != counts.shape or np.any(background < 0):
        raise ValueError("background_expected_counts must be finite, non-negative and match observed_counts")
    velocity = _finite_vector(velocity_grid_m_s, "velocity_grid_m_s", minimum=41)
    if np.any(np.diff(wavelength) <= 0) or np.any(np.diff(velocity) <= 0):
        raise ValueError("wavelength_nm and velocity_grid_m_s must be strictly increasing")
    if isinstance(laser_wavelength_nm, (bool, np.bool_)) or not np.isfinite(laser_wavelength_nm) or laser_wavelength_nm <= 0:
        raise ValueError("laser_wavelength_nm must be finite and positive")
    if isinstance(scattering_angle_rad, (bool, np.bool_)) or not np.isfinite(scattering_angle_rad) or not 0 < scattering_angle_rad < np.pi:
        raise ValueError("scattering_angle_rad must lie strictly between 0 and pi")
    if np.max(np.abs(velocity)) >= 0.1 * _C:
        raise ValueError("non-relativistic Poisson EVDF inversion requires |v| < 0.1 c")
    if isinstance(regularization_strength, (bool, np.bool_)) or not np.isscalar(regularization_strength) or not np.isfinite(regularization_strength) or regularization_strength <= 0:
        raise ValueError("regularization_strength must be a finite positive scalar; Poisson GCV is not silently approximated")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 100:
        raise ValueError("max_iter must be an integer >=100")

    response = _response_matrix(instrument_response_matrix, wavelength.size)
    forward_density, metadata = _projected_evdf_forward_matrix(
        wavelength, velocity, response, float(laser_wavelength_nm),
        2.0 * np.sin(float(scattering_angle_rad) / 2.0),
    )
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    bin_widths = np.diff(edges)
    # Convert spectral density to expected probability per finite detector bin.
    operator = bin_widths[:, None] * forward_density
    weights = _velocity_quadrature_weights(velocity)
    curvature = _nonuniform_second_derivative_matrix(velocity)
    curvature_scale = np.linalg.norm(operator, ord="fro") / max(
        np.linalg.norm(curvature, ord="fro"), np.finfo(float).tiny
    )
    penalty = curvature_scale * curvature
    lam = float(regularization_strength)

    if initial_total_counts is None:
        total_initial = max(float(np.sum(counts - background)), 1.0)
    else:
        if isinstance(initial_total_counts, (bool, np.bool_)) or not np.isscalar(initial_total_counts) or not np.isfinite(initial_total_counts) or initial_total_counts <= 0:
            raise ValueError("initial_total_counts must be finite and positive")
        total_initial = float(initial_total_counts)
    # Optimize dimensionless nodal probability masses p_j=w_j*f_j rather
    # than SI-scale densities f_j~1/(m/s). This keeps the simplex constraint
    # sum(p)=1 and the count-amplitude nuisance on numerically comparable
    # scales; the forward operator already contains the velocity weights.
    n = velocity.size
    tiny = np.finfo(float).tiny
    mass_operator = operator / weights[None, :]
    mass_initial = weights / float(np.sum(weights))
    # Softmax coordinates enforce the probability simplex exactly and avoid
    # SLSQP's fragile active-set line search on a high-dimensional equality-
    # constrained Poisson inverse. The final logit is fixed to zero to remove
    # the additive gauge freedom.
    logit_initial = np.log(mass_initial[:-1] / mass_initial[-1])
    log_amplitude_initial = float(np.log(total_initial))
    x0 = np.concatenate([logit_initial, [log_amplitude_initial]])
    log_amplitude_upper = float(np.log(max(total_initial, float(np.sum(counts)), 1.0) * 1.0e6))
    count_scale = max(float(np.sum(counts)), 1.0)

    def density_from_mass(mass):
        return mass / weights

    def mass_from_parameters(parameters):
        logits = np.concatenate([parameters[:n - 1], [0.0]])
        return softmax(logits)

    def expected(parameters):
        mass = mass_from_parameters(parameters)
        amplitude = float(np.exp(parameters[n - 1]))
        return amplitude * (mass_operator @ mass) + background

    def objective(parameters):
        mass = mass_from_parameters(parameters)
        density = density_from_mass(mass)
        mu = np.maximum(expected(parameters), tiny)
        poisson = np.sum(mu - counts * np.log(mu))
        smooth = 0.5 * lam * float(np.dot(penalty @ density, penalty @ density))
        return float((poisson + smooth) / count_scale)

    def gradient(parameters):
        mass = mass_from_parameters(parameters)
        density = density_from_mass(mass)
        amplitude = float(np.exp(parameters[n - 1]))
        mu = np.maximum(expected(parameters), tiny)
        score = 1.0 - counts / mu
        grad_mass = amplitude * (mass_operator.T @ score)
        grad_mass += lam * ((penalty.T @ (penalty @ density)) / weights)
        # Jacobian of softmax: dp/dz = diag(p)-p p^T.
        grad_logits = mass * (grad_mass - float(np.dot(mass, grad_mass)))
        grad_log_amplitude = amplitude * float(np.dot(mass_operator @ mass, score))
        return np.concatenate([
            grad_logits[:-1] / count_scale,
            [grad_log_amplitude / count_scale],
        ])

    fit = minimize(
        objective, x0, jac=gradient, method="L-BFGS-B",
        bounds=[(-35.0, 35.0)] * (n - 1) + [
            (float(np.log(tiny)), log_amplitude_upper)
        ],
        options={"maxiter": int(max_iter), "ftol": 1e-13, "gtol": 1e-8, "maxls": 50},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Poisson EVDF constrained optimizer failed: {fit.message}")
    mass = mass_from_parameters(fit.x)
    density = density_from_mass(mass)
    amplitude = float(np.exp(fit.x[n - 1]))
    normalization = float(weights @ density)
    if not np.isclose(normalization, 1.0, rtol=1e-7, atol=1e-9):
        raise ArithmeticError("Poisson EVDF optimizer violated the finite-volume normalization constraint")
    mu = expected(fit.x)
    if np.any(mu < 0) or np.any((mu == 0) & (counts > 0)) or not np.all(np.isfinite(mu)):
        raise ArithmeticError("Poisson EVDF produced invalid expected detector counts")

    # Signed square-root deviance is a goodness-of-fit diagnostic; the
    # optimized objective is the exact Poisson negative log likelihood.
    term = np.where(counts > 0, counts * np.log(np.maximum(counts, tiny) / np.maximum(mu, tiny)) - (counts - mu), mu)
    deviance = 2.0 * np.maximum(term, 0.0)
    deviance_residual = np.sign(counts - mu) * np.sqrt(deviance)
    active = np.ones(n, dtype=bool)
    free_indices = np.arange(n)
    parameter_indices = np.concatenate([free_indices, [n]])
    fisher_design = np.column_stack([
        amplitude * mass_operator[:, free_indices],
        amplitude * (mass_operator @ mass),
    ])
    data_fisher = fisher_design.T @ (fisher_design / np.maximum(mu, tiny)[:, None])
    fisher = data_fisher.copy()
    density_penalty_jacobian = penalty[:, free_indices] / weights[free_indices][None, :]
    fisher[:free_indices.size, :free_indices.size] += (
        lam * density_penalty_jacobian.T @ density_penalty_jacobian
    )
    # Project the exact simplex normalization constraint out of the active
    # parameter space. For the penalized Poisson estimator, H^-1 is not its
    # frequentist sampling covariance: use the sandwich H^-1 F H^-1, where
    # F is expected Poisson information and H=F+P includes Tikhonov curvature.
    equality = np.concatenate([np.ones(free_indices.size), [0.0]])[None, :]
    tangent = null_space(equality)
    reduced_hessian = tangent.T @ fisher @ tangent
    reduced_data_information = tangent.T @ data_fisher @ tangent
    covariance_full = None
    covariance_status = "unavailable_singular_active_set_fisher"
    try:
        if (reduced_hessian.size
                and np.linalg.matrix_rank(reduced_hessian) == reduced_hessian.shape[0]):
            inverse_hessian = np.linalg.inv(reduced_hessian)
            covariance_reduced = (
                inverse_hessian @ reduced_data_information @ inverse_hessian
            )
            covariance_reduced = 0.5 * (covariance_reduced + covariance_reduced.T)
            covariance_active_mass = tangent @ covariance_reduced @ tangent.T
            # Transform covariance from dimensionless masses p_j to density
            # coefficients f_j=p_j/w_j, retaining the amplitude cross terms.
            transform = np.diag(np.concatenate([1.0 / weights[free_indices], [amplitude]]))
            covariance_active = transform @ covariance_active_mass @ transform.T
            covariance_full = np.zeros((n + 1, n + 1))
            covariance_full[np.ix_(parameter_indices, parameter_indices)] = covariance_active
            covariance_status = "frequentist_poisson_sandwich_covariance_projected_to_simplex_tangent"
    except np.linalg.LinAlgError:
        covariance_full = None

    # Exact physical moments of the reconstructed piecewise-linear EVDF.
    # These are derived from f_q itself, not from a Gaussian spectral width.
    moment_fit = _piecewise_linear_evdf_moments(velocity, density)
    moment_order = (
        "mean_velocity_parallel_m_s",
        "velocity_parallel_variance_m2_s2",
        "velocity_parallel_standard_deviation_m_s",
        "directional_equivalent_temperature_eV",
        "central_third_velocity_moment_m3_s3",
        "central_fourth_velocity_moment_m4_s4",
        "skewness",
        "excess_kurtosis",
    )
    moment_values = np.asarray([
        moment_fit["mean"], moment_fit["variance"], moment_fit["sigma"],
        9.1093837139e-31 * moment_fit["variance"] / 1.602176634e-19,
        moment_fit["third"], moment_fit["fourth"],
        moment_fit["skewness"], moment_fit["excess_kurtosis"],
    ], dtype=float)
    moment_covariance = None
    moment_standard_uncertainty = None
    if covariance_full is not None:
        moment_covariance = (
            moment_fit["moment_jacobian"]
            @ covariance_full[:n, :n]
            @ moment_fit["moment_jacobian"].T
        )
        moment_covariance = 0.5 * (moment_covariance + moment_covariance.T)
        if np.all(np.isfinite(moment_covariance)):
            moment_standard_uncertainty = {
                name: float(np.sqrt(max(float(moment_covariance[i, i]), 0.0)))
                for i, name in enumerate(moment_order)
            }
        else:
            moment_covariance = None

    density_value = None
    density_uncertainty = 0.0
    projected_heat_flux = None
    projected_heat_flux_uncertainty = None
    if electron_density_uncertainty_m3 is not None and electron_density_m3 is None:
        raise ValueError("electron_density_uncertainty_m3 requires electron_density_m3")
    if electron_density_mu3_cross_covariance_s_minus3 is not None and (
        electron_density_m3 is None or electron_density_uncertainty_m3 is None
        or moment_covariance is None
    ):
        raise ValueError(
            "electron_density_mu3_cross_covariance_s_minus3 requires density, "
            "density uncertainty and an available EVDF moment covariance"
        )
    if electron_density_m3 is not None:
        if (isinstance(electron_density_m3, (bool, np.bool_))
                or not np.isscalar(electron_density_m3)
                or not np.isfinite(electron_density_m3)
                or float(electron_density_m3) <= 0.0):
            raise ValueError("electron_density_m3 must be a positive finite scalar")
        density_value = float(electron_density_m3)
        if electron_density_uncertainty_m3 is not None:
            if (isinstance(electron_density_uncertainty_m3, (bool, np.bool_))
                    or not np.isscalar(electron_density_uncertainty_m3)
                    or not np.isfinite(electron_density_uncertainty_m3)
                    or float(electron_density_uncertainty_m3) < 0.0):
                raise ValueError("electron_density_uncertainty_m3 must be finite and non-negative")
            density_uncertainty = float(electron_density_uncertainty_m3)
        mu3 = float(moment_fit["third"])
        mu3_variance = (
            float(moment_covariance[4, 4]) if moment_covariance is not None else None
        )
        cross = 0.0
        if electron_density_mu3_cross_covariance_s_minus3 is not None:
            cross = float(electron_density_mu3_cross_covariance_s_minus3)
            if not np.isfinite(cross):
                raise ValueError("electron_density_mu3_cross_covariance_s_minus3 must be finite")
            bound = np.sqrt(max(density_uncertainty**2 * mu3_variance, 0.0))
            if abs(cross) > bound + 1e-10 * max(bound, abs(cross), np.finfo(float).tiny):
                raise ValueError("density/third-moment cross-covariance violates the joint PSD bound")
        electron_mass = 9.1093837139e-31
        projected_heat_flux = 0.5 * electron_mass * density_value * mu3
        if mu3_variance is not None:
            flux_variance = (0.5 * electron_mass)**2 * (
                density_value**2 * mu3_variance
                + mu3**2 * density_uncertainty**2
                + 2.0 * density_value * mu3 * cross
            )
            flux_scale = (0.5 * electron_mass)**2 * max(
                density_value**2 * mu3_variance
                + mu3**2 * density_uncertainty**2
                + 2.0 * abs(density_value * mu3 * cross),
                np.finfo(float).tiny,
            )
            if flux_variance < -1e-10 * flux_scale:
                raise ValueError("joint density/EVDF covariance produced negative projected heat-flux variance")
            projected_heat_flux_uncertainty = float(np.sqrt(max(flux_variance, 0.0)))

    physical_moments = {
        "status": "conditional_poisson_evdf_moments",
        "primary_result_status": "conditional_on_poisson_inverse_regularization_and_velocity_support",
        "mean_velocity_parallel_m_s": float(moment_fit["mean"]),
        "velocity_parallel_variance_m2_s2": float(moment_fit["variance"]),
        "velocity_parallel_standard_deviation_m_s": float(moment_fit["sigma"]),
        "directional_equivalent_temperature_eV": float(moment_values[3]),
        "central_third_velocity_moment_m3_s3": float(moment_fit["third"]),
        "central_fourth_velocity_moment_m4_s4": float(moment_fit["fourth"]),
        "skewness": float(moment_fit["skewness"]),
        "excess_kurtosis": float(moment_fit["excess_kurtosis"]),
        "moment_covariance_order": moment_order,
        "moment_covariance": moment_covariance,
        "moment_standard_uncertainty": moment_standard_uncertainty,
        "parallel_projected_heat_flux_W_m2": projected_heat_flux,
        "parallel_projected_heat_flux_standard_uncertainty_W_m2": projected_heat_flux_uncertainty,
        "electron_density_input_m3": density_value,
        "heat_flux_equation": "q_parallel^(1D) = (m_e*n_e/2) * <(v_q-u_q)^3>_f",
        "directional_temperature_equation": "T_eq,q[eV] = (m_e/e) * Var_f(v_q)",
        "moment_quadrature": "exact_intervalwise_three_point_Gauss_Legendre_for_piecewise_linear_EVDF_through_fourth_order",
        "heat_flux_physical_scope": (
            "one-dimensional q-parallel kinetic-energy-flux contribution only; "
            "not the full 3-D electron heat flux, which also requires perpendicular-energy correlations"
        ),
        "full_three_dimensional_heat_flux_identified": False,
        "uncertainty_interpretation": (
            "Frequentist fixed-active-set Poisson sandwich covariance propagated "
            "through the exact normalized piecewise-linear EVDF moment Jacobian; "
            "conditional on the selected regularization, fixed background, "
            "velocity grid and measured response."
            if moment_covariance is not None else
            "Physical moments are reported, but their uncertainty is unavailable "
            "because the active-set Poisson covariance is singular."
        ),
    }

    from scipy.special import gammaln
    nll = float(np.sum(mu - counts * np.log(np.maximum(mu, tiny)) + gammaln(counts + 1.0)))
    return {
        "velocity_parallel_m_s": velocity.copy(),
        "evdf_parallel_s_per_m": density.copy(),
        "evdf_normalization": normalization,
        "fitted_expected_counts": mu,
        "fitted_scattered_expected_counts": mu - background,
        "background_expected_counts": background.copy(),
        "total_scattered_count_amplitude": amplitude,
        "poisson_negative_log_likelihood": nll,
        "poisson_deviance": float(np.sum(deviance)),
        "poisson_deviance_residual": deviance_residual,
        "regularization_strength": lam,
        "regularization_method": "explicit_positive_second_derivative_Tikhonov",
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "optimizer_iterations": int(fit.nit),
        "active_velocity_nodes": int(np.count_nonzero(active)),
        "evdf_covariance": None if covariance_full is None else covariance_full[:n, :n],
        "amplitude_variance": None if covariance_full is None else float(covariance_full[n, n]),
        "evdf_amplitude_covariance": None if covariance_full is None else covariance_full[:n, n],
        "projected_evdf_physical_moments": physical_moments,
        "covariance_status": covariance_status,
        "covariance_interpretation": "frequentist fixed-active-set Poisson sandwich covariance H^-1 F H^-1 projected onto the exact EVDF normalization simplex; conditional on explicit regularization, velocity grid, measured response and fixed background",
        "forward_operator": "finite_wavelength_bin_integrated_nonrelativistic_doppler_kernel_then_measured_detector_response",
        "forward_operator_metadata": metadata,
        "likelihood": "independent_poisson_counts",
        "physical_scope": "one_dimensional_q_parallel_velocity_marginal_only",
        "normalization_constraint": "sum_j w_j f_q(v_j)=1",
        "objective_equation": "sum_i[mu_i-y_i*log(mu_i)] + lambda/2*||D2 f_q||_2^2",
        "expected_count_equation": "mu_i=A_tot*Delta_lambda_i*[R K f_q]_i+b_i",
        "count_data_contract": "non-negative integer detector-bin counts; not calibrated spectral power",
        "regularization_not_data": True,
        "absolute_density_inferred": False,
        "limitations": [
            "One scattering view identifies only the q-parallel marginal, not a general 3-D EVDF.",
            "Background expected counts are fixed supplied inputs; their uncertainty is not included in this covariance.",
            "Regularization strength is explicit and its choice is not included in the Fisher covariance.",
            "The non-relativistic Doppler kernel requires |v|<0.1c.",
            "No empirical validation or detector calibration is performed.",
        ],
    }

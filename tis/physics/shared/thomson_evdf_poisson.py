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

from .thomson_evdf import (
    _finite_vector,
    _response_matrix,
    _projected_evdf_forward_matrix,
    _nonuniform_second_derivative_matrix,
    _velocity_quadrature_weights,
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
):
    """Infer normalized f_q(v) from independent Poisson detector-bin counts.

    Model:
      mu_i = A_tot * Delta_lambda_i * [R K f_q]_i + b_i
      L = sum_i (mu_i - y_i log(mu_i)) + lambda/2 ||D2 f_q||^2
      f_q >= 0, sum_j w_j f_q,j = 1, A_tot > 0.

    A_tot is the pre-window detected-count amplitude. Background counts are
    fixed explicit expected counts. The covariance is the inverse expected
    Poisson Fisher information on the active non-negative set, projected onto
    the normalization tangent space. It is conditional on the selected
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
    x0 = np.concatenate([mass_initial, [total_initial]])

    def density_from_mass(mass):
        return mass / weights

    def expected(parameters):
        mass = parameters[:n]
        amplitude = parameters[n]
        return amplitude * (mass_operator @ mass) + background

    def objective(parameters):
        mass = parameters[:n]
        density = density_from_mass(mass)
        mu = np.maximum(expected(parameters), tiny)
        poisson = np.sum(mu - counts * np.log(mu))
        smooth = 0.5 * lam * float(np.dot(penalty @ density, penalty @ density))
        return float(poisson + smooth)

    def gradient(parameters):
        mass = parameters[:n]
        density = density_from_mass(mass)
        amplitude = parameters[n]
        mu = np.maximum(expected(parameters), tiny)
        score = 1.0 - counts / mu
        grad_mass = amplitude * (mass_operator.T @ score)
        grad_mass += lam * ((penalty.T @ (penalty @ density)) / weights)
        grad_amplitude = float(np.dot(mass_operator @ mass, score))
        return np.concatenate([grad_mass, [grad_amplitude]])

    constraint = {
        "type": "eq",
        "fun": lambda p: float(np.sum(p[:n]) - 1.0),
        "jac": lambda p: np.concatenate([np.ones(n), [0.0]]),
    }
    fit = minimize(
        objective, x0, jac=gradient, method="SLSQP",
        bounds=[(0.0, None)] * n + [(tiny, None)],
        constraints=[constraint],
        options={"maxiter": int(max_iter), "ftol": 1e-10, "disp": False},
    )
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"Poisson EVDF constrained optimizer failed: {fit.message}")
    mass = fit.x[:n]
    density = density_from_mass(mass)
    amplitude = float(fit.x[n])
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
    active = mass > max(float(np.max(mass)) * 1e-8, tiny)
    free_indices = np.flatnonzero(active)
    parameter_indices = np.concatenate([free_indices, [n]])
    fisher_design = np.column_stack([amplitude * mass_operator[:, free_indices], mass_operator @ mass])
    fisher = fisher_design.T @ (fisher_design / np.maximum(mu, tiny)[:, None])
    density_penalty_jacobian = penalty[:, free_indices] / weights[free_indices][None, :]
    fisher[:free_indices.size, :free_indices.size] += (
        lam * density_penalty_jacobian.T @ density_penalty_jacobian
    )
    # Project out the exact simplex-normalization direction before inversion.
    equality = np.concatenate([np.ones(free_indices.size), [0.0]])[None, :]
    tangent = null_space(equality)
    reduced = tangent.T @ fisher @ tangent
    covariance_full = None
    covariance_status = "unavailable_singular_active_set_fisher"
    try:
        if reduced.size and np.linalg.matrix_rank(reduced) == reduced.shape[0]:
            covariance_reduced = np.linalg.inv(reduced)
            covariance_active_mass = tangent @ covariance_reduced @ tangent.T
            # Transform covariance from dimensionless masses p_j to density
            # coefficients f_j=p_j/w_j, retaining the amplitude cross terms.
            transform = np.diag(np.concatenate([1.0 / weights[free_indices], [1.0]]))
            covariance_active = transform @ covariance_active_mass @ transform.T
            covariance_full = np.zeros((n + 1, n + 1))
            covariance_full[np.ix_(parameter_indices, parameter_indices)] = covariance_active
            covariance_status = "expected_poisson_fisher_inverse_projected_to_normalization_tangent"
    except np.linalg.LinAlgError:
        covariance_full = None

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
        "covariance_status": covariance_status,
        "covariance_interpretation": "local expected-Poisson Fisher covariance conditional on active set, explicit regularization, velocity grid, measured response and fixed background",
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

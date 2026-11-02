"""Shared numerical primitives for uncertainty-aware Thomson/CTS inference.

These routines are deliberately model-agnostic. They do not calibrate or repair
measurements, invent missing diagnostics, or turn regularization into data.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import minimize_scalar
from scipy.special import roots_legendre


def validate_covariance(covariance, size: int, *, name: str = "covariance",
                        positive_definite: bool = False, rtol: float = 1e-10):
    """Validate a real symmetric covariance and return its symmetrized copy."""
    raw = np.asarray(covariance)
    if raw.dtype.kind not in "iuf" or raw.shape != (size, size):
        raise ValueError(f"{name} must be a real {size}x{size} matrix")
    cov = raw.astype(float, copy=False)
    if not np.all(np.isfinite(cov)):
        raise ValueError(f"{name} must be finite")
    scale = max(float(np.max(np.abs(cov))), 1.0)
    if not np.allclose(cov, cov.T, rtol=rtol, atol=rtol * scale):
        raise ValueError(f"{name} must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    tolerance = rtol * max(float(np.max(np.abs(eigenvalues))), 1.0)
    if eigenvalues[0] < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    if positive_definite and eigenvalues[0] <= tolerance:
        raise ValueError(f"{name} must be positive definite")
    return cov


def covariance_whitener(covariance):
    """Return W such that ||W r||² = r.T C^-1 r for positive-definite C."""
    cov = validate_covariance(covariance, np.asarray(covariance).shape[0],
                              positive_definite=True)
    chol = np.linalg.cholesky(cov)
    return solve_triangular(chol, np.eye(chol.shape[0]), lower=True)


def structured_background_design(axis, *, kind="polynomial", degree=1,
                                 knots=None, reference_spectra=None,
                                 components=None):
    """Build a detector-space nuisance basis without subtracting measured data.

    Supported bases: polynomial, cubic B-spline, or a fixed PCA basis from
    calibrated reference spectra. Basis coefficients are profiled jointly with
    the physical forward model and must not be interpreted as plasma parameters.
    """
    x = np.asarray(axis, dtype=float)
    if x.ndim != 1 or x.size < 5 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("axis must be a finite, strictly increasing vector with >=5 samples")
    xc = (x - np.mean(x)) / max(float(np.ptp(x)), np.finfo(float).tiny)
    if kind == "polynomial":
        if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or not 0 <= degree <= 6:
            raise ValueError("polynomial degree must be an integer in [0, 6]")
        return np.vander(xc, N=int(degree) + 1, increasing=True)
    if kind == "spline":
        from scipy.interpolate import BSpline
        if knots is None:
            raise ValueError("spline background requires explicit internal knots")
        internal = np.asarray(knots, dtype=float)
        if internal.ndim != 1 or not np.all(np.isfinite(internal)) or np.any(np.diff(internal) <= 0):
            raise ValueError("spline knots must be finite and strictly increasing")
        if np.any(internal <= x[0]) or np.any(internal >= x[-1]):
            raise ValueError("spline internal knots must lie strictly inside the measured axis")
        degree = 3
        t = np.r_[np.repeat(x[0], degree + 1), internal, np.repeat(x[-1], degree + 1)]
        n_basis = len(t) - degree - 1
        design = np.empty((x.size, n_basis), dtype=float)
        for j in range(n_basis):
            coeff = np.zeros(n_basis)
            coeff[j] = 1.0
            design[:, j] = BSpline(t, coeff, degree, extrapolate=False)(x)
        return design
    if kind == "pca":
        if reference_spectra is None:
            raise ValueError("PCA background requires calibrated reference_spectra")
        refs = np.asarray(reference_spectra, dtype=float)
        if refs.ndim != 2 or refs.shape[1] != x.size or refs.shape[0] < 2 or not np.all(np.isfinite(refs)) or np.any(refs < 0):
            raise ValueError("reference_spectra must be >=2 finite non-negative spectra on the measured axis")
        if isinstance(components, (bool, np.bool_)) or not isinstance(components, (int, np.integer)):
            raise ValueError("PCA components must be an explicit integer")
        if not 1 <= int(components) <= min(refs.shape[0] - 1, x.size):
            raise ValueError("PCA components exceed the available reference rank")
        mean = np.mean(refs, axis=0)
        _, singular, vt = np.linalg.svd(refs - mean, full_matrices=False)
        rank = int(np.count_nonzero(singular > np.finfo(float).eps * max(refs.shape) * singular[0]))
        if int(components) > rank:
            raise ValueError(f"requested PCA components exceed reference numerical rank {rank}")
        return np.column_stack((mean, vt[:int(components)].T))
    raise ValueError("kind must be 'polynomial', 'spline', or 'pca'")


def profile_structured_background(measured, physical_spectrum, design, *,
                                  covariance=None, uncertainty=None,
                                  nonnegative_physical_amplitude=True):
    """GLS-profile linear amplitude and structured background coefficients."""
    y = np.asarray(measured, dtype=float)
    s = np.asarray(physical_spectrum, dtype=float)
    b = np.asarray(design, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or b.ndim != 2 or b.shape[0] != y.size:
        raise ValueError("measured, physical_spectrum and background design shapes do not match")
    if not all(np.all(np.isfinite(v)) for v in (y, s, b)):
        raise ValueError("profile inputs must be finite")
    if (covariance is None) == (uncertainty is None):
        raise ValueError("supply exactly one of covariance or pointwise uncertainty")
    if covariance is not None:
        w = covariance_whitener(covariance)
        a_w, y_w = w @ np.column_stack((s, b)), w @ y
        uncertainty_semantics = "supplied_full_measurement_covariance"
    else:
        sigma = np.asarray(uncertainty, dtype=float)
        if sigma.shape != y.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("uncertainty must be finite, positive and match measured")
        a_w, y_w = np.column_stack((s, b)) / sigma[:, None], y / sigma
        uncertainty_semantics = "supplied_pointwise_measurement_uncertainty"
    normal = a_w.T @ a_w
    rhs = a_w.T @ y_w
    if nonnegative_physical_amplitude:
        result = minimize_scalar(
            lambda amp: float(np.linalg.norm(y_w - a_w[:, 0] * amp -
                a_w[:, 1:] @ np.linalg.lstsq(a_w[:, 1:], y_w - a_w[:, 0] * amp, rcond=None)[0]) ** 2),
            bounds=(0.0, max(1.0, 10.0 * np.linalg.norm(y_w) /
                              max(np.linalg.norm(a_w[:, 0]), np.finfo(float).tiny))),
            method="bounded",
        )
        amp = float(result.x)
        bg = np.linalg.lstsq(a_w[:, 1:], y_w - a_w[:, 0] * amp, rcond=None)[0]
        coefficients = np.r_[amp, bg]
    else:
        coefficients = np.linalg.lstsq(a_w, y_w, rcond=None)[0]
    fitted = np.column_stack((s, b)) @ coefficients
    residual = y - fitted
    rank = int(np.linalg.matrix_rank(a_w))
    parameter_count = int(a_w.shape[1])
    identifiable = rank == parameter_count
    cov = np.linalg.pinv(normal) if identifiable else None
    return {
        "fitted_spectrum": fitted,
        "residual": residual,
        "physical_amplitude": float(coefficients[0]),
        "background_coefficients": coefficients[1:],
        "coefficient_covariance": cov,
        "design_rank": rank,
        "design_parameter_count": parameter_count,
        "identifiable": identifiable,
        "uncertainty_semantics": uncertainty_semantics,
        "background_is_detector_space_nuisance": True,
        "background_basis_does_not_establish_physical_contamination_source": True,
    }


def propagate_external_covariance(mean, covariance, evaluator: Callable,
                                  *, samples=500, seed=1729,
                                  minimum_success_fraction=0.8):
    """Monte-Carlo propagate correlated external inputs through a forward/inverse evaluator.

    evaluator(draw) must return a mapping of scalar numeric output names to values.
    This produces conditional numerical sensitivity intervals, not a posterior.
    """
    mu = np.asarray(mean, dtype=float)
    if mu.ndim != 1 or mu.size == 0 or not np.all(np.isfinite(mu)):
        raise ValueError("external-input mean must be a finite non-empty vector")
    cov = validate_covariance(covariance, mu.size, name="external_input_covariance")
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 50:
        raise ValueError("samples must be an integer >=50")
    if not 0 < minimum_success_fraction <= 1:
        raise ValueError("minimum_success_fraction must lie in (0, 1]")
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mu, cov, size=int(samples), check_valid="raise")
    records, failures = [], 0
    keys = None
    for draw in draws:
        try:
            result = evaluator(draw)
            if not isinstance(result, Mapping):
                raise ValueError("evaluator must return a mapping")
            current = tuple(sorted(result))
            if keys is None:
                keys = current
            if current != keys:
                raise ValueError("evaluator returned inconsistent output keys")
            row = np.asarray([result[k] for k in keys], dtype=float)
            if not np.all(np.isfinite(row)):
                raise ValueError("evaluator returned non-finite outputs")
            records.append(row)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError, RuntimeError):
            failures += 1
    minimum = int(np.ceil(float(minimum_success_fraction) * int(samples)))
    if keys is None or len(records) < minimum:
        return {
            "status": "insufficient_valid_external_draws",
            "successful_samples": len(records),
            "failed_samples": failures,
            "requested_samples": int(samples),
            "minimum_successful_samples": minimum,
            "uncertainty_semantics": "not_available",
        }
    values = np.asarray(records)
    return {
        "status": "completed",
        "output_names": list(keys),
        "mean": dict(zip(keys, np.mean(values, axis=0).tolist())),
        "standard_uncertainty": dict(zip(keys, np.std(values, axis=0, ddof=1).tolist())),
        "lower_95_percentile": dict(zip(keys, np.percentile(values, 2.5, axis=0).tolist())),
        "upper_95_percentile": dict(zip(keys, np.percentile(values, 97.5, axis=0).tolist())),
        "output_covariance": np.cov(values, rowvar=False).tolist(),
        "successful_samples": len(records),
        "failed_samples": failures,
        "requested_samples": int(samples),
        "uncertainty_semantics": "Monte-Carlo propagation of supplied external-input covariance conditional on the selected model and evaluator",
        "not_a_posterior": True,
    }


def _second_difference(n):
    if n < 3:
        return np.zeros((0, n))
    d = np.zeros((n - 2, n))
    for i in range(n - 2):
        d[i, i:i + 3] = (1.0, -2.0, 1.0)
    return d


def reconstruct_profile_with_psf(measured_profile, mixing_matrix, *,
                                  measurement_covariance=None,
                                  measurement_uncertainty=None,
                                  regularization=None, candidate_regularizations=None):
    """Recover a latent profile through a supplied measured spatial mixing/PSF matrix.

    M maps latent grid values to measured channels: y = M x + noise.
    No PSF is guessed. GCV chooses only among the supplied candidate strengths.
    """
    y = np.asarray(measured_profile, dtype=float)
    m = np.asarray(mixing_matrix, dtype=float)
    if y.ndim != 1 or m.ndim != 2 or m.shape[0] != y.size or m.shape[1] < 3:
        raise ValueError("mixing_matrix must have shape (n_measured_channels, n_latent_points>=3)")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(m)) or np.any(m < 0):
        raise ValueError("profile and mixing matrix must be finite; mixing weights must be non-negative")
    if not np.allclose(m.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise ValueError("each measured-channel PSF row must sum to one; no implicit renormalization is performed")
    if (measurement_covariance is None) == (measurement_uncertainty is None):
        raise ValueError("supply exactly one of measurement_covariance or measurement_uncertainty")
    if measurement_covariance is not None:
        w = covariance_whitener(measurement_covariance)
        cov_semantics = "full_measured_channel_covariance"
    else:
        sigma = np.asarray(measurement_uncertainty, dtype=float)
        if sigma.shape != y.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be finite, positive and match measured_profile")
        w = np.diag(1.0 / sigma)
        cov_semantics = "independent_pointwise_measurement_uncertainty"
    a = w @ m
    b = w @ y
    d = _second_difference(m.shape[1])
    if regularization is None:
        if candidate_regularizations is None:
            raise ValueError("supply regularization or an explicit candidate_regularizations grid for GCV")
        candidates = np.asarray(candidate_regularizations, dtype=float)
        if candidates.ndim != 1 or candidates.size < 3 or np.any(~np.isfinite(candidates)) or np.any(candidates <= 0) or np.any(np.diff(candidates) <= 0):
            raise ValueError("candidate_regularizations must be >=3 finite, positive, strictly increasing values")
        def score(lam):
            h = a.T @ a + lam * (d.T @ d)
            inv = np.linalg.pinv(h)
            xhat = inv @ a.T @ b
            residual = b - a @ xhat
            hat_trace = float(np.trace(a @ inv @ a.T))
            denom = max((b.size - hat_trace) ** 2, np.finfo(float).tiny)
            return float((residual @ residual) / denom)
        scores = np.asarray([score(v) for v in candidates])
        selected = int(np.argmin(scores))
        lam = float(candidates[selected])
        selection = "GCV_over_researcher_supplied_candidate_grid"
        gcv_scores = scores
    else:
        if isinstance(regularization, (bool, np.bool_)) or not np.isscalar(regularization) or not np.isfinite(regularization) or regularization < 0:
            raise ValueError("regularization must be a finite non-negative scalar")
        lam = float(regularization)
        selection = "researcher_supplied"
        gcv_scores = None
    h = a.T @ a + lam * (d.T @ d)
    rank = int(np.linalg.matrix_rank(a))
    xhat = np.linalg.pinv(h) @ a.T @ b
    residual = y - m @ xhat
    # Local linear covariance conditional on the selected lambda and fixed PSF.
    operator = np.linalg.pinv(h) @ a.T @ w
    if measurement_covariance is not None:
        cov_y = validate_covariance(measurement_covariance, y.size)
    else:
        cov_y = np.diag(sigma ** 2)
    cov_x = operator @ cov_y @ operator.T
    singular = np.linalg.svd(a, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    identifiable = rank == min(a.shape) and np.isfinite(condition) and condition < 1e12
    return {
        "latent_profile": xhat,
        "latent_profile_standard_uncertainty": np.sqrt(np.maximum(np.diag(cov_x), 0.0)),
        "latent_profile_covariance": cov_x,
        "fitted_measured_profile": m @ xhat,
        "residual": residual,
        "mixing_matrix": m,
        "regularization": lam,
        "regularization_selection": selection,
        "gcv_scores": gcv_scores,
        "measurement_covariance_model": cov_semantics,
        "mixing_matrix_is_external_fixed_input": True,
        "mixing_matrix_row_sum_convention": "each measured channel is a convex PSF-weighted average of latent spatial values",
        "data_jacobian_rank": rank,
        "data_jacobian_shape": list(a.shape),
        "data_jacobian_condition_number": condition,
        "data_identifiability_status": "full_column_rank" if rank == m.shape[1] else "rank_deficient_or_underdetermined",
        "regularization_does_not_create_unmeasured_spatial_information": True,
    }


def relativistic_mj_projected_beta_pdf(beta_parallel, temperature_eV, *,
                                       quadrature_order=256):
    """Exact 1-D velocity-component marginal of an isotropic Maxwell-Juttner gas.

    The transverse momentum is integrated out analytically. This is a relativistic
    velocity marginal, not a non-relativistic Doppler-width approximation and not
    an arbitrary anisotropic EVDF reconstruction.
    """
    beta = np.asarray(beta_parallel, dtype=float)
    if beta.ndim != 1 or beta.size < 3 or not np.all(np.isfinite(beta)) or np.any(np.abs(beta) >= 1):
        raise ValueError("beta_parallel must be a finite vector strictly inside (-1, 1)")
    if isinstance(temperature_eV, (bool, np.bool_)) or not np.isscalar(temperature_eV) or not np.isfinite(temperature_eV) or temperature_eV <= 0:
        raise ValueError("temperature_eV must be positive and finite")
    if isinstance(quadrature_order, (bool, np.bool_)) or not isinstance(quadrature_order, (int, np.integer)) or quadrature_order < 32:
        raise ValueError("quadrature_order must be an integer >=32")
    theta = float(temperature_eV) / 510998.95069
    gamma = 1.0 / np.sqrt(1.0 - beta ** 2)
    a = gamma / theta
    # Integral_1^infinity u^2 exp(-a*u) du, including the exact
    # dp_parallel / d beta Jacobian and the cylindrical transverse measure.
    unnormalized = np.exp(-a) * (1.0 / a + 2.0 / a**2 + 2.0 / a**3) * gamma**3
    nodes, weights = roots_legendre(int(quadrature_order))
    lo, hi = -1.0 + 1e-12, 1.0 - 1e-12
    grid = 0.5 * (hi - lo) * nodes + 0.5 * (hi + lo)
    gg = 1.0 / np.sqrt(1.0 - grid**2)
    aa = gg / theta
    ug = np.exp(-aa) * (1.0 / aa + 2.0 / aa**2 + 2.0 / aa**3) * gg**3
    norm = 0.5 * (hi - lo) * float(np.dot(weights, ug))
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Maxwell-Juttner projected marginal normalization failed")
    return unnormalized / norm

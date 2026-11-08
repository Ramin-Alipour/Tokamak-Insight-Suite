"""Joint covariance, regularization and reference-basis operators for Analysis 1.

These are inference utilities, not experimental validation.  Forward-model
callbacks must return detector-space predictions on the supplied measured bins.
All covariance matrices use channel-major flattening unless stated otherwise.
"""
from __future__ import annotations

from collections.abc import Callable
import numpy as np
from scipy.optimize import least_squares


def _real(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    out = raw.astype(float, copy=False)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite and non-empty")
    return out


def validate_covariance(covariance, size, *, name="covariance", positive_definite=True):
    """Validate and symmetrize a covariance without silently repairing it."""
    cov = _real(covariance, name, 2)
    if cov.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size})")
    scale = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12 * scale):
        raise ValueError(f"{name} must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    tol = 1e-11 * max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if positive_definite and eig[0] <= tol:
        raise ValueError(f"{name} must be positive definite")
    if not positive_definite and eig[0] < -tol:
        raise ValueError(f"{name} must be positive semidefinite")
    return cov


def _whitener(covariance):
    chol = np.linalg.cholesky(covariance)
    return lambda residual: np.linalg.solve(chol, residual)


def fit_multichannel_joint_gls(
    measured_spectra, *, forward_model, physical_initial,
    measurement_covariance, physical_bounds=(-np.inf, np.inf),
    background_design=None, background_bounds=(-np.inf, np.inf),
    max_nfev=8000,
):
    """Fit all channels jointly with full cross-channel spectral covariance.

    forward_model(parameters, channel_index) returns one channel's detector-
    space physical spectrum.  The optional background_design is either
    (n_channel,n_bin,n_basis) or (n_bin,n_basis), shared by channels.
    Physical parameters are shared/global; each channel has independent
    linear background coefficients.  The full covariance is channel-major.
    """
    y = _real(measured_spectra, "measured_spectra", 2)
    nc, nb = y.shape
    p0 = _real(physical_initial, "physical_initial", 1)
    cov = validate_covariance(measurement_covariance, nc * nb,
                              name="measurement_covariance")
    whiten = _whitener(cov)
    if not callable(forward_model):
        raise ValueError("forward_model must be callable")
    if background_design is None:
        basis = np.zeros((nc, nb, 0))
    else:
        raw = _real(background_design, "background_design")
        if raw.ndim == 2 and raw.shape[0] == nb:
            basis = np.broadcast_to(raw, (nc,) + raw.shape).copy()
        elif raw.ndim == 3 and raw.shape[:2] == (nc, nb):
            basis = raw.copy()
        else:
            raise ValueError("background_design must be (n_bin,n_basis) or (n_channel,n_bin,n_basis)")
        if basis.shape[2] and any(np.linalg.matrix_rank(basis[i]) < basis.shape[2] for i in range(nc)):
            raise ValueError("each channel background design must have full column rank")
    nk = basis.shape[2]
    plo, phi = (np.broadcast_to(np.asarray(v, float), p0.shape).copy()
                for v in physical_bounds)
    blo, bhi = (np.broadcast_to(np.asarray(v, float), (nc, nk)).copy()
                for v in background_bounds)
    if np.any(plo >= phi) or np.any(p0 < plo) or np.any(p0 > phi):
        raise ValueError("physical bounds must be ordered and contain initial values")
    if nk and (np.any(blo >= bhi)):
        raise ValueError("background bounds must be strictly ordered")
    lower = np.r_[plo, blo.ravel()]
    upper = np.r_[phi, bhi.ravel()]

    def predict(z):
        physical = np.stack([
            _real(forward_model(z[:p0.size], i), f"forward_model[channel={i}]", 1)
            for i in range(nc)
        ])
        if physical.shape != y.shape:
            raise ValueError("forward_model must return one n_bin vector per channel")
        nuisance = (np.einsum("cbk,ck->cb", basis, z[p0.size:].reshape(nc, nk))
                    if nk else np.zeros_like(y))
        return physical + nuisance, physical, nuisance

    def residual(z):
        return whiten((predict(z)[0] - y).ravel())

    fit = least_squares(residual, np.r_[p0, np.zeros(nc * nk)],
                        bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"joint multichannel GLS failed: {fit.message}")
    jac = np.asarray(fit.jac, float)
    singular = np.linalg.svd(jac, compute_uv=False)
    tol = np.finfo(float).eps * max(jac.shape) * (singular[0] if singular.size else 0.)
    rank = int(np.count_nonzero(singular > tol))
    identifiable = rank == fit.x.size
    pcov = np.linalg.pinv(jac.T @ jac, rcond=1e-12) if identifiable else None
    fitted, physical, background = predict(fit.x)
    return {
        "model": "joint_multichannel_full_covariance_GLS",
        "physical_parameters": fit.x[:p0.size].copy(),
        "background_coefficients": fit.x[p0.size:].reshape(nc, nk).copy(),
        "fitted_spectra": fitted, "fitted_physical_spectra": physical,
        "fitted_background_spectra": background,
        "residual_spectra": y - fitted,
        "measurement_covariance": cov,
        "parameter_covariance": pcov,
        "parameter_order": (["physical"] * p0.size +
                            [f"background_channel_{i}_basis_{j}"
                             for i in range(nc) for j in range(nk)]),
        "identifiability": {"rank": rank, "parameter_count": int(fit.x.size),
                            "identifiable": identifiable,
                            "singular_values": singular,
                            "condition_number": float(singular[0] / singular[-1])
                            if singular.size and singular[-1] > 0 else float("inf")},
        "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
        "degrees_of_freedom": max(0, nc * nb - fit.x.size),
        "covariance_semantics": "local_GLS_covariance_conditional_on_supplied_full_channel_major_covariance",
    }


def reconstruct_profiles_regularization_sensitivity(
    coordinate, observations, observation_covariance, *, forward_matrix,
    regularization_matrix=None, strengths=None,
):
    """Linear-Gaussian profile inverse with GCV and a full lambda sensitivity scan.

    Returns the data-only resolution matrix, posterior/local covariance for
    each tested strength, effective degrees of freedom, residual objective,
    roughness and a GCV score.  The regularizer is not counted as data
    identifiability.
    """
    x = _real(coordinate, "coordinate", 1)
    y = _real(observations, "observations", 1)
    a = _real(forward_matrix, "forward_matrix", 2)
    if a.shape[0] != y.size or x.size != a.shape[1] or x.size < 2 or np.any(np.diff(x) <= 0):
        raise ValueError("coordinate, observations and forward_matrix dimensions are inconsistent")
    cov = validate_covariance(observation_covariance, y.size,
                              name="observation_covariance")
    w = np.linalg.inv(cov)
    if regularization_matrix is None:
        d = np.diff(np.eye(x.size), n=2, axis=0)
    else:
        d = _real(regularization_matrix, "regularization_matrix", 2)
        if d.shape[1] != x.size:
            raise ValueError("regularization_matrix must have one column per profile point")
    lam = np.asarray(np.logspace(-8, 4, 25) if strengths is None else strengths, float)
    if lam.ndim != 1 or lam.size == 0 or np.any(~np.isfinite(lam)) or np.any(lam < 0):
        raise ValueError("strengths must be a non-empty finite non-negative vector")
    h_data = a.T @ w @ a
    rhs = a.T @ w @ y
    data_s = np.linalg.svd(np.linalg.cholesky(w).T @ a, compute_uv=False)
    rank = int(np.linalg.matrix_rank(a))
    results = []
    for value in lam:
        normal = h_data + value * (d.T @ d)
        inv = np.linalg.pinv(normal, rcond=1e-12)
        profile = inv @ rhs
        residual = y - a @ profile
        whitened = float(residual @ w @ residual)
        resolution = inv @ h_data
        edf = float(np.trace(resolution))
        denom = max((y.size - edf) ** 2, np.finfo(float).tiny)
        results.append({
            "strength": float(value), "profile": profile,
            "profile_covariance": inv @ h_data @ inv,
            "resolution_matrix": resolution,
            "effective_degrees_of_freedom": edf,
            "whitened_residual_sum_squares": whitened,
            "roughness_squared": float(np.linalg.norm(d @ profile) ** 2),
            "gcv": float(whitened * y.size / denom),
        })
    best = int(np.argmin([item["gcv"] for item in results]))
    return {
        "model": "linear_Gaussian_regularized_profile_inverse",
        "coordinate": x.copy(), "data_only_rank": rank,
        "profile_parameter_count": int(x.size),
        "data_identifiable": bool(rank == x.size),
        "data_singular_values": data_s,
        "selected_strength_by_GCV": results[best]["strength"],
        "selected_profile": results[best]["profile"].copy(),
        "strength_scan": results,
        "regularization_sensitivity": {
            "profile_min": np.min(np.stack([r["profile"] for r in results]), axis=0),
            "profile_max": np.max(np.stack([r["profile"] for r in results]), axis=0),
            "profile_span": np.ptp(np.stack([r["profile"] for r in results]), axis=0),
            "strengths": lam.copy(),
        },
        "warning": "Regularization stabilizes the inverse but cannot create data identifiability.",
    }


def sample_reference_basis_uncertainty(
    reference_spectra, measured_spectrum, *, n_samples=500,
    random_seed=20261103, covariance=None, nonnegative=True,
):
    """Propagate finite reference-library uncertainty through PCA projection.

    Each reference row is a library realization. Optional covariance is the
    full flattened library covariance (n_reference*n_bin square), otherwise
    row-wise empirical sampling with replacement is used. PCA is refit for
    every draw; outputs are basis and reconstruction ensembles, not a claim
    that the library represents all possible backgrounds.
    """
    ref = _real(reference_spectra, "reference_spectra", 2)
    y = _real(measured_spectrum, "measured_spectrum", 1)
    nr, nb = ref.shape
    if nr < 3 or y.shape != (nb,):
        raise ValueError("reference_spectra needs >=3 rows matching measured_spectrum")
    if isinstance(n_samples, bool) or int(n_samples) < 20:
        raise ValueError("n_samples must be an integer >=20")
    rng = np.random.default_rng(random_seed)
    if covariance is not None:
        cv = validate_covariance(covariance, nr * nb, name="reference_library_covariance",
                                 positive_definite=False)
        eig, vec = np.linalg.eigh(cv)
        root = vec @ np.diag(np.sqrt(np.maximum(eig, 0.0)))
    else:
        cv, root = None, None
    reconstructions, eigenvalues, bases, ranks = [], [], [], []
    for _ in range(int(n_samples)):
        if root is None:
            draw = ref[rng.integers(0, nr, size=nr)].copy()
        else:
            draw = ref + (root @ rng.standard_normal(nr * nb)).reshape(nr, nb)
        mean = draw.mean(axis=0)
        centered = draw - mean
        _, singular, vt = np.linalg.svd(centered, full_matrices=False)
        values = singular ** 2 / max(nr - 1, 1)
        # Retain all numerically supported components; callers may truncate
        # the returned basis consistently with their chosen explained variance.
        basis = vt.T
        coeff = basis.T @ (y - mean)
        reconstruction = mean + basis @ coeff
        if nonnegative:
            reconstruction = np.maximum(reconstruction, 0.0)
        reconstructions.append(reconstruction)
        eigenvalues.append(values)
        bases.append(basis)
        ranks.append(int(np.linalg.matrix_rank(centered)))
    rec = np.stack(reconstructions)
    eig = np.stack(eigenvalues)
    return {
        "model": "reference_library_PCA_refit_monte_carlo",
        "mean_reconstruction": rec.mean(axis=0),
        "reconstruction_covariance": np.cov(rec, rowvar=False, ddof=1),
        "reconstruction_standard_deviation": rec.std(axis=0, ddof=1),
        "eigenvalue_mean": eig.mean(axis=0),
        "eigenvalue_standard_deviation": eig.std(axis=0, ddof=1),
        "basis_ensembles": np.stack(bases),
        "rank_distribution": np.asarray(ranks),
        "sample_count": int(n_samples), "random_seed": int(random_seed),
        "reference_covariance_supplied": covariance is not None,
        "uncertainty_semantics": "finite_reference_library_sampling_or_supplied_library_covariance; not measurement noise",
    }


def propagate_geometry_ensemble(
    forward_model: Callable, geometry_mean, geometry_covariance, *,
    n_samples=500, random_seed=20261103, minimum_success_fraction=0.8,
):
    """Monte-Carlo propagation of correlated geometry/beam parameter errors.

    forward_model(geometry_vector) returns a finite output vector. Failed
    draws are counted and never silently replaced with the nominal prediction.
    """
    mean = _real(geometry_mean, "geometry_mean", 1)
    cov = validate_covariance(geometry_covariance, mean.size,
                              name="geometry_covariance", positive_definite=False)
    if not callable(forward_model):
        raise ValueError("forward_model must be callable")
    if isinstance(n_samples, bool) or int(n_samples) < 20:
        raise ValueError("n_samples must be an integer >=20")
    if not 0 < float(minimum_success_fraction) <= 1:
        raise ValueError("minimum_success_fraction must lie in (0,1]")
    rng = np.random.default_rng(random_seed)
    draws = rng.multivariate_normal(mean, cov, size=int(n_samples), check_valid="raise")
    outputs, failed = [], 0
    expected_shape = None
    for draw in draws:
        try:
            result = _real(forward_model(draw), "forward_model_output", 1)
            if expected_shape is None:
                expected_shape = result.shape
            if result.shape != expected_shape:
                raise ValueError("forward_model output shape changed between geometry draws")
            outputs.append(result)
        except (ValueError, FloatingPointError, ArithmeticError, np.linalg.LinAlgError):
            failed += 1
    fraction = len(outputs) / int(n_samples)
    if fraction < float(minimum_success_fraction):
        raise RuntimeError(f"geometry Monte Carlo success fraction {fraction:.3f} is below required {minimum_success_fraction:.3f}")
    ensemble = np.stack(outputs)
    return {
        "mean": ensemble.mean(axis=0),
        "covariance": np.cov(ensemble, rowvar=False, ddof=1),
        "standard_deviation": ensemble.std(axis=0, ddof=1),
        "successful_samples": len(outputs), "failed_samples": failed,
        "success_fraction": float(fraction), "random_seed": int(random_seed),
        "uncertainty_semantics": "Monte_Carlo_propagation_of_supplied_correlated_geometry_covariance",
    }

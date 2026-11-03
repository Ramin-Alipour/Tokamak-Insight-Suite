"""Uncertainty-aware reusable operators for Thomson profile inversions.

All inputs are calibrated measurements or explicitly supplied forward-operator
metadata. These utilities do not estimate missing correlations, calibrate a
diagnostic, or turn regularization diagnostics into physical measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear


def _real_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    out = raw.astype(float, copy=False)
    if ndim is not None and out.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _psd(value, shape, name, *, positive_definite=False):
    a = _real_array(value, name, 2)
    if a.shape != shape or not np.allclose(a, a.T, rtol=1e-10, atol=1e-13):
        raise ValueError(f"{name} must be symmetric with shape {shape}")
    a = 0.5 * (a + a.T)
    eig = np.linalg.eigvalsh(a)
    scale = max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if eig.min() < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    if positive_definite and eig.min() <= np.finfo(float).eps * scale:
        raise ValueError(f"{name} must be positive definite")
    return a


def assemble_joint_covariance(marginal_covariances, cross_covariances=None):
    """Assemble and validate a block covariance without inventing cross terms.

    `marginal_covariances` is an ordered sequence of square covariance blocks.
    `cross_covariances` maps (i,j), i<j, to Cov(block_i, block_j). Missing
    cross-blocks are exactly zero and are explicitly reported as assumed-zero.
    """
    blocks = [_real_array(c, f"marginal_covariances[{i}]", 2)
              for i, c in enumerate(marginal_covariances)]
    if not blocks:
        raise ValueError("at least one marginal covariance block is required")
    sizes = []
    for i, block in enumerate(blocks):
        if block.shape[0] != block.shape[1]:
            raise ValueError(f"marginal covariance {i} must be square")
        _psd(block, block.shape, f"marginal_covariances[{i}]")
        sizes.append(block.shape[0])
    offsets = np.cumsum([0] + sizes)
    result = np.zeros((offsets[-1], offsets[-1]), dtype=float)
    for i, block in enumerate(blocks):
        result[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]] = block
    supplied = set()
    for key, value in (cross_covariances or {}).items():
        if (not isinstance(key, tuple) or len(key) != 2
                or any(isinstance(k, bool) or not isinstance(k, int) for k in key)):
            raise ValueError("cross-covariance keys must be integer (i,j) tuples")
        i, j = key
        if not 0 <= i < j < len(blocks):
            raise ValueError("cross-covariance keys must satisfy 0 <= i < j < n_blocks")
        cross = _real_array(value, f"cross_covariances[{key}]", 2)
        if cross.shape != (sizes[i], sizes[j]):
            raise ValueError(f"cross-covariance {key} has incompatible block dimensions")
        result[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = cross
        result[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = cross.T
        supplied.add(key)
    _psd(result, result.shape, "assembled_joint_covariance")
    return {
        "covariance": result,
        "block_sizes": tuple(sizes),
        "supplied_cross_blocks": tuple(sorted(supplied)),
        "zero_assumed_cross_blocks": tuple(
            (i, j) for i in range(len(blocks)) for j in range(i+1, len(blocks))
            if (i, j) not in supplied
        ),
        "covariance_order": "block-major; order follows marginal_covariances",
    }


def propagate_pca_reference_uncertainty(reference_spectra, query_spectrum, *,
                                        reference_covariance=None,
                                        query_covariance=None,
                                        n_components=None):
    """Propagate supplied reference/query covariance through a PCA projection.

    Rows of reference_spectra are reference spectra and columns are wavelength
    bins. The returned covariance is conditional on the fitted reference mean
    and basis. Reference uncertainty is propagated by a first-order numerical
    Jacobian of the projected scores; basis-estimation uncertainty is not
    silently treated as zero and is explicitly flagged as conditional.
    """
    refs = _real_array(reference_spectra, "reference_spectra", 2)
    query = _real_array(query_spectrum, "query_spectrum", 1)
    nr, nw = refs.shape
    if nr < 2 or query.shape != (nw,):
        raise ValueError("reference_spectra needs >=2 rows and query_spectrum must match its wavelength axis")
    k = min(nr - 1, nw) if n_components is None else n_components
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or not 1 <= k <= min(nr-1, nw):
        raise ValueError("n_components must be an integer from 1 to min(n_reference-1,n_wavelength)")
    mean = refs.mean(axis=0)
    centered = refs - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:k].T
    scores = (query - mean) @ basis
    ref_cov = None
    if reference_covariance is not None:
        ref_cov = _psd(reference_covariance, (nr*nw, nr*nw), "reference_covariance")
    query_cov = None if query_covariance is None else _psd(
        query_covariance, (nw, nw), "query_covariance"
    )
    # Delta-method Jacobians include changes in the reference mean and PCA
    # subspace, evaluated away from singular-value degeneracies.
    def project(flat):
        matrix = flat.reshape(nr, nw)
        m = matrix.mean(axis=0)
        _, _, vectors = np.linalg.svd(matrix-m, full_matrices=False)
        local_basis = vectors[:k].copy()
        for component in range(k):
            if float(local_basis[component] @ vt[component]) < 0.0:
                local_basis[component] *= -1.0
        return (query-m) @ local_basis.T
    cov_scores = np.zeros((k, k), dtype=float)
    if ref_cov is not None:
        flat = refs.reshape(-1)
        jac = np.empty((k, flat.size), dtype=float)
        for j in range(flat.size):
            h = np.cbrt(np.finfo(float).eps) * max(1.0, abs(float(flat[j])))
            plus, minus = flat.copy(), flat.copy()
            plus[j] += h
            minus[j] -= h
            jac[:, j] = (project(plus)-project(minus))/(2*h)
        cov_scores += jac @ ref_cov @ jac.T
    if query_cov is not None:
        cov_scores += basis.T @ query_cov @ basis
    return {
        "scores": scores,
        "reference_mean": mean,
        "components": basis,
        "explained_variance": singular[:k]**2 / max(nr-1, 1),
        "score_covariance": 0.5*(cov_scores+cov_scores.T),
        "reference_uncertainty_propagated": ref_cov is not None,
        "query_uncertainty_propagated": query_cov is not None,
        "basis_uncertainty_treatment": (
            "first_order_delta_method_reference_spectrum_perturbations"
            if ref_cov is not None else "conditional_fixed_reference_basis"
        ),
        "limitations": [
            "PCA scores are conditional on the selected number of components.",
            "Near-degenerate singular values can make individual component vectors non-identifiable.",
            "The delta-method covariance is local and is not a posterior distribution.",
        ],
    }


def profile_regularization_sensitivity(measurement, forward_matrix, covariance, *,
                                       regularization_matrix=None,
                                       strengths=None, methods=("identity", "first_difference", "second_difference"),
                                       nonnegative=True):
    """Compare a regularization path using one fixed measured-data likelihood.

    Reports data-only rank/condition separately from the regularized system.
    The returned candidates are diagnostics; the function does not select a
    scientifically true profile or claim modes in the data null space.
    """
    y = _real_array(measurement, "measurement", 1)
    a = _real_array(forward_matrix, "forward_matrix", 2)
    if a.shape[0] != y.size or min(a.shape) < 2:
        raise ValueError("forward_matrix must have shape (n_measurements,n_profile_nodes)")
    cov = _psd(covariance, (y.size, y.size), "covariance", positive_definite=True)
    chol = np.linalg.cholesky(cov)
    aw = np.linalg.solve(chol, a)
    yw = np.linalg.solve(chol, y)
    singular = np.linalg.svd(aw, compute_uv=False)
    tol = np.finfo(float).eps * max(aw.shape) * singular[0]
    rank = int(np.sum(singular > tol))
    data_condition = float(singular[0]/singular[-1]) if singular[-1] > tol else float("inf")
    n = a.shape[1]
    if strengths is None:
        strengths = np.logspace(-8, 4, 25)
    lam = _real_array(strengths, "strengths", 1)
    if np.any(lam < 0) or np.any(np.diff(lam) <= 0):
        raise ValueError("strengths must be non-negative and strictly increasing")
    if regularization_matrix is not None:
        penalties = {"custom": _real_array(regularization_matrix, "regularization_matrix", 2)}
    else:
        penalties = {}
        for method in methods:
            if method == "identity":
                penalties[method] = np.eye(n)
            elif method == "first_difference":
                penalties[method] = np.diff(np.eye(n), n=1, axis=0)
            elif method == "second_difference":
                if n < 3:
                    raise ValueError("second_difference requires at least three profile nodes")
                penalties[method] = np.diff(np.eye(n), n=2, axis=0)
            else:
                raise ValueError(f"unsupported regularization method: {method}")
    results = []
    for name, l in penalties.items():
        if l.ndim != 2 or l.shape[1] != n:
            raise ValueError(f"regularization matrix {name} must have {n} columns")
        lw = np.asarray(l, dtype=float)
        for strength in lam:
            system = np.vstack((aw, strength*lw))
            target = np.concatenate((yw, np.zeros(lw.shape[0])))
            if nonnegative:
                fit = lsq_linear(system, target, bounds=(0.0, np.inf), method="trf")
                profile = fit.x
                success = bool(fit.success)
            else:
                profile, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
                success = True
            residual = yw-aw@profile
            roughness = float(np.linalg.norm(lw@profile))
            results.append({
                "method": name, "strength": float(strength),
                "profile": profile, "whitened_residual_norm": float(np.linalg.norm(residual)),
                "roughness_norm": roughness, "objective": float(residual@residual+(strength*roughness)**2),
                "success": success,
            })
    return {
        "candidates": results,
        "data_only_rank": rank,
        "profile_dimension": n,
        "data_only_identifiable": rank == n,
        "data_only_condition_number": data_condition,
        "covariance_model": "researcher_supplied_full_measurement_covariance",
        "selection": "none; compare residual/roughness and use external physical constraints",
        "null_space_warning": "regularization cannot create data information in the forward-operator null space",
    }

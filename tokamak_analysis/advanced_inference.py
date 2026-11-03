"""Advanced inverse/statistical operators for Thomson and CTS analysis.

All operators act on researcher-supplied, already calibrated measurements and
physical forward kernels. They do not calibrate instruments or validate models
against experiments. A reconstructed distribution is only data-identifiable
when the stacked multi-view measurement operator supports its degrees of
freedom; regularization never upgrades a null-space mode into a measurement.
"""
from __future__ import annotations

from collections.abc import Sequence
import numpy as np
from scipy.optimize import lsq_linear


def _real_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    out = raw.astype(float, copy=False)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite and non-empty")
    return out


def poisson_deviance_residual(observed_counts, expected_counts):
    """Signed square-root Poisson deviance residual for independent count bins.

    This is the exact Poisson likelihood residual, not a Gaussian approximation.
    Inputs are counts accumulated over known bins, not calibrated power-density
    values. The expected counts must already include exposure, collection
    efficiency and the complete spectral forward operator.
    """
    y = _real_array(observed_counts, "observed_counts", 1)
    mu = _real_array(expected_counts, "expected_counts", 1)
    if y.shape != mu.shape or np.any(y < 0) or np.any(mu <= 0):
        raise ValueError("observed counts must be non-negative and expected counts positive with matching shapes")
    if np.any(np.abs(y - np.rint(y)) > 1e-8):
        raise ValueError("observed_counts must be integer-valued photon/electron counts; do not pass calibrated spectral power")
    term = np.empty_like(y)
    positive = y > 0
    term[positive] = y[positive] * np.log(y[positive] / mu[positive]) - (y[positive] - mu[positive])
    term[~positive] = mu[~positive]
    deviance = 2.0 * np.maximum(term, 0.0)
    return np.sign(y - mu) * np.sqrt(deviance)


def poisson_negative_log_likelihood(observed_counts, expected_counts, *, include_constant=True):
    """Independent Poisson negative log likelihood, optionally including log(y!)."""
    y = _real_array(observed_counts, "observed_counts", 1)
    mu = _real_array(expected_counts, "expected_counts", 1)
    if y.shape != mu.shape or np.any(y < 0) or np.any(mu <= 0):
        raise ValueError("observed counts must be non-negative and expected counts positive with matching shapes")
    if np.any(np.abs(y - np.rint(y)) > 1e-8):
        raise ValueError("observed_counts must be integer-valued counts")
    value = np.sum(mu - y * np.log(mu))
    if include_constant:
        from scipy.special import gammaln
        value += np.sum(gammaln(y + 1.0))
    return float(value)


def gaussian_process_discrepancy_covariance(
    coordinate, *, amplitude, length_scale, kernel="matern32", nugget=0.0,
):
    """Return a positive-definite correlated model-discrepancy covariance.

    The discrepancy is a separate additive residual process with its own
    amplitude and correlation length. It is not inferred from reduced chi-square.
    The returned covariance is in the squared units of the supplied spectrum.
    """
    x = _real_array(coordinate, "coordinate", 1)
    if np.any(np.diff(x) <= 0):
        raise ValueError("coordinate must be strictly increasing")
    for name, value, allow_zero in (
        ("amplitude", amplitude, False), ("length_scale", length_scale, False),
        ("nugget", nugget, True),
    ):
        if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite real scalar")
        if value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} must be {'non-negative' if allow_zero else 'positive'}")
    r = np.abs(x[:, None] - x[None, :]) / float(length_scale)
    if kernel == "squared_exponential":
        correlation = np.exp(-0.5 * r * r)
    elif kernel == "matern32":
        z = np.sqrt(3.0) * r
        correlation = (1.0 + z) * np.exp(-z)
    elif kernel == "matern52":
        z = np.sqrt(5.0) * r
        correlation = (1.0 + z + z * z / 3.0) * np.exp(-z)
    else:
        raise ValueError("kernel must be squared_exponential, matern32 or matern52")
    covariance = float(amplitude) ** 2 * correlation
    covariance.flat[:: x.size + 1] += float(nugget) ** 2
    # Fail loudly if numerical conditioning or an invalid kernel ever breaks PSD.
    eigenvalues = np.linalg.eigvalsh(0.5 * (covariance + covariance.T))
    if eigenvalues.min() < -1e-10 * max(float(np.max(np.abs(eigenvalues))), 1.0):
        raise ValueError("model-discrepancy covariance is not positive semidefinite")
    return covariance


def propagate_paraxial_beam_covariance(
    initial_phase_space_covariance, transfer_matrices,
):
    """Propagate a 4-D transverse Gaussian beam covariance through supplied ABCD maps.

    The state ordering is [x, y, theta_x, theta_y]. Each 4x4 transfer matrix
    must be a paraxial linearized Hamiltonian map from one ray sample to the
    next. Matrices are supplied by a dispersion-Hessian/quasi-optical solver;
    this function does not invent plasma focusing or mode-coupling coefficients.
    """
    covariance = _real_array(initial_phase_space_covariance, "initial_phase_space_covariance", 2)
    if covariance.shape != (4, 4) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13):
        raise ValueError("initial_phase_space_covariance must be symmetric 4x4")
    eig = np.linalg.eigvalsh(covariance)
    if eig.min() < -1e-10 * max(float(np.max(np.abs(eig))), 1.0):
        raise ValueError("initial phase-space covariance must be positive semidefinite")
    matrices = _real_array(transfer_matrices, "transfer_matrices", 3)
    if matrices.shape[1:] != (4, 4):
        raise ValueError("transfer_matrices must have shape (n_segment,4,4)")
    # Canonical symplectic form for (x,y,theta_x,theta_y).
    zero = np.zeros((2, 2))
    identity = np.eye(2)
    symplectic = np.block([[zero, identity], [-identity, zero]])
    states = [covariance.copy()]
    for index, matrix in enumerate(matrices):
        defect = matrix.T @ symplectic @ matrix - symplectic
        if np.linalg.norm(defect, ord="fro") > 1e-6 * max(1.0, np.linalg.norm(matrix, ord="fro") ** 2):
            raise ValueError(f"transfer_matrices[{index}] is not symplectic within tolerance")
        covariance = matrix @ covariance @ matrix.T
        covariance = 0.5 * (covariance + covariance.T)
        states.append(covariance.copy())
    phase_space = np.asarray(states)
    spatial = phase_space[:, :2, :2]
    return {
        "phase_space_covariance": phase_space,
        "transverse_position_covariance_m2": spatial,
        "state_order": ["x_m", "y_m", "theta_x_rad", "theta_y_rad"],
        "propagation_model": "linearized_paraxial_Hamiltonian_ABCD_covariance_transport",
        "transfer_matrices_supplied_externally": True,
        "limitations": [
            "Valid within the paraxial linearized regime represented by the supplied symplectic maps.",
            "Does not derive dispersion Hessians, mode conversion, absorption or non-Gaussian beam structure.",
            "A full 3-D spatial overlap requires embedding the transverse covariance in a supplied local ray frame.",
        ],
    }


def bootstrap_reference_pca_basis(
    reference_spectra, *, components, bootstrap_samples=300, random_seed=1729,
):
    """Bootstrap uncertainty of a reference-derived PCA mean and subspace.

    Rows are independently resampled reference spectra on the same calibrated
    frequency grid. Component rotations are Procrustes-aligned to the nominal
    basis before covariance estimation. Near-degenerate singular subspaces
    remain intrinsically weakly identifiable and are flagged.
    """
    from scipy.linalg import orthogonal_procrustes
    reference = _real_array(reference_spectra, "reference_spectra", 2)
    if reference.shape[0] < 3 or reference.shape[1] < 2:
        raise ValueError("at least three reference spectra and two frequency bins are required")
    if isinstance(components, bool) or not isinstance(components, (int, np.integer)):
        raise ValueError("components must be an integer")
    maximum = min(reference.shape[0] - 1, reference.shape[1])
    if not 1 <= int(components) <= maximum:
        raise ValueError(f"components must be between 1 and {maximum}")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, (int, np.integer)) or bootstrap_samples < 30:
        raise ValueError("bootstrap_samples must be an integer >=30")
    mean = reference.mean(axis=0)
    _, singular, vt = np.linalg.svd(reference - mean, full_matrices=False)
    nominal = vt[:int(components)].T
    scale = max(float(singular[0]) if singular.size else 0.0, np.finfo(float).tiny)
    gaps = []
    for index in range(int(components)):
        next_value = singular[index + 1] if index + 1 < singular.size else 0.0
        gaps.append(float((singular[index] - next_value) / scale))
    rng = np.random.default_rng(int(random_seed))
    draws = []
    aligned_components = []
    for _ in range(int(bootstrap_samples)):
        sample = reference[rng.integers(0, reference.shape[0], size=reference.shape[0])]
        sample_mean = sample.mean(axis=0)
        _, _, sample_vt = np.linalg.svd(sample - sample_mean, full_matrices=False)
        basis = sample_vt[:int(components)].T
        rotation, _ = orthogonal_procrustes(basis, nominal)
        basis = basis @ rotation
        aligned_components.append(basis)
        draws.append(np.concatenate((sample_mean, basis.ravel())))
    draws = np.asarray(draws)
    covariance = np.cov(draws, rowvar=False, ddof=1)
    return {
        "mean_spectrum": mean,
        "pca_components": nominal,
        "bootstrap_parameter_mean": draws.mean(axis=0),
        "bootstrap_parameter_covariance": covariance,
        "bootstrap_parameter_order": (
            [f"reference_mean_bin_{i}" for i in range(reference.shape[1])]
            + [f"pca_component_{j}_bin_{i}" for j in range(int(components))
               for i in range(reference.shape[1])]
        ),
        "component_relative_singular_value_gaps": np.asarray(gaps),
        "weakly_identified_components": [
            int(i) for i, gap in enumerate(gaps) if gap < 1e-3
        ],
        "bootstrap_samples": int(bootstrap_samples),
        "random_seed": int(random_seed),
        "uncertainty_scope": "finite-reference-sample uncertainty conditional on independent row resampling and fixed frequency grid; not measurement calibration uncertainty",
    }


def reconstruct_multiview_3d_evdf(
    spectral_kernels, measured_spectra, velocity_grid_m_s, *,
    voxel_volume_m3_s3, regularization_matrix,
    measurement_uncertainty=None, measurement_covariance=None,
    regularization_strength, max_iter=3000,
):
    """Reconstruct a non-negative 3-D velocity distribution from multi-view spectra.

    spectral_kernels has shape (n_view,n_frequency,n_velocity_voxel) and must
    be produced by a forward model for each measured scattering geometry.
    For a relativistic Thomson reconstruction, callers must supply kernels
    computed with the relativistic kinematics, polarization and cross-section
    appropriate to each view. A single-view or rank-deficient system is never
    promoted to a unique 3-D result by regularization.
    """
    kernels = _real_array(spectral_kernels, "spectral_kernels", 3)
    data = _real_array(measured_spectra, "measured_spectra", 2)
    velocity = _real_array(velocity_grid_m_s, "velocity_grid_m_s", 2)
    volume = _real_array(voxel_volume_m3_s3, "voxel_volume_m3_s3", 1)
    penalty = _real_array(regularization_matrix, "regularization_matrix", 2)
    nview, nfreq, nvoxel = kernels.shape
    if data.shape != (nview, nfreq) or velocity.shape != (nvoxel, 3) or volume.shape != (nvoxel,):
        raise ValueError("multi-view data, 3-D velocity grid and voxel-volume dimensions are inconsistent")
    if np.any(volume <= 0) or np.any(data < 0) or penalty.shape[1] != nvoxel:
        raise ValueError("voxel volumes must be positive, spectra non-negative and regularization columns must match voxels")
    lam = float(regularization_strength)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("regularization_strength must be positive and finite")
    design = kernels.reshape(nview * nfreq, nvoxel) * volume[None, :]
    target = data.ravel()
    if measurement_uncertainty is not None and measurement_covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    if measurement_uncertainty is not None:
        sigma = _real_array(measurement_uncertainty, "measurement_uncertainty", 2)
        if sigma.shape != data.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match measured_spectra")
        whiten = lambda z: np.asarray(z) / sigma.ravel()
        noise_status = "supplied_pointwise_uncertainty"
    elif measurement_covariance is not None:
        covariance = _real_array(measurement_covariance, "measurement_covariance", 2)
        if covariance.shape != (target.size, target.size) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13):
            raise ValueError("measurement_covariance must be symmetric and match the stacked multi-view spectrum")
        chol = np.linalg.cholesky(covariance)
        whiten = lambda z: np.linalg.solve(chol, np.asarray(z))
        noise_status = "supplied_full_multi_view_covariance"
    else:
        whiten = lambda z: np.asarray(z)
        noise_status = "unweighted_no_measurement_uncertainty_supplied"
    a = whiten(design)
    b = whiten(target)
    augmented = np.vstack((a, np.sqrt(lam) * penalty))
    rhs = np.r_[b, np.zeros(penalty.shape[0])]
    fit = lsq_linear(augmented, rhs, bounds=(0.0, np.inf), method="trf", max_iter=int(max_iter), tol=1e-10)
    if not fit.success:
        raise ValueError(f"3-D EVDF inversion failed: {fit.message}")
    singular = np.linalg.svd(a, compute_uv=False)
    tolerance = np.finfo(float).eps * max(a.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > tolerance else float("inf")
    identifiable = rank == nvoxel and np.isfinite(condition) and condition < 1e12
    raw = np.maximum(fit.x, 0.0)
    norm = float(volume @ raw)
    if norm <= np.finfo(float).tiny:
        raise ValueError("3-D EVDF reconstruction has zero normalization")
    distribution = raw / norm
    fitted = (design @ raw).reshape(nview, nfreq)
    residual = data - fitted
    active = raw > max(float(raw.max()) * 1e-9, np.finfo(float).tiny)
    aa = a[:, active]
    pp = penalty[:, active]
    hessian = aa.T @ aa + lam * (pp.T @ pp)
    active_covariance = np.linalg.pinv(hessian, hermitian=True)
    if noise_status == "unweighted_no_measurement_uncertainty_supplied":
        whitened_residual = b - a @ raw
        active_covariance *= float(whitened_residual @ whitened_residual / max(1, target.size - rank))
    coefficient_covariance = np.zeros((nvoxel, nvoxel))
    coefficient_covariance[np.ix_(np.flatnonzero(active), np.flatnonzero(active))] = active_covariance
    jacobian = (np.eye(nvoxel) * norm - np.outer(raw, volume)) / norm**2
    distribution_covariance = jacobian @ coefficient_covariance @ jacobian.T
    return {
        "status": "identified_from_multi_view_measurements" if identifiable else "conditional_regularized_3d_candidate",
        "evdf_velocity_grid_m_s": velocity.copy(),
        "evdf_m3_s3": distribution if identifiable else None,
        "regularized_candidate_evdf_m3_s3": distribution,
        "normalization": float(volume @ distribution),
        "fitted_spectra": fitted,
        "residual_spectra": residual,
        "distribution_covariance": distribution_covariance if identifiable else None,
        "data_only_jacobian_rank": rank,
        "velocity_voxel_count": int(nvoxel),
        "view_count": int(nview),
        "data_only_condition_number": condition,
        "null_space_dimension": int(max(0, nvoxel - rank)),
        "regularization_strength": lam,
        "noise_model": noise_status,
        "forward_kernel_contract": "caller-supplied per-view spectral kernels; relativistic claims require relativistic per-view kernels",
        "identifiability": {
            "primary": ["3-D EVDF"] if identifiable else [],
            "conditional": ["regularized 3-D candidate conditional on supplied kernels, velocity grid and regularization"],
            "non_identifiable": [] if identifiable else ["3-D velocity-space modes in the stacked multi-view data null space"],
        },
        "experimental_validation": "not_performed",
    }

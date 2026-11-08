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
        _, sample_singular, sample_vt = np.linalg.svd(sample - sample_mean, full_matrices=False)
        if sample_vt.shape[0] < int(components):
            continue
        sample_rank = int(np.count_nonzero(
            sample_singular > np.finfo(float).eps * max(sample.shape)
            * (sample_singular[0] if sample_singular.size else 0.0)
        ))
        if sample_rank < int(components):
            continue
        basis = sample_vt[:int(components)].T
        rotation, _ = orthogonal_procrustes(basis, nominal)
        basis = basis @ rotation
        aligned_components.append(basis)
        draws.append(np.concatenate((sample_mean, basis.ravel())))
    if len(draws) < max(30, int(np.ceil(0.7 * bootstrap_samples))):
        raise ValueError("too few full-rank bootstrap PCA bases; reduce the requested component count or supply more independent reference spectra")
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
    model_discrepancy_covariance=None, systematic_covariance=None,
    regularization_strength, max_iter=3000,
):
    """Reconstruct a non-negative 3-D velocity distribution from multi-view spectra.

    spectral_kernels has shape (n_view,n_frequency,n_velocity_voxel) and must
    be produced by a forward model for each measured scattering geometry.
    For a relativistic Thomson reconstruction, callers must supply kernels
    computed with the relativistic kinematics, polarization and cross-section
    appropriate to each view. A single-view or rank-deficient system is never
    promoted to a unique 3-D result by regularization.

    Independent spectral covariance components are additive in stacked
    view-major order: C_total = C_measurement + C_model_discrepancy
    + C_systematic. For the fixed non-negative active set, the frequentist
    sampling covariance of the Tikhonov coefficients is the sandwich
    H^-1 A_w.T A_w H^-1, where H=A_w.T A_w+lambda L.T L. H^-1 alone
    is not the sampling covariance of a regularized estimator.
    """
    kernels = _real_array(spectral_kernels, "spectral_kernels", 3)
    data = _real_array(measured_spectra, "measured_spectra", 2)
    velocity = _real_array(velocity_grid_m_s, "velocity_grid_m_s", 2)
    volume = _real_array(voxel_volume_m3_s3, "voxel_volume_m3_s3", 1)
    penalty = _real_array(regularization_matrix, "regularization_matrix", 2)
    nview, nfreq, nvoxel = kernels.shape
    if data.shape != (nview, nfreq) or velocity.shape != (nvoxel, 3) or volume.shape != (nvoxel,):
        raise ValueError("multi-view data, 3-D velocity grid and voxel-volume dimensions are inconsistent")
    # Velocity-space voxels must lie on the physical subluminal mass shell.
    # A converged non-negative inverse cannot make |v| >= c physical.
    speed = np.linalg.norm(velocity, axis=1)
    if np.any(speed >= 299792458.0):
        raise ValueError("every 3-D EVDF velocity voxel must be strictly subluminal (|v| < c)")
    # Background-subtracted spectra are allowed to be signed: negative
    # residual bins are expected for finite-noise measurements. The physical
    # EVDF remains non-negative through the bounded inverse below.
    if np.any(volume <= 0) or penalty.shape[1] != nvoxel:
        raise ValueError("voxel volumes must be positive and regularization columns must match voxels")
    lam = float(regularization_strength)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("regularization_strength must be positive and finite")
    design = kernels.reshape(nview * nfreq, nvoxel) * volume[None, :]
    target = data.ravel()
    if measurement_uncertainty is not None and measurement_covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    covariance_components = {}
    if measurement_uncertainty is not None:
        sigma = _real_array(measurement_uncertainty, "measurement_uncertainty", 2)
        if sigma.shape != data.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match measured_spectra")
        covariance_components["measurement"] = np.diag(sigma.ravel() ** 2)
    elif measurement_covariance is not None:
        covariance_components["measurement"] = _real_array(
            measurement_covariance, "measurement_covariance", 2
        )
    for name, value in (
        ("model_discrepancy_covariance", model_discrepancy_covariance),
        ("systematic_covariance", systematic_covariance),
    ):
        if value is not None:
            covariance_components[name] = _real_array(value, name, 2)
    for name, component in covariance_components.items():
        if component.shape != (target.size, target.size):
            raise ValueError(f"{name} must match the stacked (view-major) multi-view spectrum")
        component_scale = max(
            float(np.max(np.abs(component))), np.finfo(float).tiny
        )
        if not np.allclose(
            component, component.T, rtol=1e-10,
            atol=1e-12 * component_scale,
        ):
            raise ValueError(f"{name} must be symmetric")
        component = 0.5 * (component + component.T)
        eigenvalues = np.linalg.eigvalsh(component)
        eigenvalue_scale = max(
            float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny
        )
        if eigenvalues[0] < -1e-10 * eigenvalue_scale:
            raise ValueError(f"{name} must be positive semidefinite")
        covariance_components[name] = component
    if covariance_components:
        total_covariance = np.sum(list(covariance_components.values()), axis=0)
        try:
            chol = np.linalg.cholesky(total_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "the sum of supplied multi-view covariance components must be positive definite"
            ) from exc
        whiten = lambda z: np.linalg.solve(chol, np.asarray(z))
        noise_status = "full_stacked_measurement_plus_model_discrepancy_plus_systematic_covariance"
    else:
        total_covariance = None
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
    inverse_hessian = np.linalg.pinv(hessian, hermitian=True)
    active_information = aa.T @ aa
    active_covariance = inverse_hessian @ active_information @ inverse_hessian
    if total_covariance is None:
        whitened_residual = b - a @ raw
        active_covariance *= float(
            whitened_residual @ whitened_residual / max(1, target.size - rank)
        )
    active_covariance = 0.5 * (active_covariance + active_covariance.T)
    coefficient_covariance = np.zeros((nvoxel, nvoxel))
    coefficient_covariance[np.ix_(np.flatnonzero(active), np.flatnonzero(active))] = active_covariance
    jacobian = (np.eye(nvoxel) * norm - np.outer(raw, volume)) / norm**2
    distribution_covariance = jacobian @ coefficient_covariance @ jacobian.T

    # Derive the first and second 3-D velocity moments from the normalized
    # reconstructed EVDF using the same voxel-volume measure as normalization.
    # These are distribution moments, not a claim that every velocity-space
    # degree of freedom is independently identifiable.
    probability_mass = distribution * volume
    candidate_mean_velocity = np.einsum("i,ij->j", probability_mass, velocity)
    centered_velocity = velocity - candidate_mean_velocity[None, :]
    candidate_velocity_covariance = np.einsum(
        "i,ij,ik->jk", probability_mass, centered_velocity, centered_velocity
    )
    candidate_velocity_covariance = 0.5 * (
        candidate_velocity_covariance + candidate_velocity_covariance.T
    )
    mean_velocity_covariance = None
    velocity_covariance_tensor_covariance = None
    if identifiable:
        mean_jacobian = velocity.T * volume[None, :]
        covariance_jacobian = np.empty((9, nvoxel), dtype=float)
        for voxel, vector in enumerate(velocity):
            derivative = volume[voxel] * (
                np.outer(vector, vector)
                - np.outer(vector, candidate_mean_velocity)
                - np.outer(candidate_mean_velocity, vector)
            )
            covariance_jacobian[:, voxel] = derivative.reshape(9)
        mean_velocity_covariance = (
            mean_jacobian @ distribution_covariance @ mean_jacobian.T
        )
        velocity_covariance_tensor_covariance = (
            covariance_jacobian @ distribution_covariance @ covariance_jacobian.T
        )
        mean_velocity_covariance = 0.5 * (
            mean_velocity_covariance + mean_velocity_covariance.T
        )
        velocity_covariance_tensor_covariance = 0.5 * (
            velocity_covariance_tensor_covariance
            + velocity_covariance_tensor_covariance.T
        )
    return {
        "status": "identified_from_multi_view_measurements" if identifiable else "conditional_regularized_3d_candidate",
        "evdf_velocity_grid_m_s": velocity.copy(),
        "evdf_m3_s3": distribution if identifiable else None,
        "regularized_candidate_evdf_m3_s3": distribution,
        "unnormalized_nonnegative_coefficients": raw.copy(),
        "normalization": float(volume @ distribution),
        "fitted_spectra": fitted,
        "residual_spectra": residual,
        "distribution_covariance": distribution_covariance if identifiable else None,
        "unnormalized_coefficient_covariance": coefficient_covariance if identifiable else None,
        "effective_spectral_covariance": total_covariance,
        "covariance_components_included": {
            "measurement": "measurement" in covariance_components,
            "model_discrepancy": "model_discrepancy_covariance" in covariance_components,
            "systematic": "systematic_covariance" in covariance_components,
        },
        "coefficient_covariance_method": "fixed_active_set_frequentist_sandwich_Hinv_AwT_Aw_Hinv",
        "mean_velocity_vector_m_s": candidate_mean_velocity.copy() if identifiable else None,
        "velocity_covariance_tensor_m2_s2": candidate_velocity_covariance.copy() if identifiable else None,
        "mean_velocity_covariance_m2_s2": mean_velocity_covariance,
        "velocity_covariance_tensor_covariance_m4_s4": velocity_covariance_tensor_covariance,
        "candidate_mean_velocity_vector_m_s": candidate_mean_velocity.copy(),
        "candidate_velocity_covariance_tensor_m2_s2": candidate_velocity_covariance.copy(),
        "velocity_moment_definition": (
            "mean_v=integral(f*v*d3v); covariance_v=integral(f*(v-mean_v)*(v-mean_v)^T*d3v); "
            "normalization and moments use the supplied voxel-volume quadrature"
        ),
        "velocity_moment_status": (
            "identified_from_multi_view_measurements" if identifiable
            else "conditional_candidate_only_velocity_moments_not_identified"
        ),
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




def build_relativistic_3d_thomson_kernels(
    wavelength_nm,
    velocity_grid_m_s,
    *,
    laser_wavelength_nm,
    incident_direction,
    scattered_direction,
    view_gain,
    instrument_response_matrix,
    include_compton_recoil=False,
):
    r"""Build exact per-view 3-D relativistic single-electron Thomson/Compton kernels.

    For each velocity voxel, transform both photon four-vectors into the
    electron rest frame. The rest-frame scattered photon obeys
    E_s'/E_i' = [1 + (E_i'/m_e c^2)(1-cos(chi'))]^-1 when recoil is enabled.
    The outgoing photon is Lorentz transformed back to the laboratory frame.
    The unpolarized Klein-Nishina differential cross-section is transformed
    with the incident-flux and outgoing-solid-angle Jacobians. In the
    zero-recoil limit the wavelength map reduces exactly to
    lambda_s/lambda_i=(1-beta.n_i)/(1-beta.n_s).

    Returned kernels are relative spectral-power kernels (cross-section
    normalized by r_e^2, multiplied by E_s/E_i), before the caller-supplied
    velocity-space voxel volumes. They are suitable for the rank-gated
    non-negative 3-D inverse; they do not infer absolute density or unknown
    view gains. Polarization-resolved scattering is not included.
    """
    wavelength = _real_array(wavelength_nm, "wavelength_nm", 1)
    velocity = _real_array(velocity_grid_m_s, "velocity_grid_m_s", 2)
    incident = _real_array(incident_direction, "incident_direction", 2)
    scattered = _real_array(scattered_direction, "scattered_direction", 2)
    gains = _real_array(view_gain, "view_gain", 1)
    if wavelength.size < 3 or np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must contain at least three strictly increasing positive bin centres")
    if velocity.shape[1:] != (3,):
        raise ValueError("velocity_grid_m_s must have shape (n_voxel,3)")
    if incident.shape != scattered.shape or incident.ndim != 2 or incident.shape[1] != 3:
        raise ValueError("incident_direction and scattered_direction must both have shape (n_view,3)")
    n_view = incident.shape[0]
    if gains.shape != (n_view,) or np.any(gains <= 0.0):
        raise ValueError("view_gain must contain one finite positive relative gain per view")
    for name, directions in (("incident_direction", incident), ("scattered_direction", scattered)):
        norms = np.linalg.norm(directions, axis=1)
        if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError(f"{name} vectors must already be unit length; geometry is not silently renormalized")
    if isinstance(include_compton_recoil, (bool, np.bool_)):
        recoil = bool(include_compton_recoil)
    else:
        raise ValueError("include_compton_recoil must be an explicit boolean")
    raw_laser = np.asarray(laser_wavelength_nm)
    if raw_laser.dtype.kind not in "iuf":
        raise ValueError("laser_wavelength_nm must contain real numeric values")
    laser = raw_laser.astype(float, copy=False)
    if laser.ndim == 0:
        laser = np.full(n_view, float(laser))
    if laser.shape != (n_view,) or not np.all(np.isfinite(laser)) or np.any(laser <= 0.0):
        raise ValueError("laser_wavelength_nm must be one positive scalar or one positive value per view")
    speed2 = np.einsum("ij,ij->i", velocity, velocity)
    beta = velocity / 299792458.0
    beta2 = speed2 / 299792458.0**2
    if np.any(beta2 >= 1.0):
        raise ValueError("every relativistic 3-D EVDF velocity voxel must be strictly subluminal (|v| < c)")
    gamma = 1.0 / np.sqrt(1.0 - beta2)
    safe_beta2 = np.where(beta2 > 0.0, beta2, 1.0)

    def aberrate_to_rest(direction, projection):
        coefficient = ((gamma - 1.0) * projection / safe_beta2) - gamma
        transformed = (
            direction[None, :] + coefficient[:, None] * beta
        ) / (gamma * (1.0 - projection))[:, None]
        return transformed

    response = np.asarray(instrument_response_matrix)
    if response.dtype.kind not in "iuf":
        raise ValueError("instrument_response_matrix must contain real numeric values")
    response = response.astype(float, copy=False)
    if response.ndim == 2:
        response = np.broadcast_to(response, (n_view, wavelength.size, wavelength.size))
    if (response.shape != (n_view, wavelength.size, wavelength.size)
            or not np.all(np.isfinite(response)) or np.any(response < 0.0)):
        raise ValueError("instrument_response_matrix must be a finite non-negative (n_wave,n_wave) or (n_view,n_wave,n_wave) array")
    response_column_sums = response.sum(axis=1)
    if (np.any(response_column_sums <= 0.0)
            or not np.allclose(response_column_sums, 1.0, rtol=1e-8, atol=1e-10)):
        raise ValueError("each measured response matrix must have unit-sum columns to conserve integrated spectral power")

    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength_nm must define positive finite-volume bins")

    kernels = np.zeros((n_view, wavelength.size, velocity.shape[0]), dtype=float)
    line_centres = np.empty((n_view, velocity.shape[0]), dtype=float)
    cross_section_relative = np.empty_like(line_centres)
    photon_energy_ratio = np.empty_like(line_centres)
    rest_frame_scattering_cosine = np.empty_like(line_centres)
    recoil_fraction = np.zeros_like(line_centres)
    compton_wavelength_shift_nm = np.zeros(n_view, dtype=float)
    h_planck = 6.62607015e-34
    electron_mass = 9.1093837015e-31
    light_speed = 299792458.0
    electron_rest_energy = electron_mass * light_speed**2
    classical_electron_radius = 2.8179403262e-15

    for view in range(n_view):
        ni = incident[view]
        ns = scattered[view]
        dot_i = beta @ ni
        dot_s = beta @ ns
        denominator_i = 1.0 - dot_i
        denominator_s = 1.0 - dot_s
        ni_rest = aberrate_to_rest(ni, dot_i)
        ns_rest = aberrate_to_rest(ns, dot_s)
        cos_chi = np.clip(np.einsum("ij,ij->i", ni_rest, ns_rest), -1.0, 1.0)
        incident_energy = h_planck * light_speed / (laser[view] * 1.0e-9)
        incident_energy_rest = gamma * denominator_i * incident_energy
        epsilon_rest = incident_energy_rest / electron_rest_energy
        if recoil:
            rest_energy_ratio = 1.0 / (1.0 + epsilon_rest * (1.0 - cos_chi))
        else:
            rest_energy_ratio = np.ones_like(cos_chi)
        scattered_energy_ratio = (
            gamma**2 * denominator_i * rest_energy_ratio
            * (1.0 + np.einsum("ij,ij->i", beta, ns_rest))
        )
        if np.any(scattered_energy_ratio <= 0.0) or not np.all(np.isfinite(scattered_energy_ratio)):
            raise ValueError("relativistic photon energy transformation produced a non-positive or non-finite result")
        centres = laser[view] / scattered_energy_ratio
        if np.any(centres < edges[0]) or np.any(centres > edges[-1]):
            raise ValueError(
                f"relativistic velocity grid maps outside the measured wavelength window for view {view}; expand wavelength_nm or narrow velocity_grid_m_s"
            )
        solid_angle_jacobian = denominator_i / (
            gamma**2 * denominator_s**2
        )
        sine2_chi = np.maximum(0.0, 1.0 - cos_chi**2)
        differential_kn_relative = 0.5 * rest_energy_ratio**2 * (
            rest_energy_ratio + 1.0 / rest_energy_ratio - sine2_chi
        )
        differential_lab_relative = (
            solid_angle_jacobian * differential_kn_relative
        )
        # Spectral power, rather than photon number, carries the outgoing
        # photon energy. This factor is essential for recoil-consistent power.
        power_weight = differential_lab_relative * scattered_energy_ratio
        if (np.any(power_weight < 0.0) or not np.all(np.isfinite(power_weight))):
            raise ValueError("relativistic Thomson/Compton kernel weights must be finite and non-negative")

        latent_mass = np.zeros((wavelength.size, velocity.shape[0]), dtype=float)
        for voxel, centre in enumerate(centres):
            if centre <= wavelength[0]:
                latent_mass[0, voxel] = 1.0
            elif centre >= wavelength[-1]:
                latent_mass[-1, voxel] = 1.0
            else:
                right = int(np.searchsorted(wavelength, centre, side="right"))
                left = right - 1
                fraction = (centre - wavelength[left]) / (wavelength[right] - wavelength[left])
                latent_mass[left, voxel] = 1.0 - fraction
                latent_mass[right, voxel] = fraction
        measured_mass = response[view] @ (
            latent_mass * (power_weight * gains[view])[None, :]
        )
        kernels[view] = measured_mass / widths[:, None]
        line_centres[view] = centres
        cross_section_relative[view] = differential_lab_relative
        photon_energy_ratio[view] = scattered_energy_ratio
        rest_frame_scattering_cosine[view] = cos_chi
        recoil_fraction[view] = 1.0 - rest_energy_ratio
        compton_wavelength_shift_nm[view] = (
            h_planck / (electron_mass * light_speed) * (1.0 - np.dot(ni, ns)) * 1.0e9
        )

    return {
        "spectral_kernels": kernels,
        "line_centres_nm": line_centres,
        "relative_lab_differential_cross_section": cross_section_relative,
        "scattered_photon_energy_over_incident": photon_energy_ratio,
        "rest_frame_scattering_cosine": rest_frame_scattering_cosine,
        "rest_frame_compton_recoil_fraction": recoil_fraction,
        "compton_wavelength_shift_at_rest_nm": compton_wavelength_shift_nm,
        "scattering_angle_rad": np.arccos(np.clip(np.einsum("ij,ij->i", incident, scattered), -1.0, 1.0)),
        "wavelength_bin_widths_nm": widths,
        "instrument_response_per_view": response.copy(),
        "maximum_velocity_beta": float(np.max(np.sqrt(beta2))),
        "include_compton_recoil": recoil,
        "cross_section_model": "unpolarized_Klein_Nishina" if recoil else "unpolarized_relativistic_Thomson",
        "spectral_weight_semantics": "relative_differential_cross_section_times_outgoing_to_incident_photon_energy_ratio",
    }


def reconstruct_relativistic_optical_thomson_3d_evdf(
    wavelength_nm,
    measured_spectra,
    incident_direction,
    scattered_direction,
    velocity_grid_m_s,
    *,
    laser_wavelength_nm,
    view_gain,
    instrument_response_matrix,
    background_spectra,
    voxel_volume_m3_s3,
    regularization_matrix,
    regularization_strength,
    include_compton_recoil=False,
    measurement_uncertainty=None,
    measurement_covariance=None,
    model_discrepancy_covariance=None,
    systematic_covariance=None,
    max_iter=3000,
):
    r"""Reconstruct a non-negative general 3-D EVDF with relativistic optical kernels.

    Unlike a gyrotropic closure or a q-parallel one-dimensional projection, the
    input velocity grid spans all three Cartesian components. Each view has its
    own incident and scattered photon directions. The forward kernel uses exact
    Lorentz aberration and relativistic Doppler kinematics; optional recoil uses
    the Compton four-momentum relation and unpolarized Klein-Nishina weighting.
    A 3-D distribution is returned only if the stacked, whitened data-only
    Jacobian identifies every velocity voxel. Regularization cannot identify
    missing angular modes. Absolute density and polarization-resolved response
    are not inferred.
    """
    wavelength = _real_array(wavelength_nm, "wavelength_nm", 1)
    spectra = _real_array(measured_spectra, "measured_spectra", 2)
    background = _real_array(background_spectra, "background_spectra", 2)
    if spectra.shape != background.shape or spectra.shape[1] != wavelength.size:
        raise ValueError("measured_spectra and background_spectra must match (n_view,n_wavelength)")
    if wavelength.size < 3 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must contain at least three strictly increasing calibrated bin centres")
    kernel_data = build_relativistic_3d_thomson_kernels(
        wavelength, velocity_grid_m_s,
        laser_wavelength_nm=laser_wavelength_nm,
        incident_direction=incident_direction,
        scattered_direction=scattered_direction,
        view_gain=view_gain,
        instrument_response_matrix=instrument_response_matrix,
        include_compton_recoil=include_compton_recoil,
    )
    if spectra.shape[0] != kernel_data["spectral_kernels"].shape[0]:
        raise ValueError("measured_spectra view count must match the supplied incident/scattered geometry")
    result = reconstruct_multiview_3d_evdf(
        kernel_data["spectral_kernels"],
        spectra - background,
        velocity_grid_m_s,
        voxel_volume_m3_s3=voxel_volume_m3_s3,
        regularization_matrix=regularization_matrix,
        measurement_uncertainty=measurement_uncertainty,
        measurement_covariance=measurement_covariance,
        model_discrepancy_covariance=model_discrepancy_covariance,
        systematic_covariance=systematic_covariance,
        regularization_strength=regularization_strength,
        max_iter=max_iter,
    )
    result = dict(result)
    result.update({
        "physical_forward_model": "general_3d_relativistic_unpolarized_thomson_with_optional_compton_recoil",
        "spectral_forward_equation": "S_j(lambda_i)=sum_a f(v_a) K_j(lambda_i|v_a) DeltaV_a; lambda_s=lambda_L/[gamma^2(1-beta.n_i) r (1+beta.n_s')]; r=[1+epsilon_i'*(1-cos(chi'))]^-1",
        "incident_direction": np.asarray(incident_direction, dtype=float),
        "scattered_direction": np.asarray(scattered_direction, dtype=float),
        "laser_wavelength_nm": np.asarray(laser_wavelength_nm, dtype=float),
        "view_gain": np.asarray(view_gain, dtype=float),
        "background_subtracted_spectra": spectra - background,
        "physical_spectral_kernels": kernel_data["spectral_kernels"],
        "forward_kernel_contract": "internally generated per-view relativistic 3-D optical Thomson/Compton kernels; exact Lorentz aberration and Doppler kinematics; optional rest-frame recoil and unpolarized Klein-Nishina weighting; measured response applied to finite-bin integrated spectral power",
        "relativistic_kinematics": kernel_data,
        "validity_domain": {
            "velocity_grid_subluminal": True,
            "maximum_velocity_beta": kernel_data["maximum_velocity_beta"],
            "relativistic_doppler_kinematics_included": True,
            "compton_recoil_included": bool(include_compton_recoil),
            "cross_section": kernel_data["cross_section_model"],
            "polarization_resolved": False,
            "absolute_density_inferred": False,
        },
        "identifiability_scope": (
            "general_3D_velocity_space_only_if_stacked_data_only_kernel_has_full_column_rank"
        ),
        "experimental_validation": "synthetic_physics_recovery_required_before_experimental_claim",
    })
    return result

def reconstruct_optical_thomson_3d_evdf(
    wavelength_nm,
    measured_spectra,
    scattering_vector_unit,
    velocity_grid_m_s,
    *,
    laser_wavelength_nm,
    scattering_angle_rad,
    view_gain,
    instrument_response_matrix,
    background_spectra,
    voxel_volume_m3_s3,
    regularization_matrix,
    regularization_strength,
    measurement_uncertainty=None,
    measurement_covariance=None,
    model_discrepancy_covariance=None,
    systematic_covariance=None,
    max_iter=3000,
):
    r"""Build and invert the physical non-relativistic optical-Thomson 3-D kernel.

    For view j and velocity voxel a, the elastic single-particle Doppler map is
    lambda_ja = lambda_L [1 - 2 sin(theta/2) (qhat_j . v_a)/c].
    A unit-integral monochromatic line is deposited conservatively into the
    finite wavelength bins, then the measured response operator acts on bin
    *integrated* power. The inverse solves S_j(lambda)=int f(v)K_j(lambda|v)d^3v.

    This route is explicitly restricted to unpolarized, non-relativistic,
    single-scattering Thomson shape inference. It requires a measured response
    per view and explicit background; it does not infer absolute density,
    unknown view gains, polarization, Compton recoil or a relativistic EVDF.
    The 3-D distribution is reported only when the stacked data-only kernel
    identifies every velocity voxel; regularization cannot supply angular data.
    """
    wavelength = _real_array(wavelength_nm, "wavelength_nm", 1)
    spectra = _real_array(measured_spectra, "measured_spectra", 2)
    directions = _real_array(scattering_vector_unit, "scattering_vector_unit", 2)
    velocity = _real_array(velocity_grid_m_s, "velocity_grid_m_s", 2)
    background = _real_array(background_spectra, "background_spectra", 2)
    if wavelength.size < 3 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must contain at least three strictly increasing calibrated bin centres")
    n_view, n_wave = spectra.shape
    if n_wave != wavelength.size or background.shape != spectra.shape:
        raise ValueError("measured_spectra and background_spectra must match (n_view, n_wavelength)")
    if directions.shape != (n_view, 3) or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError("scattering_vector_unit must be (n_view,3) and velocity_grid_m_s must be (n_voxel,3)")
    direction_norm = np.linalg.norm(directions, axis=1)
    if np.any(direction_norm <= 0.0) or not np.allclose(
        direction_norm, 1.0, rtol=1e-8, atol=1e-10
    ):
        raise ValueError("every scattering_vector_unit must already be a unit vector; geometry is not silently renormalized")
    if not np.isfinite(laser_wavelength_nm) or laser_wavelength_nm <= 0.0:
        raise ValueError("laser_wavelength_nm must be finite and positive")
    angles = np.asarray(scattering_angle_rad, dtype=float)
    if angles.ndim == 0:
        angles = np.full(n_view, float(angles))
    if (angles.shape != (n_view,) or not np.all(np.isfinite(angles))
            or np.any(angles <= 0.0) or np.any(angles >= np.pi)):
        raise ValueError("scattering_angle_rad must be one finite angle in (0, pi), or one angle per measured view")
    gains = _real_array(view_gain, "view_gain", 1)
    if gains.shape != (n_view,) or np.any(gains <= 0.0):
        raise ValueError("view_gain must contain one finite positive calibrated relative gain per view")
    from .formula_library import (
        thomson_nonrelativistic_doppler_wavelength,
        thomson_unpolarized_differential_cross_section,
    )
    classical_electron_radius_m = 2.8179403262e-15
    angular_factor = (
        thomson_unpolarized_differential_cross_section(angles)
        / classical_electron_radius_m**2
    )
    speed = np.linalg.norm(velocity, axis=1)
    beta_max = float(np.max(speed) / 299792458.0)
    if beta_max >= 0.1:
        raise ValueError(
            "the physical 3-D optical Thomson kernel is non-relativistic and requires max(|v|)/c < 0.1; use the relativistic 3-D kernel when available"
        )
    response = np.asarray(instrument_response_matrix, dtype=float)
    if response.ndim == 2:
        response = np.broadcast_to(response, (n_view, n_wave, n_wave))
    if (response.shape != (n_view, n_wave, n_wave)
            or not np.all(np.isfinite(response)) or np.any(response < 0.0)):
        raise ValueError("instrument_response_matrix must be non-negative finite (n_wave,n_wave) or (n_view,n_wave,n_wave)")
    # response has shape (view, output_wavelength, input_wavelength).
    # Column stochasticity is checked over output wavelength independently
    # for each view; summing over views could hide a non-conserving operator.
    response_column_sums = response.sum(axis=1)
    if (np.any(response_column_sums <= 0.0)
            or not np.allclose(response_column_sums, 1.0, rtol=1e-8, atol=1e-10)):
        raise ValueError(
            "each measured response matrix must independently conserve integrated spectral power: "
            "every view's columns must sum to one"
        )
    edges = np.empty(n_wave + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("calibrated wavelength centres must define positive finite-volume bins")
    q_over_ki = 2.0 * np.sin(angles / 2.0)
    kernels = np.zeros((n_view, n_wave, velocity.shape[0]), dtype=float)
    line_centres = np.empty((n_view, velocity.shape[0]), dtype=float)
    for view, qhat in enumerate(directions):
        projected_velocity = velocity @ qhat
        centres = thomson_nonrelativistic_doppler_wavelength(
            float(laser_wavelength_nm), float(angles[view]), projected_velocity
        )
        line_centres[view] = centres
        if np.any(centres < edges[0]) or np.any(centres > edges[-1]):
            raise ValueError(
                f"velocity grid for view {view} maps outside the measured wavelength window; expand the wavelength range or narrow the velocity grid"
            )
        latent_mass = np.zeros((n_wave, velocity.shape[0]), dtype=float)
        for voxel, centre in enumerate(centres):
            if centre <= wavelength[0]:
                latent_mass[0, voxel] = 1.0
            elif centre >= wavelength[-1]:
                latent_mass[-1, voxel] = 1.0
            else:
                right = int(np.searchsorted(wavelength, centre, side="right"))
                left = right - 1
                fraction = (centre - wavelength[left]) / (wavelength[right] - wavelength[left])
                latent_mass[left, voxel] = 1.0 - fraction
                latent_mass[right, voxel] = fraction
        # R acts on bin-integrated power; divide by bin widths only afterwards.
        measured_mass = response[view] @ latent_mass
        kernels[view] = measured_mass / widths[:, None] * angular_factor[view] * gains[view]
    # Preserve signed background-subtracted residuals. Gaussian/GLS
    # measurement models permit negative noise excursions; non-negativity is
    # imposed on the latent EVDF coefficients, not on each observed bin.
    background_subtracted = spectra - background
    result = reconstruct_multiview_3d_evdf(
        kernels, background_subtracted, velocity,
        voxel_volume_m3_s3=voxel_volume_m3_s3,
        regularization_matrix=regularization_matrix,
        measurement_uncertainty=measurement_uncertainty,
        measurement_covariance=measurement_covariance,
        model_discrepancy_covariance=model_discrepancy_covariance,
        systematic_covariance=systematic_covariance,
        regularization_strength=regularization_strength,
        max_iter=max_iter,
    )
    result = dict(result)
    result.update({
        "physical_forward_model": "elastic_single_particle_unpolarized_nonrelativistic_optical_thomson",
        "spectral_forward_equation": "S_j(lambda_i)=sum_a f_a K_ji,a DeltaV_a; lambda_ja=lambda_L*(1-2*sin(theta/2)*(qhat_j dot v_a)/c)",
        "scattering_vector_unit": directions.copy(),
        "laser_wavelength_nm": float(laser_wavelength_nm),
        "scattering_angle_rad": angles.copy(),
        "view_gain": gains.copy(),
        "unpolarized_thomson_angular_factor": angular_factor.copy(),
        "unpolarized_differential_cross_section_m2_sr": (
            thomson_unpolarized_differential_cross_section(angles)
        ),
        "q_over_k_incident_per_view": q_over_ki.copy(),
        "mapped_line_centres_nm": line_centres,
        "wavelength_bin_widths_nm": widths,
        "background_subtracted_spectra": background_subtracted,
        "physical_spectral_kernels": kernels,
        "instrument_response_per_view": response.copy(),
        "maximum_velocity_beta": beta_max,
        "validity_domain": {
            "maximum_speed_over_c": 0.1,
            "actual_maximum_speed_over_c": beta_max,
            "relativistic_corrections_included": False,
            "compton_recoil_included": False,
            "polarization_resolved": False,
            "absolute_density_inferred": False,
        },
        "forward_kernel_contract": "internally generated physical Doppler/finite-volume/response kernel; per-view measured response applied to integrated bin power",
        "experimental_validation": "synthetic_physics_recovery_required_before_experimental_claim",
    })
    return result

def assess_multiview_evdf_regularization_sensitivity(
    spectral_kernels, measured_spectra, velocity_grid_m_s, *,
    voxel_volume_m3_s3, regularization_matrix, regularization_strengths,
    measurement_uncertainty=None, measurement_covariance=None,
    model_discrepancy_covariance=None, systematic_covariance=None,
    max_iter=3000,
):
    """Evaluate a non-negative multi-view EVDF inverse over an explicit lambda grid.

    No optimum regularization strength is selected automatically. The returned
    L-curve quantities, data-only rank/condition diagnostics and candidate
    distributions let the researcher assess the bias/residual trade-off without
    presenting regularization as additional measurement information.
    """
    strengths = _real_array(
        regularization_strengths, "regularization_strengths", 1
    )
    if strengths.size < 3 or np.any(strengths <= 0.0) or np.any(np.diff(strengths) <= 0.0):
        raise ValueError(
            "regularization_strengths must contain at least three positive, strictly increasing values"
        )
    kernels = _real_array(spectral_kernels, "spectral_kernels", 3)
    data = _real_array(measured_spectra, "measured_spectra", 2)
    penalty = _real_array(regularization_matrix, "regularization_matrix", 2)
    if data.shape != kernels.shape[:2] or penalty.shape[1] != kernels.shape[2]:
        raise ValueError("measured spectra and regularization matrix must match the spectral kernel dimensions")
    if measurement_uncertainty is not None and measurement_covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    covariance_components = {}
    if measurement_uncertainty is not None:
        sigma = _real_array(measurement_uncertainty, "measurement_uncertainty", 2)
        if sigma.shape != data.shape or np.any(sigma <= 0.0):
            raise ValueError("measurement_uncertainty must be positive and match measured_spectra")
        covariance_components["measurement"] = np.diag(sigma.ravel() ** 2)
    elif measurement_covariance is not None:
        covariance_components["measurement"] = _real_array(
            measurement_covariance, "measurement_covariance", 2
        )
    for name, value in (
        ("model_discrepancy", model_discrepancy_covariance),
        ("systematic", systematic_covariance),
    ):
        if value is not None:
            covariance_components[name] = _real_array(value, name + "_covariance", 2)
    for name, covariance_component in covariance_components.items():
        if covariance_component.shape != (data.size, data.size):
            raise ValueError(
                f"{name} covariance must match the stacked (view-major) multi-view spectra"
            )
        scale = max(float(np.max(np.abs(covariance_component))), np.finfo(float).tiny)
        if not np.allclose(
            covariance_component, covariance_component.T,
            rtol=1e-10, atol=1e-12 * scale,
        ):
            raise ValueError(f"{name} covariance must be symmetric")
        covariance_component = 0.5 * (
            covariance_component + covariance_component.T
        )
        eigenvalues = np.linalg.eigvalsh(covariance_component)
        if eigenvalues[0] < -1e-10 * max(
            float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny
        ):
            raise ValueError(f"{name} covariance must be positive semidefinite")
        covariance_components[name] = covariance_component
    if covariance_components:
        total_covariance = np.sum(list(covariance_components.values()), axis=0)
        try:
            cholesky = np.linalg.cholesky(total_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "combined measurement, model-discrepancy and systematic covariance "
                "must be positive definite"
            ) from exc
        whiten = lambda value: np.linalg.solve(
            cholesky, np.asarray(value, dtype=float).ravel()
        ).reshape(data.shape)
        noise_model = (
            "measurement_plus_model_discrepancy_plus_systematic_covariance"
        )
    else:
        whiten = lambda value: np.asarray(value, dtype=float)
        noise_model = "unweighted_no_measurement_uncertainty_supplied"

    levels = []
    candidates = []
    for strength in strengths:
        result = reconstruct_multiview_3d_evdf(
            kernels, data, velocity_grid_m_s,
            voxel_volume_m3_s3=voxel_volume_m3_s3,
            regularization_matrix=penalty,
            measurement_uncertainty=measurement_uncertainty,
            measurement_covariance=measurement_covariance,
            model_discrepancy_covariance=model_discrepancy_covariance,
            systematic_covariance=systematic_covariance,
            regularization_strength=float(strength),
            max_iter=max_iter,
        )
        residual = np.asarray(result["residual_spectra"], dtype=float)
        candidate = np.asarray(result["regularized_candidate_evdf_m3_s3"], dtype=float)
        coefficients = np.asarray(
            result["unnormalized_nonnegative_coefficients"], dtype=float
        )
        whitened_residual = whiten(residual)
        data_residual_norm = float(np.linalg.norm(whitened_residual))
        # Match the exact coefficient vector to which the inverse solver applies
        # sqrt(lambda) * L; the normalized PDF is a separate reported quantity.
        penalty_norm = float(np.linalg.norm(penalty @ coefficients))
        levels.append({
            "regularization_strength": float(strength),
            "weighted_data_residual_norm": data_residual_norm,
            "regularization_penalty_norm": penalty_norm,
            "data_only_jacobian_rank": int(result["data_only_jacobian_rank"]),
            "data_only_condition_number": float(result["data_only_condition_number"]),
            "null_space_dimension": int(result["null_space_dimension"]),
            "identifiability_status": result["status"],
        })
        candidates.append(candidate)
    candidate_stack = np.asarray(candidates, dtype=float)
    if candidate_stack.ndim != 2 or candidate_stack.shape[0] != strengths.size:
        raise RuntimeError("regularization ensemble did not produce one normalized candidate per strength")
    # The lambda ensemble is a sensitivity ensemble, not a sampling distribution.
    # Propagate its spread through the same voxel-volume definitions used by the
    # physical inverse, keeping this component separate from measurement covariance.
    distribution_spread = (
        np.std(candidate_stack, axis=0, ddof=1) if strengths.size > 1
        else np.zeros(candidate_stack.shape[1], dtype=float)
    )
    distribution_covariance = (
        np.cov(candidate_stack, rowvar=False, ddof=1) if strengths.size > 1
        else np.zeros((candidate_stack.shape[1], candidate_stack.shape[1]), dtype=float)
    )
    distribution_covariance = np.atleast_2d(distribution_covariance)
    volume = _real_array(voxel_volume_m3_s3, "voxel_volume_m3_s3", 1)
    velocity = _real_array(velocity_grid_m_s, "velocity_grid_m_s", 2)
    moment_vectors = []
    for pdf in candidate_stack:
        probability_mass = pdf * volume
        mean_velocity = probability_mass @ velocity
        centered = velocity - mean_velocity[None, :]
        velocity_tensor = np.einsum(
            "i,ij,ik->jk", probability_mass, centered, centered
        )
        moment_vectors.append(np.concatenate((mean_velocity, velocity_tensor.reshape(9))))
    moment_stack = np.asarray(moment_vectors, dtype=float)
    moment_covariance = (
        np.cov(moment_stack, rowvar=False, ddof=1) if strengths.size > 1
        else np.zeros((12, 12), dtype=float)
    )
    moment_covariance = np.atleast_2d(moment_covariance)
    return {
        "levels": levels,
        "regularization_strengths": strengths.copy(),
        "candidate_distributions_m3_s3": candidate_stack,
        "regularization_selection_uncertainty": {
            "status": "candidate_spread_available" if strengths.size > 1 else "single_candidate_no_selection_spread",
            "candidate_count": int(strengths.size),
            "distribution_standard_spread_m3_s3": distribution_spread,
            "distribution_covariance_m6_s6": distribution_covariance,
            "distribution_lower_envelope_m3_s3": np.min(candidate_stack, axis=0),
            "distribution_upper_envelope_m3_s3": np.max(candidate_stack, axis=0),
            "physical_moment_order": [
                "mean_vx_m_s", "mean_vy_m_s", "mean_vz_m_s",
                "cov_vxx_m2_s2", "cov_vxy_m2_s2", "cov_vxz_m2_s2",
                "cov_vyx_m2_s2", "cov_vyy_m2_s2", "cov_vyz_m2_s2",
                "cov_vzx_m2_s2", "cov_vzy_m2_s2", "cov_vzz_m2_s2",
            ],
            "physical_moment_candidate_values": moment_stack,
            "physical_moment_covariance_moment_units_squared": moment_covariance,
            "interpretation": (
                "Empirical spread across the explicitly supplied regularization strengths, "
                "propagated through the normalized 3-D EVDF and its first/second velocity moments. "
                "This is model/regularization-selection sensitivity, not measurement uncertainty, "
                "a confidence interval, or a posterior covariance."
            ),
        },
        "noise_model": noise_model,
        "covariance_components_included": {
            "measurement": "measurement" in covariance_components,
            "model_discrepancy": "model_discrepancy" in covariance_components,
            "systematic": "systematic" in covariance_components,
        },
        "effective_spectral_covariance": (
            np.sum(list(covariance_components.values()), axis=0)
            if covariance_components else None
        ),
        "selection_policy": "no automatic lambda selection; inspect the L-curve and external physical constraints",
        "interpretation": (
            "Sensitivity of the regularized solution to the supplied lambda grid. "
            "The data-only rank and null space are invariant to regularization; "
            "regularization does not create measurement identifiability."
        ),
        "experimental_validation": "not_performed",
    }



def fit_nonlinear_nonnegative_distribution(
    measured_spectrum, initial_distribution, quadrature_weights, *,
    forward_model, regularization_matrix, regularization_strength,
    measurement_uncertainty=None, measurement_covariance=None,
    max_nfev=2000,
):
    """Fit a normalized non-negative distribution through a nonlinear forward model.

    The forward_model(pdf) callback must evaluate the complete physical
    spectrum for the supplied distribution. A softmax mass parameterization
    enforces non-negativity and exact normalization. The regularizer is explicit
    and acts on the physical PDF, not on an arbitrary optimizer parameter.
    """
    from scipy.optimize import least_squares
    y = _real_array(measured_spectrum, "measured_spectrum", 1)
    pdf0 = _real_array(initial_distribution, "initial_distribution", 1)
    weights = _real_array(quadrature_weights, "quadrature_weights", 1)
    penalty = _real_array(regularization_matrix, "regularization_matrix", 2)
    if pdf0.shape != weights.shape or penalty.shape[1] != pdf0.size:
        raise ValueError("initial distribution, quadrature weights and regularization matrix dimensions must agree")
    if np.any(pdf0 < 0) or np.any(weights <= 0) or float(pdf0 @ weights) <= 0:
        raise ValueError("initial distribution must be non-negative with positive weighted integral; quadrature weights must be positive")
    if not callable(forward_model):
        raise ValueError("forward_model must be callable")
    lam = float(regularization_strength)
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError("regularization_strength must be positive and finite")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 10:
        raise ValueError("max_nfev must be an integer >=10")
    if measurement_uncertainty is not None and measurement_covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    if measurement_uncertainty is not None:
        sigma = _real_array(measurement_uncertainty, "measurement_uncertainty", 1)
        if sigma.shape != y.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match measured_spectrum")
        whiten = lambda z: np.asarray(z) / sigma
        noise_status = "supplied_pointwise_measurement_uncertainty"
    elif measurement_covariance is not None:
        covariance = _real_array(measurement_covariance, "measurement_covariance", 2)
        if covariance.shape != (y.size, y.size) or not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-13):
            raise ValueError("measurement_covariance must be symmetric and match measured_spectrum")
        chol = np.linalg.cholesky(covariance)
        whiten = lambda z: np.linalg.solve(chol, np.asarray(z))
        noise_status = "supplied_full_measurement_covariance"
    else:
        whiten = lambda z: np.asarray(z)
        noise_status = "unweighted_no_measurement_uncertainty_supplied"
    masses0 = np.maximum(pdf0 * weights, np.finfo(float).tiny)
    masses0 /= masses0.sum()
    logits0 = np.log(masses0[:-1] / masses0[-1])
    def decode(logits):
        values = np.r_[np.asarray(logits, dtype=float), 0.0]
        values -= np.max(values)
        masses = np.exp(values)
        masses /= masses.sum()
        return masses / weights
    def profiled_spectrum(pdf):
        spectrum = _real_array(forward_model(pdf), "forward_model spectrum", 1)
        if spectrum.shape != y.shape or np.any(spectrum < 0):
            raise ValueError("forward_model must return a non-negative spectrum matching measured_spectrum")
        design = whiten(np.column_stack((spectrum, np.ones_like(spectrum))))
        target = whiten(y)
        amplitude, offset = np.linalg.lstsq(design, target, rcond=None)[0]
        if amplitude < 0:
            amplitude = 0.0
            one = whiten(np.ones_like(y))
            offset = float(one @ target / (one @ one))
        residual = whiten(amplitude * spectrum + offset - y)
        return residual, amplitude, offset, spectrum
    penalty_scale = max(float(np.linalg.norm(penalty, ord="fro")), np.finfo(float).tiny)
    def residual(logits):
        pdf = decode(logits)
        data_residual, _, _, _ = profiled_spectrum(pdf)
        smoothness = np.sqrt(lam) * (penalty @ pdf) / max(float(np.max(pdf)), np.finfo(float).tiny)
        return np.r_[data_residual, smoothness]
    fit = least_squares(residual, logits0, method="trf", x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError(f"nonlinear distribution-shape inversion failed: {fit.message}")
    pdf = decode(fit.x)
    data_residual, amplitude, offset, intrinsic = profiled_spectrum(pdf)
    jac = np.asarray(fit.jac, dtype=float)
    # Identifiability is assessed from data residuals only; regularization rows
    # must never turn a data-null-space mode into an apparently measured mode.
    data_jac = jac[:y.size, :]
    singular = np.linalg.svd(data_jac, compute_uv=False)
    tolerance = np.finfo(float).eps * max(data_jac.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > tolerance else float("inf")
    identifiable = rank == fit.x.size and np.isfinite(condition) and condition < 1e12
    covariance_logits = np.linalg.pinv(jac.T @ jac, hermitian=True)
    if noise_status == "unweighted_no_measurement_uncertainty_supplied":
        covariance_logits *= float(fit.fun @ fit.fun / max(1, y.size - fit.x.size))
    # Exact softmax Jacobian from logits to normalized PDF.
    masses = pdf * weights
    softmax_jac = (np.diag(masses) - np.outer(masses, masses))[:, :-1]
    pdf_jac = softmax_jac / weights[:, None]
    covariance_pdf = pdf_jac @ covariance_logits @ pdf_jac.T
    fitted = amplitude * intrinsic + offset
    return {
        "status": "identified_conditional_distribution_shape" if identifiable else "conditional_regularized_distribution_candidate",
        "distribution": pdf if identifiable else None,
        "regularized_candidate_distribution": pdf,
        "distribution_covariance": covariance_pdf if identifiable else None,
        "fitted_spectrum": fitted,
        "residual": y - fitted,
        "profiled_amplitude": float(amplitude),
        "profiled_offset": float(offset),
        "data_only_or_regularized_jacobian_rank": rank,
        "parameter_count": int(fit.x.size),
        "jacobian_condition_number": condition,
        "regularization_strength": lam,
        "noise_model": noise_status,
        "normalization": float(pdf @ weights),
        "uncertainty_interpretation": "local covariance conditional on the physical forward model, momentum grid, regularization and fixed plasma/geometry inputs",
        "experimental_validation": "not_performed",
    }

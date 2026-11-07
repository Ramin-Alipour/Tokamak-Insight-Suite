"""Non-parametric 1-D projected electron-velocity reconstruction from optical Thomson spectra.

This is an inverse problem on calibrated measured spectra, not a calibration or
measurement-correction routine. A single scattering geometry identifies only
the marginal along the scattering vector, never an unrestricted 3-D EVDF.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid
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
    if not np.allclose(sums, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "measured instrument-response matrix columns must already sum to one; "
            "the calibrated response operator is not renormalized"
        )
    return response


def _wavelength_bin_widths(wavelength):
    """Return midpoint finite-volume bin widths for a calibrated wavelength axis."""
    axis = np.asarray(wavelength, dtype=float)
    if (axis.ndim != 1 or axis.size < 2 or np.any(~np.isfinite(axis))
            or np.any(np.diff(axis) <= 0.0)):
        raise ValueError("wavelength axis must be finite, one-dimensional and strictly increasing")
    edges = np.empty(axis.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength axis must define positive finite-volume bin widths")
    return widths


def _normalize_evdf_coefficients(coefficients, velocity):
    """Normalize non-negative velocity-bin coefficients using trapezoidal weights.

    Returns the normalized density and its exact Jacobian with respect to the
    unnormalized coefficients. Endpoint half-weights are essential because
    the reported density is normalized with trapezoidal integration.
    """
    values = np.asarray(coefficients, dtype=float)
    grid = np.asarray(velocity, dtype=float)
    if (values.ndim != 1 or grid.shape != values.shape or values.size < 2
            or np.any(~np.isfinite(values)) or np.any(values < 0.0)
            or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("coefficients and velocity must be matching finite non-negative/strictly increasing vectors")
    quadrature_weights = np.empty(values.size, dtype=float)
    intervals = np.diff(grid)
    quadrature_weights[0] = 0.5 * intervals[0]
    quadrature_weights[-1] = 0.5 * intervals[-1]
    quadrature_weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    weighted_sum = float(quadrature_weights @ values)
    if not np.isfinite(weighted_sum) or weighted_sum <= np.finfo(float).tiny:
        raise ValueError("EVDF coefficients must have positive trapezoidal integral")
    distribution = values / weighted_sum
    jacobian = (
        np.eye(values.size) * weighted_sum
        - np.outer(values, quadrature_weights)
    ) / weighted_sum**2
    return distribution, jacobian


def _velocity_quadrature_weights(velocity):
    """Trapezoidal finite-volume measure for nodal EVDF density coefficients.

    The inverse coefficients represent f(v) [s/m], not discrete probability
    masses. Therefore each monochromatic Doppler kernel must be weighted by
    its velocity-space quadrature measure when approximating
    S(lambda) = integral f(v) K(lambda | v) dv.
    """
    grid = np.asarray(velocity, dtype=float)
    if (grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid))
            or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("velocity grid must be finite, strictly increasing and contain at least two nodes")
    intervals = np.diff(grid)
    weights = np.empty(grid.size, dtype=float)
    weights[0] = 0.5 * intervals[0]
    weights[-1] = 0.5 * intervals[-1]
    weights[1:-1] = 0.5 * (intervals[:-1] + intervals[1:])
    return weights


def _projected_evdf_forward_matrix(wavelength, velocity, response, laser_wavelength_nm, q_over_ki):
    """Build the finite-volume Doppler/response operator for a nodal 1-D EVDF.

    Each column is a unit-integral monochromatic line mapped to wavelength
    bins, convolved by the measured response, and multiplied by the
    trapezoidal velocity-space quadrature weight. Thus the matrix acts on
    nodal probability densities and approximates the physical velocity
    integral rather than an unweighted sum over velocity nodes.
    """
    x = _finite_vector(wavelength, "wavelength_nm")
    v = _finite_vector(velocity, "velocity_grid_m_s", minimum=2)
    matrix = _response_matrix(response, x.size)
    laser = float(laser_wavelength_nm)
    q_factor = float(q_over_ki)
    if not np.isfinite(laser) or laser <= 0.0 or not np.isfinite(q_factor) or q_factor <= 0.0:
        raise ValueError("laser wavelength and scattering-vector magnitude must be finite and positive")
    centres = laser * (1.0 - q_factor * v / _C)
    if np.any(centres < x[0]) or np.any(centres > x[-1]):
        raise ValueError("velocity grid maps outside the measured wavelength interval; narrow the velocity grid or widen the measured spectral window")
    widths = _wavelength_bin_widths(x)
    latent = np.zeros((x.size, v.size), dtype=float)
    for j, centre in enumerate(centres):
        right = int(np.searchsorted(x, centre, side="right"))
        if right == 0:
            latent[0, j] = 1.0 / widths[0]
        elif right >= x.size:
            latent[-1, j] = 1.0 / widths[-1]
        else:
            left = right - 1
            fraction = (centre - x[left]) / (x[right] - x[left])
            latent[left, j] = (1.0 - fraction) / widths[left]
            latent[right, j] = fraction / widths[right]
    response_projected = matrix @ (latent * widths[:, None])
    response_projected /= widths[:, None]
    velocity_weights = _velocity_quadrature_weights(v)
    forward = response_projected * velocity_weights[None, :]
    return forward, {
        "wavelength_bin_widths_nm": widths,
        "velocity_quadrature_weights_m_s": velocity_weights,
        "mapped_line_centres_nm": centres,
        "column_integrated_spectral_power_per_unit_evdf": (
            np.sum(forward * widths[:, None], axis=0)
        ),
    }


def _nonuniform_second_derivative_matrix(velocity):
    """Return the three-point second-derivative operator on a monotone grid.

    At interior node i this is 2/(h_left+h_right) times the difference
    between the right and left first-derivative slopes. It is exact for
    quadratic polynomials on arbitrary strictly increasing node locations.
    """
    grid = np.asarray(velocity, dtype=float)
    if (grid.ndim != 1 or grid.size < 3 or not np.all(np.isfinite(grid))
            or np.any(np.diff(grid) <= 0.0)):
        raise ValueError("velocity grid must be finite, strictly increasing and contain at least three nodes")
    intervals = np.diff(grid)
    operator = np.zeros((grid.size - 2, grid.size), dtype=float)
    for row, (h_left, h_right) in enumerate(zip(intervals[:-1], intervals[1:])):
        factor = 2.0 / (h_left + h_right)
        operator[row, row] = factor / h_left
        operator[row, row + 1] = -factor * (1.0 / h_left + 1.0 / h_right)
        operator[row, row + 2] = factor / h_right
    return operator


def reconstruct_thomson_evdf(
    wavelength_nm,
    signal,
    *,
    laser_wavelength_nm,
    scattering_angle_rad,
    instrument_response_matrix,
    background_signal,
    laser_spectrum_wavelength_nm=None,
    laser_spectrum_density_per_nm=None,
    regularization_strength=None,
    velocity_grid_m_s=None,
    velocity_limit_m_s=8.0e6,
    velocity_points=241,
    signal_uncertainty=None,
    signal_covariance=None,
    background_uncertainty=None,
    background_covariance=None,
    signal_background_cross_covariance=None,
    model_discrepancy_covariance=None,
    systematic_covariance=None,
    max_iter=3000,
):
    """Reconstruct the non-negative 1-D electron-velocity marginal along q.

    The Doppler mapping uses the non-relativistic single-particle relation
    delta_omega = q dot v, with |q|/|k_i| = 2 sin(theta/2). The latent spectrum
    is convolved with the supplied measured response matrix before fitting.
    The fitted background is an explicit input. Regularization strength is optional: when omitted, the solver selects it by generalized cross-validation (GCV); a finite positive value overrides that selection. The result is
    shape-normalized; absolute density is not inferred by this inverse routine.

    Optional model_discrepancy_covariance and systematic_covariance are additive
    wavelength-space covariance components. Each is assumed independent of the
    measurement/background components and of the other supplied component; no
    unprovided cross-covariance is invented.

    The conditional active-set covariance of the normalized EVDF is also
    propagated by the delta method into the correlated covariance of its first
    through fourth central physical moments, including q-parallel flow, the
    directional equivalent temperature, skewness and excess kurtosis. This
    covariance remains conditional on the selected grid, response, background
    and regularization; it is not a posterior distribution.
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
    if regularization_strength is not None and (
        isinstance(regularization_strength, (bool, np.bool_))
        or not np.isfinite(regularization_strength)
        or regularization_strength <= 0
    ):
        raise ValueError("regularization_strength must be None for automatic GCV selection or a finite positive value")
    if isinstance(velocity_points, (bool, np.bool_)) or not isinstance(velocity_points, (int, np.integer)) or velocity_points < 41 or velocity_points > 2001:
        raise ValueError("velocity_points must be an integer from 41 to 2001")
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter < 100:
        raise ValueError("max_iter must be an integer >= 100")

    response = _response_matrix(instrument_response_matrix, wavelength.size)
    if (laser_spectrum_wavelength_nm is None) != (laser_spectrum_density_per_nm is None):
        raise ValueError("Supply both laser_spectrum_wavelength_nm and laser_spectrum_density_per_nm, or neither.")
    laser_source_metadata = None
    if laser_spectrum_wavelength_nm is not None:
        from .thomson_inference import _laser_source_response
        source_operator, laser_source_metadata = _laser_source_response(
            wavelength, float(laser_wavelength_nm), laser_spectrum_wavelength_nm,
            laser_spectrum_density_per_nm,
        )
        response = response @ source_operator
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

    # Use the same midpoint finite-volume measure as the spectral-distribution
    # analysis; endpoint bins are half-width, including on non-uniform grids.
    widths = _wavelength_bin_widths(wavelength)

    # Integrate the nodal EVDF density over velocity space with the same
    # trapezoidal measure used for normalization and physical moments.
    # Without these weights, nonuniform velocity grids solve an unweighted
    # sum of delta-lines rather than the physical integral int f(v)K(lambda|v)dv.
    forward, forward_metadata = _projected_evdf_forward_matrix(
        wavelength, velocity, response, float(laser_wavelength_nm), q_over_ki
    )

    residual_data = measured - background
    if signal_uncertainty is not None and signal_covariance is not None:
        raise ValueError("Supply signal_uncertainty or signal_covariance, not both")
    if background_uncertainty is not None and background_covariance is not None:
        raise ValueError("Supply background_uncertainty or background_covariance, not both")

    def _validated_covariance(value, name):
        matrix = np.asarray(value)
        if matrix.dtype.kind not in "iuf":
            raise ValueError(f"{name} must contain real numeric values")
        matrix = matrix.astype(float, copy=False)
        if matrix.shape != (measured.size, measured.size) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be a finite square matrix matching signal")
        covariance_scale = max(float(np.max(np.abs(matrix))), np.finfo(float).tiny)
        if not np.allclose(
            matrix, matrix.T, rtol=1e-10, atol=1e-12 * covariance_scale
        ):
            raise ValueError(f"{name} must be symmetric")
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if eigenvalues[0] < -1e-10 * scale:
            raise ValueError(f"{name} must be positive semidefinite")
        return matrix

    signal_covariance_matrix = None
    if signal_uncertainty is not None:
        sigma = _finite_vector(signal_uncertainty, "signal_uncertainty")
        if sigma.size != measured.size or np.any(sigma <= 0):
            raise ValueError("signal_uncertainty must be strictly positive and match signal")
        signal_covariance_matrix = np.diag(sigma**2)
    elif signal_covariance is not None:
        signal_covariance_matrix = _validated_covariance(signal_covariance, "signal_covariance")

    background_covariance_matrix = None
    if background_uncertainty is not None:
        background_sigma = _finite_vector(background_uncertainty, "background_uncertainty")
        if background_sigma.size != measured.size or np.any(background_sigma <= 0):
            raise ValueError("background_uncertainty must be strictly positive and match signal")
        background_covariance_matrix = np.diag(background_sigma**2)
    elif background_covariance is not None:
        background_covariance_matrix = _validated_covariance(background_covariance, "background_covariance")

    if signal_background_cross_covariance is not None:
        if signal_covariance_matrix is None or background_covariance_matrix is None:
            raise ValueError("signal_background_cross_covariance requires both signal and background uncertainty models")
        cross = np.asarray(signal_background_cross_covariance)
        if cross.dtype.kind not in "iuf" or cross.shape != (measured.size, measured.size) or not np.all(np.isfinite(cross)):
            raise ValueError("signal_background_cross_covariance must be a finite square matrix matching signal")
        cross = cross.astype(float, copy=False)
        joint_covariance = np.block([
            [signal_covariance_matrix, cross],
            [cross.T, background_covariance_matrix],
        ])
        try:
            np.linalg.cholesky(joint_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("joint signal/background covariance must be positive definite") from exc
        residual_covariance = signal_covariance_matrix + background_covariance_matrix - cross - cross.T
        uncertainty_model = "joint_signal_background_covariance"
    elif signal_covariance_matrix is not None and background_covariance_matrix is not None:
        residual_covariance = signal_covariance_matrix + background_covariance_matrix
        uncertainty_model = "independent_signal_and_background_covariance"
    elif signal_covariance_matrix is not None:
        residual_covariance = signal_covariance_matrix
        uncertainty_model = "signal_measurement_covariance_background_fixed"
    elif background_covariance_matrix is not None:
        residual_covariance = background_covariance_matrix
        uncertainty_model = "background_covariance_signal_noise_not_supplied"
    else:
        residual_covariance = None
        uncertainty_model = "not_supplied_unweighted_fit"

    discrepancy_covariance_matrix = (
        None if model_discrepancy_covariance is None else
        _validated_covariance(model_discrepancy_covariance, "model_discrepancy_covariance")
    )
    systematic_covariance_matrix = (
        None if systematic_covariance is None else
        _validated_covariance(systematic_covariance, "systematic_covariance")
    )
    additive_components = [matrix for matrix in (
        discrepancy_covariance_matrix, systematic_covariance_matrix
    ) if matrix is not None]
    if additive_components:
        additive_covariance = np.sum(additive_components, axis=0)
        residual_covariance = (
            additive_covariance if residual_covariance is None
            else residual_covariance + additive_covariance
        )
        if signal_covariance_matrix is None and background_covariance_matrix is None:
            labels = []
            if discrepancy_covariance_matrix is not None:
                labels.append("model_discrepancy")
            if systematic_covariance_matrix is not None:
                labels.append("systematic")
            uncertainty_model = "_plus_".join(labels) + "_only_weighting"
        else:
            labels = []
            if discrepancy_covariance_matrix is not None:
                labels.append("model_discrepancy")
            if systematic_covariance_matrix is not None:
                labels.append("systematic")
            uncertainty_model += "+independent_" + "_plus_".join(labels) + "_covariance"

    if residual_covariance is not None:
        try:
            chol = np.linalg.cholesky(residual_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("combined signal/background residual covariance must be positive definite") from exc
        design = np.linalg.solve(chol, forward)
        target = np.linalg.solve(chol, residual_data)
    else:
        scale = max(float(np.max(np.abs(residual_data))), np.finfo(float).tiny)
        design = forward / scale
        target = residual_data / scale

    dv = np.diff(velocity)
    spacing = float(np.min(dv))
    # Non-uniform-grid second derivative: 2/(h_left+h_right) times
    # the difference of adjacent first-derivative slopes.  This keeps the
    # Tikhonov penalty in physical (m/s)^-2 units rather than penalizing
    # changes per array index when velocity nodes are unequally spaced.
    smooth = _nonuniform_second_derivative_matrix(velocity)
    smooth_scale = np.linalg.norm(design, ord="fro") / max(np.linalg.norm(smooth, ord="fro"), np.finfo(float).tiny)
    regularization_candidates = (
        np.logspace(-8.0, 4.0, 13)
        if regularization_strength is None
        else np.asarray([float(regularization_strength)])
    )
    gcv_scores = []
    candidate_fits = []
    for candidate in regularization_candidates:
        augmented_design = np.vstack([
            design, np.sqrt(float(candidate)) * smooth_scale * smooth
        ])
        augmented_target = np.concatenate([target, np.zeros(smooth.shape[0])])
        candidate_fit = lsq_linear(
            augmented_design, augmented_target, bounds=(0.0, np.inf),
            method="trf", max_iter=int(max_iter), tol=1e-10,
        )
        if not candidate_fit.success or not np.all(np.isfinite(candidate_fit.x)):
            gcv_scores.append(float("inf"))
            candidate_fits.append(None)
            continue
        active = candidate_fit.x > max(
            float(np.max(candidate_fit.x)) * 1e-9, np.finfo(float).tiny
        )
        if not np.any(active):
            gcv_scores.append(float("inf"))
            candidate_fits.append(None)
            continue
        active_design = design[:, active]
        active_penalty = smooth[:, active] * smooth_scale
        hessian = (
            active_design.T @ active_design
            + float(candidate) * (active_penalty.T @ active_penalty)
        )
        try:
            effective_df = float(np.trace(
                np.linalg.solve(hessian, active_design.T @ active_design)
            ))
        except np.linalg.LinAlgError:
            effective_df = float("inf")
        residual_candidate = target - design @ candidate_fit.x
        rss_candidate = float(residual_candidate @ residual_candidate)
        denominator = max(float(target.size) - effective_df, 1e-9)
        gcv_scores.append(float(target.size * rss_candidate / denominator**2))
        candidate_fits.append(candidate_fit)

    if not np.any(np.isfinite(gcv_scores)):
        raise ValueError("EVDF regularization selection failed for every candidate; the measured spectrum does not constrain a stable non-negative solution")
    selected_index = int(np.argmin(gcv_scores))
    selected_regularization_strength = float(regularization_candidates[selected_index])
    fit = candidate_fits[selected_index]
    if fit is None or not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError("EVDF non-negative regularized inversion failed after GCV selection")
    coefficients = np.maximum(fit.x, 0.0)
    distribution, normalization_jacobian = _normalize_evdf_coefficients(
        coefficients, velocity
    )
    norm = float(trapezoid(distribution, velocity))
    if not np.isfinite(norm) or not np.isclose(norm, 1.0, rtol=1e-10, atol=1e-12):
        raise ValueError("EVDF inversion failed trapezoidal probability normalization")

    # Conditional active-set covariance. This is a local linear approximation
    # to the constrained inverse, conditional on the selected regularization.
    active = coefficients > max(float(np.max(coefficients)) * 1e-9, np.finfo(float).tiny)
    active_design = design[:, active]
    active_penalty = smooth[:, active] * smooth_scale
    hessian = (
        active_design.T @ active_design
        + selected_regularization_strength * (active_penalty.T @ active_penalty)
    )
    covariance_scale = 1.0
    if uncertainty_model == "not_supplied_unweighted_fit":
        active_residual = target - design @ coefficients
        covariance_scale = float(active_residual @ active_residual) / max(
            1, target.size - int(np.count_nonzero(active))
        )
    coefficient_covariance = np.zeros((velocity.size, velocity.size), dtype=float)
    try:
        active_covariance = np.linalg.pinv(hessian, hermitian=True) * covariance_scale
        active_indices = np.flatnonzero(active)
        coefficient_covariance[np.ix_(active_indices, active_indices)] = active_covariance
        evdf_covariance = (
            normalization_jacobian @ coefficient_covariance @ normalization_jacobian.T
        )
        evdf_standard_uncertainty = np.sqrt(
            np.maximum(np.diag(evdf_covariance), 0.0)
        )
        evdf_covariance_available = bool(np.all(np.isfinite(evdf_covariance)))
    except (np.linalg.LinAlgError, FloatingPointError):
        evdf_covariance = None
        evdf_standard_uncertainty = None
        evdf_covariance_available = False

    fitted = forward @ coefficients
    residual = residual_data - fitted
    integral = lambda values: float(trapezoid(values, velocity))
    mean_v = integral(velocity * distribution)
    centred = velocity - mean_v
    variance = integral(centred**2 * distribution)
    if variance <= 0 or not np.isfinite(variance):
        raise ValueError("Reconstructed EVDF has non-positive variance")
    sigma_v = float(np.sqrt(variance))
    skewness = integral(centred**3 * distribution) / sigma_v**3
    excess_kurtosis = integral(centred**4 * distribution) / variance**2 - 3.0

    # Quantify how the inferred velocity marginal changes across the same
    # regularization candidates used by GCV. This is a numerical sensitivity
    # diagnostic, not an additional physical model or uncertainty interval.
    regularization_sensitivity_candidates = []
    for candidate, candidate_score, candidate_fit in zip(
        regularization_candidates, gcv_scores, candidate_fits
    ):
        if candidate_fit is None:
            continue
        candidate_coefficients = np.maximum(
            np.asarray(candidate_fit.x, dtype=float), 0.0
        )
        try:
            candidate_distribution, _ = _normalize_evdf_coefficients(
                candidate_coefficients, velocity
            )
        except ValueError:
            continue
        candidate_mean = integral(velocity * candidate_distribution)
        candidate_centred = velocity - candidate_mean
        candidate_variance = integral(
            candidate_centred**2 * candidate_distribution
        )
        if not np.isfinite(candidate_variance) or candidate_variance <= 0.0:
            continue
        candidate_sigma = float(np.sqrt(candidate_variance))
        candidate_skewness = (
            integral(candidate_centred**3 * candidate_distribution)
            / candidate_sigma**3
        )
        candidate_excess_kurtosis = (
            integral(candidate_centred**4 * candidate_distribution)
            / candidate_variance**2 - 3.0
        )
        regularization_sensitivity_candidates.append({
            "regularization_strength": float(candidate),
            "gcv_score": (
                float(candidate_score) if np.isfinite(candidate_score) else None
            ),
            "mean_velocity_parallel_m_s": float(candidate_mean),
            "velocity_parallel_standard_deviation_m_s": candidate_sigma,
            "skewness": float(candidate_skewness),
            "excess_kurtosis": float(candidate_excess_kurtosis),
            "total_variation_distance_from_selected": float(
                0.5 * trapezoid(
                    np.abs(candidate_distribution - distribution), velocity
                )
            ),
        })
    sensitivity_metrics = (
        "mean_velocity_parallel_m_s",
        "velocity_parallel_standard_deviation_m_s",
        "skewness",
        "excess_kurtosis",
        "total_variation_distance_from_selected",
    )
    regularization_sensitivity = {
        "status": (
            "assessed_over_gcv_candidate_grid"
            if regularization_strength is None
            else "single_explicit_regularization_value"
        ),
        "candidate_count": len(regularization_sensitivity_candidates),
        "selected_regularization_strength": selected_regularization_strength,
        "candidate_results": regularization_sensitivity_candidates,
        "ranges": {
            key: (
                {
                    "minimum": float(min(item[key] for item in regularization_sensitivity_candidates)),
                    "maximum": float(max(item[key] for item in regularization_sensitivity_candidates)),
                }
                if regularization_sensitivity_candidates else None
            )
            for key in sensitivity_metrics
        },
        "interpretation": (
            "Candidate-to-candidate numerical variation on the declared GCV grid; "
            "not a measurement uncertainty, posterior interval, or experimental validation."
        ),
    }
    electron_mass = 9.1093837139e-31
    boltzmann = 1.380649e-23
    directional_temperature_ev = electron_mass * variance / (boltzmann * 1.602176634e-19)

    # Propagate the normalized-EVDF covariance into the actual physical
    # velocity moments.  The covariance above is on the sampled probability
    # density, so use the same trapezoidal quadrature measure as the reported
    # moments; do not treat the grid samples as independent point estimates.
    moment_covariance = None
    moment_standard_uncertainty = None
    moment_covariance_order = [
        "mean_velocity_parallel_m_s",
        "velocity_parallel_variance_m2_s2",
        "velocity_parallel_standard_deviation_m_s",
        "directional_equivalent_temperature_eV",
        "skewness",
        "excess_kurtosis",
    ]
    if evdf_covariance_available:
        spacing_weights = np.empty_like(velocity)
        spacing_weights[0] = 0.5 * (velocity[1] - velocity[0])
        spacing_weights[-1] = 0.5 * (velocity[-1] - velocity[-2])
        spacing_weights[1:-1] = 0.5 * (velocity[2:] - velocity[:-2])
        raw_m1 = spacing_weights * velocity
        raw_m2 = spacing_weights * velocity**2
        raw_m3 = spacing_weights * velocity**3
        raw_m4 = spacing_weights * velocity**4
        m1 = mean_v
        m2 = integral(velocity**2 * distribution)
        m3 = integral(velocity**3 * distribution)
        m4 = integral(velocity**4 * distribution)
        central_m3 = m3 - 3.0*m1*m2 + 2.0*m1**3
        central_m4 = m4 - 4.0*m1*m3 + 6.0*m1**2*m2 - 3.0*m1**4
        variance_gradient = raw_m2 - 2.0*m1*raw_m1
        sigma_gradient = variance_gradient / (2.0*sigma_v)
        temperature_gradient = (
            electron_mass / (boltzmann * 1.602176634e-19)
        ) * variance_gradient
        central_m3_gradient = (
            raw_m3 + (-3.0*m2 + 6.0*m1**2)*raw_m1 - 3.0*m1*raw_m2
        )
        skewness_gradient = (
            central_m3_gradient / variance**1.5
            - 1.5*central_m3*variance_gradient / variance**2.5
        )
        central_m4_gradient = (
            raw_m4 + (-4.0*m3 + 12.0*m1*m2 - 12.0*m1**3)*raw_m1
            - 4.0*m1*raw_m3 + 6.0*m1**2*raw_m2
        )
        kurtosis_gradient = (
            central_m4_gradient / variance**2
            - 2.0*central_m4*variance_gradient / variance**3
        )
        moment_jacobian = np.vstack((
            raw_m1,
            variance_gradient,
            sigma_gradient,
            temperature_gradient,
            skewness_gradient,
            kurtosis_gradient,
        ))
        moment_covariance = moment_jacobian @ evdf_covariance @ moment_jacobian.T
        moment_covariance = 0.5 * (moment_covariance + moment_covariance.T)
        if not np.all(np.isfinite(moment_covariance)):
            moment_covariance = None
        else:
            moment_standard_uncertainty = {
                name: float(np.sqrt(max(float(moment_covariance[i, i]), 0.0)))
                for i, name in enumerate(moment_covariance_order)
            }

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
        area_pdf = float(trapezoid(raw_pdf, velocity))
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
            # The projected Student-t variance exists for nu > 2. Do not
            # impose an arbitrary finite upper limit on the physical tail
            # parameter; only enforce the distribution's mathematical domain.
            lower_bounds = [0.0, float(velocity[0]), spacing, np.nextafter(2.0, np.inf)]
            upper_bounds = [np.inf, float(velocity[-1]), float(velocity[-1]-velocity[0]), np.inf]
        else:
            initial = [float(np.max(np.abs(target))), mean_v, max(sigma_v, spacing), 2.0]
            # exp(-|z|**p) is normalizable for every p > 0. The previous
            # [1.2, 8] interval was an undocumented researcher-independent
            # restriction and is therefore not part of the inference contract.
            lower_bounds = [0.0, float(velocity[0]), spacing, np.nextafter(0.0, np.inf)]
            upper_bounds = [np.inf, float(velocity[-1]), float(velocity[-1]-velocity[0]), np.inf]

        def model_residual(parameters):
            pdf = projected_pdf(model_name, parameters[1:])
            quadrature = np.empty_like(velocity)
            quadrature[0] = 0.5 * (velocity[1] - velocity[0])
            quadrature[-1] = 0.5 * (velocity[-1] - velocity[-2])
            quadrature[1:-1] = 0.5 * (velocity[2:] - velocity[:-2])
            probability = pdf * quadrature
            probability /= max(float(np.sum(probability)), np.finfo(float).tiny)
            return design @ (probability * parameters[0]) - target

        try:
            model_fit = least_squares(
                model_residual, initial, bounds=(lower_bounds, upper_bounds),
                max_nfev=4000, xtol=1e-10, ftol=1e-10, gtol=1e-10,
            )
            if not model_fit.success or not np.all(np.isfinite(model_fit.x)):
                parametric_comparison[model_name] = {
                    "success": False,
                    "message": str(model_fit.message),
                    "weighted_residual_sum_squares": None,
                    "aic": None,
                    "bic": None,
                    "parameter_covariance": None,
                    "parameter_standard_uncertainty": None,
                    "parameter_one_sigma_intervals": None,
                    "parameter_uncertainty_status": "unavailable_optimizer_not_converged",
                    "uncertainty_intervals": None,
                }
                continue
            model_resid = model_residual(model_fit.x)
            rss = float(model_resid @ model_resid)
            sample_count = int(target.size)
            parameter_count = int(model_fit.x.size)
            safe_rss = max(rss, np.finfo(float).tiny)
            # With supplied, correctly whitened measurement errors, the
            # Gaussian -2 log-likelihood is chi-square plus a parameter-
            # independent constant. The n*log(RSS/n) form is appropriate
            # only when a common residual variance is estimated from data.
            if uncertainty_model == "not_supplied_unweighted_fit":
                aic = float(sample_count * np.log(safe_rss / sample_count) + 2 * parameter_count)
                bic = float(sample_count * np.log(safe_rss / sample_count) + parameter_count * np.log(sample_count))
                information_criterion_likelihood = "unknown_common_residual_variance_estimated_from_unweighted_residuals"
            else:
                aic = float(rss + 2 * parameter_count)
                bic = float(rss + parameter_count * np.log(sample_count))
                information_criterion_likelihood = "known_measurement_covariance_gaussian_likelihood_up_to_parameter_independent_constant"
            # Local parameter covariance for each candidate model. The residual
            # vector is already whitened when a measurement uncertainty/covariance
            # was supplied; otherwise its residual variance provides only a
            # residual-scaled, non-measurement-calibrated estimate.
            jacobian = np.asarray(model_fit.jac, dtype=float)
            jacobian_rank = int(np.linalg.matrix_rank(jacobian))
            parameter_covariance = None
            parameter_standard_uncertainty = None
            parameter_one_sigma_intervals = None
            parameter_uncertainty_status = "unavailable_rank_deficient_jacobian"
            if jacobian_rank == parameter_count and np.all(np.isfinite(jacobian)):
                try:
                    parameter_covariance = np.linalg.pinv(
                        jacobian.T @ jacobian, hermitian=True
                    )
                    if uncertainty_model == "not_supplied_unweighted_fit":
                        parameter_covariance *= rss / max(1, sample_count - parameter_count)
                    diagonal = np.maximum(np.diag(parameter_covariance), 0.0)
                    parameter_standard_uncertainty = np.sqrt(diagonal)
                    parameter_names = [
                        "amplitude_scaled", "mean_velocity_parallel_m_s",
                        "scale_velocity_m_s",
                    ]
                    if model_name == "kappa_student_t":
                        parameter_names.append("kappa_degrees_of_freedom")
                    elif model_name == "super_gaussian":
                        parameter_names.append("super_gaussian_exponent")
                    intervals = {}
                    boundary_limited = []
                    for index, name in enumerate(parameter_names):
                        value = float(model_fit.x[index])
                        sigma_value = float(parameter_standard_uncertainty[index])
                        lower_gap = value - float(lower_bounds[index])
                        upper_gap = (
                            float(upper_bounds[index]) - value
                            if np.isfinite(upper_bounds[index]) else np.inf
                        )
                        tolerance = 1e-7 * max(1.0, abs(value))
                        if lower_gap <= tolerance or upper_gap <= tolerance:
                            intervals[name] = None
                            boundary_limited.append(name)
                        else:
                            intervals[name] = [value - sigma_value, value + sigma_value]
                    parameter_one_sigma_intervals = intervals
                    parameter_uncertainty_status = (
                        "boundary_limited_for_some_parameters"
                        if boundary_limited else "conditional_local_gaussian_covariance"
                    )
                except (np.linalg.LinAlgError, FloatingPointError):
                    parameter_covariance = None
                    parameter_standard_uncertainty = None
                    parameter_one_sigma_intervals = None
                    parameter_uncertainty_status = "unavailable_covariance_factorization"
            parametric_comparison[model_name] = {
                "parameters": {
                    "amplitude_scaled": float(model_fit.x[0]),
                    "mean_velocity_parallel_m_s": float(model_fit.x[1]),
                    "scale_velocity_m_s": float(model_fit.x[2]),
                    **({"kappa_degrees_of_freedom": float(model_fit.x[3])} if model_name == "kappa_student_t" else {}),
                    **({"super_gaussian_exponent": float(model_fit.x[3])} if model_name == "super_gaussian" else {}),
                },
                "weighted_residual_sum_squares": rss,
                "aic": aic,
                "bic": bic,
                "information_criterion_likelihood": information_criterion_likelihood,
                "success": True,
                "message": str(model_fit.message),
                "parameter_covariance": parameter_covariance,
                "parameter_standard_uncertainty": parameter_standard_uncertainty,
                "parameter_one_sigma_intervals": parameter_one_sigma_intervals,
                "parameter_uncertainty_status": parameter_uncertainty_status,
                "parameter_covariance_interpretation": (
                    "Local linearized covariance conditional on the selected projected-marginal model, "
                    "velocity grid, measured response, background and residual/noise representation. "
                    "It excludes model discrepancy and unprovided systematic uncertainty."
                ),
                "uncertainty_intervals": parameter_one_sigma_intervals,
            }
        except (ValueError, FloatingPointError) as exc:
            parametric_comparison[model_name] = {
                "success": False, "message": str(exc),
                "weighted_residual_sum_squares": None, "aic": None, "bic": None,
                "uncertainty_intervals": None,
            }

    # Model-selection uncertainty: average the candidate projected marginals
    # using normalized Akaike relative support. These weights quantify support
    # only within the declared candidate set and likelihood; they are not
    # posterior probabilities and do not include models absent from that set.
    eligible_models = {
        name: item for name, item in parametric_comparison.items()
        if item.get("success") is True
        and item.get("aic") is not None
        and np.isfinite(item.get("aic"))
        and isinstance(item.get("parameters"), dict)
    }
    model_selection_uncertainty = {
        "status": "unavailable_no_converged_candidate_models",
        "candidate_set": ["maxwellian", "kappa_student_t", "super_gaussian"],
        "weights": {},
        "delta_aic": {},
        "model_averaged": None,
        "interpretation": (
            "Akaike relative support within the explicitly fitted candidate set; "
            "not Bayesian posterior model probability and not evidence for models "
            "outside the candidate set."
        ),
    }
    if eligible_models:
        minimum_aic = min(float(item["aic"]) for item in eligible_models.values())
        relative_likelihoods = {
            name: float(np.exp(-0.5 * (float(item["aic"]) - minimum_aic)))
            for name, item in eligible_models.items()
        }
        likelihood_sum = float(sum(relative_likelihoods.values()))
        weights = {
            name: value / likelihood_sum
            for name, value in relative_likelihoods.items()
        }
        def _model_average(parameter_index, parameter_name):
            values = {
                name: float(item["parameters"][parameter_name])
                for name, item in eligible_models.items()
                if parameter_name in item["parameters"]
                and np.isfinite(item["parameters"][parameter_name])
            }
            active_weights = {name: weights[name] for name in values}
            normalization = float(sum(active_weights.values()))
            if not values or normalization <= 0.0:
                return None
            active_weights = {
                name: value / normalization for name, value in active_weights.items()
            }
            mean_value = float(sum(active_weights[name] * values[name] for name in values))
            between_variance = float(sum(
                active_weights[name] * (values[name] - mean_value) ** 2
                for name in values
            ))
            within_variance = 0.0
            within_available = True
            for name in values:
                covariance = eligible_models[name].get("parameter_covariance")
                if covariance is None:
                    within_available = False
                    break
                covariance = np.asarray(covariance, dtype=float)
                if (covariance.ndim != 2 or covariance.shape[0] <= parameter_index
                        or covariance.shape[1] <= parameter_index
                        or not np.isfinite(covariance[parameter_index, parameter_index])
                        or covariance[parameter_index, parameter_index] < 0.0):
                    within_available = False
                    break
                within_variance += active_weights[name] * float(
                    covariance[parameter_index, parameter_index]
                )
            total_variance = between_variance + within_variance if within_available else None
            return {
                "mean": mean_value,
                "between_model_standard_uncertainty": float(np.sqrt(max(between_variance, 0.0))),
                "within_model_standard_uncertainty": (
                    float(np.sqrt(max(within_variance, 0.0))) if within_available else None
                ),
                "total_model_averaged_standard_uncertainty": (
                    float(np.sqrt(max(total_variance, 0.0)))
                    if total_variance is not None else None
                ),
                "within_model_covariance_available": within_available,
                "candidate_count": len(values),
            }
        model_selection_uncertainty = {
            "status": "akaike_weighted_model_uncertainty",
            "candidate_set": ["maxwellian", "kappa_student_t", "super_gaussian"],
            "weights": weights,
            "delta_aic": {
                name: float(item["aic"] - minimum_aic)
                for name, item in eligible_models.items()
            },
            "model_averaged": {
                "mean_velocity_parallel_m_s": _model_average(
                    1, "mean_velocity_parallel_m_s"
                ),
                "scale_velocity_m_s": _model_average(2, "scale_velocity_m_s"),
            },
            "interpretation": (
                "Between-model spread is added to weighted within-model local "
                "covariance where every contributing covariance is available. "
                "Akaike weights are relative support conditional on this candidate "
                "set and likelihood, not Bayesian posterior probabilities; the "
                "result does not cover unmodeled EVDF families or regularization "
                "selection uncertainty."
            ),
        }

    return {
        "laser_spectral_profile": laser_source_metadata or {"assumption": "monochromatic_incident_laser"},
        "velocity_parallel_m_s": velocity,
        "evdf_parallel_s_per_m": distribution,
        "evdf_normalization": integral(distribution),
        "mean_velocity_parallel_m_s": mean_v,
        "velocity_parallel_standard_deviation_m_s": sigma_v,
        "directional_equivalent_temperature_eV": directional_temperature_ev,
        "velocity_moment_covariance_order": moment_covariance_order,
        "velocity_moment_covariance": moment_covariance,
        "velocity_moment_standard_uncertainty": moment_standard_uncertainty,
        "velocity_moment_uncertainty_status": (
            "delta_method_propagated_from_normalized_evdf_covariance"
            if moment_covariance is not None else
            "unavailable_evdf_covariance_not_finite_or_not_identifiable"
        ),
        "velocity_moment_covariance_interpretation": (
            "First-order Jacobian propagation of the conditional normalized-EVDF covariance through the same trapezoidal first-to-fourth velocity moments; correlations among flow, directional variance, equivalent temperature, skewness and kurtosis are retained."
            if moment_covariance is not None else
            "No physical-moment uncertainty is reported because a finite normalized-EVDF covariance is unavailable."
        ),
        "skewness": float(skewness),
        "excess_kurtosis": float(excess_kurtosis),
        "fitted_spectrum": fitted,
        "residual": residual,
        "residual_sum_squares": float(np.sum(residual**2)),
        "regularization_strength": selected_regularization_strength,
        "regularization_selection": (
            "generalized_cross_validation_over_logarithmic_candidate_grid"
            if regularization_strength is None else "explicit_researcher_supplied_value"
        ),
        "regularization_candidates": regularization_candidates.copy(),
        "regularization_gcv_scores": np.asarray(gcv_scores, dtype=float),
        "regularization_sensitivity": regularization_sensitivity,
        "regularization": "non-negative second-difference Tikhonov",
        "evdf_covariance": evdf_covariance,
        "evdf_standard_uncertainty_s_per_m": evdf_standard_uncertainty,
        "evdf_covariance_available": evdf_covariance_available,
        "evdf_uncertainty_interpretation": (
            "local active-set covariance conditional on the selected grid, response, background, "
            "regularization and active set; not a posterior or experimental validation"
        ),
        "measurement_uncertainty_model": uncertainty_model,
        "model_discrepancy_covariance": (
            None if discrepancy_covariance_matrix is None else discrepancy_covariance_matrix.copy()
        ),
        "model_discrepancy_covariance_included": discrepancy_covariance_matrix is not None,
        "systematic_covariance": (
            None if systematic_covariance_matrix is None else systematic_covariance_matrix.copy()
        ),
        "systematic_covariance_included": systematic_covariance_matrix is not None,
        "additive_covariance_independence_assumption": (
            "model discrepancy and systematic components are mutually independent and independent of measurement/background errors; all omitted cross-covariances are zero by explicit assumption"
            if additive_components else "no additive model-discrepancy/systematic covariance supplied"
        ),
        "background_uncertainty_included": background_covariance_matrix is not None,
        "signal_background_cross_covariance_included": signal_background_cross_covariance is not None,
        "residual_covariance_interpretation": (
            "Cov(signal-background)=Cov(signal)+Cov(background)-Cov(signal,background)-Cov(background,signal); "
            "conditional on the supplied covariance matrices"
            if residual_covariance is not None else "no signal or background uncertainty supplied; unweighted fit"
        ),
        "forward_model": "non-relativistic single-particle Doppler projection with measured response and trapezoidal velocity-space quadrature",
        "velocity_quadrature_weights_m_s": forward_metadata["velocity_quadrature_weights_m_s"],
        "forward_operator_column_integrated_spectral_power_per_unit_evdf": forward_metadata["column_integrated_spectral_power_per_unit_evdf"],
        "velocity_integral_discretization": "trapezoidal_finite_volume_measure_on_nodal_EVDF_density",
        "scattering_vector_projection_only": True,
        "absolute_density_identified": False,
        "numerical_design_rank": numerical_rank,
        "velocity_grid_points": int(velocity.size),
        "design_matrix_condition_number": condition,
        "parametric_model_comparison": parametric_comparison,
        "parametric_model_selection_uncertainty": model_selection_uncertainty,
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


# Exact relativistic Thomson-limit projected-marginal inverse (separate from the non-relativistic routine).
from .thomson_gap_closure import reconstruct_relativistic_projected_thomson_evdf

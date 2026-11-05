"""Kinetic Thomson models beyond non-relativistic Maxwellian fitting.

The relativistic single-particle model uses exact photon-frequency kinematics
for a Lorentz-boosted Maxwell-Juttner electron population. The collective
kappa model evaluates the unmagnetized longitudinal dielectric response from
the supplied 1-D projected electron VDF and Maxwellian ion populations.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.special import gammaln, ndtr, stdtr, kve
from numpy.polynomial.legendre import leggauss

E_CHARGE = 1.602176634e-19
EPS0 = 8.8541878128e-12
ME = 9.1093837015e-31
AMU = 1.66053906660e-27
C = 299792458.0
E_REST_EV = 510998.95069
ELECTRON_CLASSICAL_RADIUS_M = 2.8179403262e-15
_LANDAU_NODES, _LANDAU_WEIGHTS = leggauss(192)
_FULL_AXIS_LANDAU_QUADRATURE = {order: leggauss(order) for order in (384, 768, 1536)}


def _trapezoid(y, x):
    """NumPy-version-independent trapezoidal integration."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(y, x))


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value):
        raise ValueError(f"{name} must be positive and finite")
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def project_bi_maxwellian_electron_vdf(
    velocity_grid_m_s, *, temperature_parallel_eV, temperature_perpendicular_eV,
    magnetic_field_unit, scattering_vector_unit, flow_projection_m_s=0.0,
):
    """Return the gyrotropic bi-Maxwellian marginal along the scattering vector.

    For a gyrotropic bi-Maxwellian with zero heat flux, the 1-D marginal along
    q-hat is Gaussian with
        T_q = T_perp + (T_parallel - T_perp) (b-hat . q-hat)**2.
    This is a forward marginal, not an inverse fit. A single scattering
    direction identifies T_q only; it cannot separate T_parallel from
    T_perpendicular. Multiple independent pitch angles are required for that
    separation. The analytic distribution is normalized on the infinite velocity axis. A finite supplied grid retains the represented probability mass and reports its coverage without renormalization.
    """
    grid = np.asarray(velocity_grid_m_s)
    if grid.dtype.kind not in "iuf" or grid.ndim != 1 or grid.size < 3:
        raise ValueError("velocity_grid_m_s must be a real one-dimensional grid with at least three points")
    grid = grid.astype(float, copy=False)
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("velocity_grid_m_s must be finite and strictly increasing")
    t_parallel = _positive(temperature_parallel_eV, "temperature_parallel_eV")
    t_perpendicular = _positive(temperature_perpendicular_eV, "temperature_perpendicular_eV")
    b = np.asarray(magnetic_field_unit)
    q = np.asarray(scattering_vector_unit)
    if b.dtype.kind not in "iuf" or q.dtype.kind not in "iuf" or b.shape != (3,) or q.shape != (3,):
        raise ValueError("magnetic_field_unit and scattering_vector_unit must be real three-component vectors")
    b = b.astype(float, copy=False)
    q = q.astype(float, copy=False)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(q)):
        raise ValueError("direction vectors must be finite")
    if not np.isclose(np.linalg.norm(b), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("magnetic_field_unit must have unit norm")
    if not np.isclose(np.linalg.norm(q), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("scattering_vector_unit must have unit norm")
    if isinstance(flow_projection_m_s, (bool, np.bool_)) or not np.isscalar(flow_projection_m_s):
        raise ValueError("flow_projection_m_s must be a finite scalar")
    flow = float(flow_projection_m_s)
    if not np.isfinite(flow):
        raise ValueError("flow_projection_m_s must be a finite scalar")
    pitch_cosine = float(np.clip(np.dot(b, q), -1.0, 1.0))
    directional_temperature = (
        t_perpendicular
        + (t_parallel - t_perpendicular) * pitch_cosine**2
    )
    variance = E_CHARGE * directional_temperature / ME
    sigma = float(np.sqrt(variance))
    standardized = (grid - flow) / sigma
    distribution = np.exp(-0.5 * standardized**2) / (np.sqrt(2.0 * np.pi) * sigma)
    represented_probability = float(
        ndtr((grid[-1] - flow) / sigma) - ndtr((grid[0] - flow) / sigma)
    )
    if not np.isfinite(represented_probability) or represented_probability <= 0.0:
        raise ValueError("velocity_grid_m_s does not cover a positive finite portion of the projected distribution")
    # Do not renormalize a truncated grid: doing so would silently change the
    # physical distribution and bias its moments near the velocity boundaries.
    return {
        "velocity_grid_m_s": grid.copy(),
        "projected_vdf_s_m": distribution,
        "velocity_grid_probability_coverage": represented_probability,
        "projected_temperature_eV": float(directional_temperature),
        "projected_velocity_mean_m_s": flow,
        "projected_velocity_standard_deviation_m_s": sigma,
        "magnetic_pitch_cosine": pitch_cosine,
        "anisotropy_identifiability": (
            "only_the_directional_temperature_combination_is_identified_from_one_scattering_direction"
        ),
        "separate_parallel_perpendicular_temperatures_identifiable": False,
        "model": "gyrotropic_nonrelativistic_bi_maxwellian_projected_marginal",
        "assumptions": [
            "Gyrotropic bi-Maxwellian electron distribution",
            "Zero heat flux and no non-Gaussian tails",
            "Non-relativistic velocity-space Gaussian marginal",
            "Magnetic-field and scattering-vector directions are supplied unit vectors",
        ],
        "limitations": [
            "This callable is a forward projected-VDF kernel, not a measured-spectrum optimizer",
            "A single scattering direction cannot independently identify parallel and perpendicular temperatures",
            "The finite velocity grid may truncate probability mass; coverage is reported and the represented marginal is not renormalized",
        ],
    }



def infer_gyrotropic_bi_maxwellian_temperatures(
    projected_temperature_eV, magnetic_pitch_cosine, *,
    projected_temperature_covariance_eV2,
    magnetic_pitch_cosine_covariance=None,
    projected_temperature_pitch_cross_covariance=None,
):
    """Infer T_parallel and T_perpendicular from independent 1-D TS projections.

    For a gyrotropic bi-Maxwellian with zero heat flux, the projected variance
    along scattering vector q is
        T_q = T_perp (1-mu**2) + T_parallel mu**2,
    where mu = b_hat dot q_hat.  This is a two-parameter GLS inversion of
    measured directional temperatures; it does not use a supplied 3-D flow.

    A single pitch angle (or repeated equivalent |mu|) cannot separate the
    two temperatures. In that case no anisotropy parameters are returned.
    Optional pitch-cosine covariance is propagated by first-order errors-in-
    variables GLS. An optional cross-covariance C_Tmu between measured
    directional temperatures and pitch cosines is also retained. With
    D=diag(2 mu (T_parallel-T_perp)), the effective covariance is
      C_eff = C_T + D C_mu D^T + C_Tmu D^T + D C_Tmu^T.
    The fit iterates because D depends on the inferred anisotropy. The
    temperature, geometry and cross-covariance inputs must form a positive
    semidefinite joint covariance block.
    """
    tq = np.asarray(projected_temperature_eV)
    mu = np.asarray(magnetic_pitch_cosine)
    cov = np.asarray(projected_temperature_covariance_eV2)
    if tq.dtype.kind not in "iuf" or tq.ndim != 1 or tq.size < 1:
        raise ValueError("projected_temperature_eV must be a non-empty real vector")
    if mu.dtype.kind not in "iuf" or mu.shape != tq.shape:
        raise ValueError("magnetic_pitch_cosine must match projected_temperature_eV")
    if cov.dtype.kind not in "iuf" or cov.shape != (tq.size, tq.size):
        raise ValueError("projected_temperature_covariance_eV2 must be square and match the observations")
    tq = tq.astype(float, copy=False)
    mu = mu.astype(float, copy=False)
    cov = cov.astype(float, copy=False)
    if (np.any(~np.isfinite(tq)) or np.any(tq <= 0)
            or np.any(~np.isfinite(mu)) or np.any(np.abs(mu) > 1.0)
            or np.any(~np.isfinite(cov))):
        raise ValueError("projected temperatures must be positive, pitch cosines in [-1,1], and covariance finite")
    covariance_scale = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12*covariance_scale):
        raise ValueError("projected_temperature_covariance_eV2 must be symmetric")
    cov = 0.5*(cov+cov.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    tolerance = np.finfo(float).eps*max(cov.shape)*max(float(eigenvalues[-1]), np.finfo(float).tiny)
    if eigenvalues[0] <= tolerance:
        raise ValueError("projected_temperature_covariance_eV2 must be positive definite")
    pitch_covariance = None
    if magnetic_pitch_cosine_covariance is not None:
        raw_pitch_covariance = np.asarray(magnetic_pitch_cosine_covariance)
        if (raw_pitch_covariance.dtype.kind not in "iuf"
                or raw_pitch_covariance.shape != (tq.size, tq.size)):
            raise ValueError("magnetic_pitch_cosine_covariance must be a real square matrix matching the views")
        pitch_covariance = raw_pitch_covariance.astype(float, copy=False)
        if np.any(~np.isfinite(pitch_covariance)):
            raise ValueError("magnetic_pitch_cosine_covariance must be finite")
        pitch_scale = max(float(np.max(np.abs(pitch_covariance))), np.finfo(float).tiny)
        if not np.allclose(pitch_covariance, pitch_covariance.T, rtol=1e-10, atol=1e-12*pitch_scale):
            raise ValueError("magnetic_pitch_cosine_covariance must be symmetric")
        pitch_covariance = 0.5*(pitch_covariance+pitch_covariance.T)
        pitch_eigenvalues = np.linalg.eigvalsh(pitch_covariance)
        pitch_tolerance = np.finfo(float).eps*max(pitch_covariance.shape)*max(float(pitch_eigenvalues[-1]), np.finfo(float).tiny)
        if pitch_eigenvalues[0] < -pitch_tolerance:
            raise ValueError("magnetic_pitch_cosine_covariance must be positive semidefinite")
    temperature_pitch_cross_covariance = None
    if projected_temperature_pitch_cross_covariance is not None:
        temperature_pitch_cross_covariance = np.asarray(
            projected_temperature_pitch_cross_covariance
        )
        if (temperature_pitch_cross_covariance.dtype.kind not in "iuf"
                or temperature_pitch_cross_covariance.shape != (tq.size, tq.size)):
            raise ValueError(
                "projected_temperature_pitch_cross_covariance must be a real "
                "n_views x n_views matrix Cov(T_q, mu)"
            )
        temperature_pitch_cross_covariance = temperature_pitch_cross_covariance.astype(
            float, copy=False
        )
        if np.any(~np.isfinite(temperature_pitch_cross_covariance)):
            raise ValueError("projected_temperature_pitch_cross_covariance must be finite")
        if pitch_covariance is None:
            raise ValueError(
                "projected_temperature_pitch_cross_covariance requires "
                "magnetic_pitch_cosine_covariance"
            )
        joint_input_covariance = np.block([
            [cov, temperature_pitch_cross_covariance],
            [temperature_pitch_cross_covariance.T, pitch_covariance],
        ])
        joint_input_covariance = 0.5 * (
            joint_input_covariance + joint_input_covariance.T
        )
        joint_input_eigenvalues = np.linalg.eigvalsh(joint_input_covariance)
        joint_input_scale = max(
            float(np.max(np.abs(joint_input_eigenvalues))), np.finfo(float).tiny
        )
        joint_input_tolerance = (
            1e-10 * joint_input_scale
        )
        if joint_input_eigenvalues[0] < -joint_input_tolerance:
            raise ValueError(
                "temperature, pitch-angle and cross-covariance block must be "
                "positive semidefinite"
            )
    mu2 = mu*mu
    design = np.column_stack((1.0-mu2, mu2))
    rank = int(np.linalg.matrix_rank(design))
    result = {
        "model": "gyrotropic_bi_maxwellian_multi_pitch_gls",
        "projected_temperature_model_eV": "T_q = T_perp*(1-mu^2) + T_parallel*mu^2",
        "magnetic_pitch_cosine": mu.copy(),
        "design_matrix": design,
        "design_rank": rank,
        "number_of_independent_pitch_angles": int(np.unique(np.round(mu2, 12)).size),
        "input_projected_temperature_eV": tq.copy(),
        "input_projected_temperature_covariance_eV2": cov.copy(),
        "geometry_uncertainty_included": pitch_covariance is not None,
        "input_magnetic_pitch_cosine_covariance": None if pitch_covariance is None else pitch_covariance.copy(),
        "input_projected_temperature_pitch_cross_covariance": (
            None if temperature_pitch_cross_covariance is None
            else temperature_pitch_cross_covariance.copy()
        ),
        "temperature_geometry_cross_covariance_included": (
            temperature_pitch_cross_covariance is not None
        ),
        "geometry_covariance_method": (
            "iterative_first_order_errors_in_variables_GLS_with_temperature_geometry_cross_terms"
            if temperature_pitch_cross_covariance is not None
            else "iterative_first_order_errors_in_variables_GLS"
            if pitch_covariance is not None else "not_supplied"
        ),
        "assumptions": [
            "Gyrotropic bi-Maxwellian electron distribution",
            "Zero heat flux and a common magnetic-field axis",
            "Each input is an independently inferred directional temperature",
            "Supplied covariance fully describes projected-temperature measurement errors",
        ],
    }
    if rank < 2:
        return {
            **result,
            "identifiability": "non_identifiable",
            "reportable_as_inferred_parameter": False,
            "electron_temperature_parallel_eV": None,
            "electron_temperature_perpendicular_eV": None,
            "temperature_covariance_eV2": None,
            "reason": "At least two distinct squared magnetic pitch cosines are required to separate T_parallel and T_perpendicular.",
        }

    def gls_solve(data_covariance):
        chol_local = np.linalg.cholesky(data_covariance)
        whitened_design_local = np.linalg.solve(chol_local, design)
        whitened_data_local = np.linalg.solve(chol_local, tq)
        information_local = whitened_design_local.T @ whitened_design_local
        # The unconstrained GLS solution can yield negative T_parallel or
        # T_perpendicular when noisy directional measurements conflict. Such
        # values are not physical temperatures. Solve the same GLS objective
        # with the thermodynamic constraints T_parallel,T_perpendicular > 0.
        # The lower bound is numerical only; an active bound is reported below
        # and its local Gaussian covariance must not be read as a symmetric CI.
        lower_temperature_eV = 1.0e-10
        solution = lsq_linear(
            whitened_design_local, whitened_data_local,
            bounds=(np.full(2, lower_temperature_eV), np.full(2, np.inf)),
            method="trf", tol=1.0e-12, lsmr_tol="auto", max_iter=1000,
        )
        if not solution.success or np.any(~np.isfinite(solution.x)):
            raise ValueError(
                "positive-temperature anisotropy GLS failed: " + str(solution.message)
            )
        covariance_local = np.linalg.inv(information_local)
        active_local = solution.x <= lower_temperature_eV * (1.0 + 1.0e-6)
        return solution.x, covariance_local, active_local

    effective_covariance = cov.copy()
    parameters, parameter_covariance, active_temperature_bounds = gls_solve(effective_covariance)
    geometry_iterations = 0
    if pitch_covariance is not None:
        for geometry_iterations in range(1, 51):
            derivative_mu = 2.0*mu*(parameters[1]-parameters[0])
            effective_covariance = (
                cov + derivative_mu[:, None]*pitch_covariance*derivative_mu[None, :]
            )
            if temperature_pitch_cross_covariance is not None:
                effective_covariance += (
                    temperature_pitch_cross_covariance*derivative_mu[None, :]
                    + derivative_mu[:, None]*temperature_pitch_cross_covariance.T
                )
            updated_parameters, updated_covariance, updated_active_bounds = gls_solve(effective_covariance)
            change = np.linalg.norm(updated_parameters-parameters) / max(
                np.linalg.norm(parameters), np.finfo(float).tiny
            )
            parameters, parameter_covariance = updated_parameters, updated_covariance
            active_temperature_bounds = updated_active_bounds
            if change < 1e-10:
                break
        else:
            raise ValueError("anisotropy geometry-covariance GLS did not converge in 50 iterations")
    residual = tq-design@parameters
    chi2 = float(residual @ np.linalg.solve(effective_covariance, residual))
    dof = int(tq.size-2)
    physical = bool(np.all(np.isfinite(parameters)) and np.all(parameters > 0.0))
    return {
        **result,
        "identifiability": "identifiable_under_gyrotropic_bi_maxwellian_assumption",
        "reportable_as_inferred_parameter": physical,
        "electron_temperature_perpendicular_eV": float(parameters[0]) if physical else None,
        "electron_temperature_parallel_eV": float(parameters[1]) if physical else None,
        "temperature_covariance_eV2": parameter_covariance if physical else None,
        "effective_projected_temperature_covariance_eV2": effective_covariance,
        "geometry_covariance_iterations": geometry_iterations,
        "parameter_order": ["electron_temperature_perpendicular_eV", "electron_temperature_parallel_eV"],
        "fit_residual_projected_temperature_eV": residual,
        "chi_square": chi2,
        "degrees_of_freedom": dof,
        "reduced_chi_square": chi2/dof if dof > 0 else None,
        "identifiability_reason": (
            "Two directional temperatures are separately identifiable from the supplied pitch-angle diversity."
        ),
        "physicality_status": (
            "positive_temperatures_at_active_lower_bound"
            if physical and np.any(active_temperature_bounds)
            else "positive_temperatures"
            if physical
            else "non_positive_GLS_temperature_estimate; no physical temperatures reported"
        ),
        "temperature_constraint": "T_perpendicular > 0 and T_parallel > 0 eV",
        "active_temperature_lower_bounds": {
            "T_perpendicular": bool(active_temperature_bounds[0]),
            "T_parallel": bool(active_temperature_bounds[1]),
        },
        "temperature_covariance_status": (
            "local_unconstrained_GLS_curvature_at_active_physical_bound; "
            "not_a_symmetric_confidence_interval"
            if np.any(active_temperature_bounds)
            else "local_GLS_covariance"
        ),
    }


def infer_three_dimensional_flow_from_thomson_projections(
    velocity_projection_m_s, scattering_vector_unit, *,
    projection_covariance_m2_s2,
    scattering_vector_covariance=None,
):
    """Infer a 3-D bulk-flow vector from independent measured TS projections.

    Each view measures v_q = q_hat dot v_e.  The vector is identifiable only
    when the stacked scattering-vector directions span R^3.  A single view
    therefore returns a non-identifiable result rather than an invented
    transverse velocity. Optional full Cartesian covariance of all scattering
    vectors is propagated by first-order iterative errors-in-variables GLS.
    Direction perturbations are projected onto each vector's tangent plane,
    because a normalized scattering direction has no radial degree of freedom.
    """
    projections = np.asarray(velocity_projection_m_s)
    directions = np.asarray(scattering_vector_unit)
    covariance = np.asarray(projection_covariance_m2_s2)
    if projections.dtype.kind not in "iuf" or projections.ndim != 1 or projections.size < 1:
        raise ValueError("velocity_projection_m_s must be a non-empty real vector")
    if directions.dtype.kind not in "iuf" or directions.shape != (projections.size, 3):
        raise ValueError("scattering_vector_unit must have shape (n_views, 3)")
    if covariance.dtype.kind not in "iuf" or covariance.shape != (projections.size, projections.size):
        raise ValueError("projection_covariance_m2_s2 must be square and match the views")
    projections = projections.astype(float, copy=False)
    directions = directions.astype(float, copy=False)
    covariance = covariance.astype(float, copy=False)
    norms = np.linalg.norm(directions, axis=1)
    if (np.any(~np.isfinite(projections)) or np.any(~np.isfinite(directions))
            or np.any(~np.isfinite(norms)) or np.any(norms <= 0.0)
            or np.any(~np.isfinite(covariance))):
        raise ValueError("projections, non-zero directions and covariance must be finite")
    if not np.allclose(norms, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("each scattering_vector_unit row must have unit norm")
    covariance_scale = max(float(np.max(np.abs(covariance))), np.finfo(float).tiny)
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12*covariance_scale):
        raise ValueError("projection_covariance_m2_s2 must be symmetric")
    covariance = 0.5*(covariance+covariance.T)
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = np.finfo(float).eps*max(covariance.shape)*max(float(eigenvalues[-1]), np.finfo(float).tiny)
    if eigenvalues[0] <= tolerance:
        raise ValueError("projection_covariance_m2_s2 must be positive definite")
    direction_covariance = None
    supplied_direction_covariance = None
    tangent_projector = np.zeros((3*projections.size, 3*projections.size), dtype=float)
    for view, direction in enumerate(directions):
        tangent_projector[3*view:3*view+3, 3*view:3*view+3] = (
            np.eye(3)-np.outer(direction, direction)
        )
    if scattering_vector_covariance is not None:
        raw_direction_covariance = np.asarray(scattering_vector_covariance)
        expected_shape = (3*projections.size, 3*projections.size)
        if raw_direction_covariance.dtype.kind not in "iuf" or raw_direction_covariance.shape != expected_shape:
            raise ValueError("scattering_vector_covariance must be a real (3*n_views) square covariance matrix")
        direction_covariance = raw_direction_covariance.astype(float, copy=False)
        if np.any(~np.isfinite(direction_covariance)):
            raise ValueError("scattering_vector_covariance must be finite")
        direction_scale = max(float(np.max(np.abs(direction_covariance))), np.finfo(float).tiny)
        if not np.allclose(direction_covariance, direction_covariance.T, rtol=1e-10, atol=1e-12*direction_scale):
            raise ValueError("scattering_vector_covariance must be symmetric")
        direction_covariance = 0.5*(direction_covariance+direction_covariance.T)
        supplied_direction_covariance = direction_covariance.copy()
        direction_eigenvalues = np.linalg.eigvalsh(direction_covariance)
        direction_tolerance = np.finfo(float).eps*max(direction_covariance.shape)*max(float(direction_eigenvalues[-1]), np.finfo(float).tiny)
        if direction_eigenvalues[0] < -direction_tolerance:
            raise ValueError("scattering_vector_covariance must be positive semidefinite")
        direction_covariance = tangent_projector @ direction_covariance @ tangent_projector.T
    rank = int(np.linalg.matrix_rank(directions))
    result = {
        "model": "three_dimensional_electron_flow_from_multi_view_thomson_projections",
        "projection_relation": "v_q = q_hat dot v_e",
        "scattering_vector_unit": directions.copy(),
        "design_rank": rank,
        "input_velocity_projection_m_s": projections.copy(),
        "input_projection_covariance_m2_s2": covariance.copy(),
        "geometry_uncertainty_included": direction_covariance is not None,
        "input_scattering_vector_covariance": supplied_direction_covariance,
        "tangent_plane_scattering_vector_covariance": direction_covariance,
        "geometry_covariance_method": (
            "iterative_first_order_tangent_plane_errors_in_variables_GLS"
            if direction_covariance is not None else "not_supplied"
        ),
    }
    if rank < 3:
        return {
            **result,
            "identifiability": "non_identifiable",
            "reportable_as_inferred_parameter": False,
            "electron_velocity_vector_m_s": None,
            "electron_velocity_covariance_m2_s2": None,
            "reason": "At least three linearly independent scattering-vector directions are required to identify all three velocity components.",
        }
    def flow_gls_solve(data_covariance):
        chol_local = np.linalg.cholesky(data_covariance)
        whitened_design_local = np.linalg.solve(chol_local, directions)
        whitened_data_local = np.linalg.solve(chol_local, projections)
        information_local = whitened_design_local.T @ whitened_design_local
        covariance_local = np.linalg.inv(information_local)
        velocity_local = covariance_local @ (whitened_design_local.T @ whitened_data_local)
        return velocity_local, covariance_local

    effective_covariance = covariance.copy()
    velocity, parameter_covariance = flow_gls_solve(effective_covariance)
    geometry_iterations = 0
    if direction_covariance is not None:
        for geometry_iterations in range(1, 51):
            jacobian_geometry = np.zeros((projections.size, 3*projections.size), dtype=float)
            for view, direction in enumerate(directions):
                tangent = np.eye(3)-np.outer(direction, direction)
                jacobian_geometry[view, 3*view:3*view+3] = tangent @ velocity
            effective_covariance = (
                covariance + jacobian_geometry @ direction_covariance @ jacobian_geometry.T
            )
            updated_velocity, updated_covariance = flow_gls_solve(effective_covariance)
            change = np.linalg.norm(updated_velocity-velocity) / max(
                np.linalg.norm(velocity), 1.0
            )
            velocity, parameter_covariance = updated_velocity, updated_covariance
            if change < 1e-10:
                break
        else:
            raise ValueError("3-D flow geometry-covariance GLS did not converge in 50 iterations")
    residual = projections-directions@velocity
    chi2 = float(residual @ np.linalg.solve(effective_covariance, residual))
    speed = float(np.linalg.norm(velocity))
    physical = bool(np.all(np.isfinite(velocity)) and speed < C)
    return {
        **result,
        "identifiability": "identifiable_from_three_dimensional_view_geometry",
        "reportable_as_inferred_parameter": physical,
        "electron_velocity_vector_m_s": velocity if physical else None,
        "electron_velocity_covariance_m2_s2": parameter_covariance if physical else None,
        "effective_projection_covariance_m2_s2": effective_covariance,
        "geometry_covariance_iterations": geometry_iterations,
        "electron_speed_m_s": speed if physical else None,
        "fit_residual_projection_m_s": residual,
        "chi_square": chi2,
        "degrees_of_freedom": int(projections.size-3),
        "reduced_chi_square": chi2/(projections.size-3) if projections.size > 3 else None,
        "physicality_status": "subluminal" if physical else "non_finite_or_superluminal_solution; vector withheld",
    }

def _apply_response(response, spectrum, wavelength_nm):
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(spectrum, dtype=float)
    r = np.asarray(response, dtype=float)
    if r.shape != (x.size, x.size) or y.shape != x.shape:
        raise ValueError("response must be square on the supplied wavelength grid")
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5*(x[:-1] + x[1:])
    edges[0] = x[0] - 0.5*(x[1]-x[0])
    edges[-1] = x[-1] + 0.5*(x[-1]-x[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength grid must define positive bin widths")
    return (r @ (y*widths))/widths


def relativistic_maxwell_juttner_spectrum(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad,
    electron_temperature_eV, *, flow_projection_m_s=0.0,
    incident_polarization, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
    require_full_spectrum=True,
):
    """Return a relativistic Maxwell-Juttner Thomson or Klein-Nishina spectrum.

    Set include_compton_recoil=True to evaluate the exact single-scattering
    Compton energy transfer and polarized Klein-Nishina differential cross-section.
    The default remains the elastic Thomson limit, guarded by the supplied
    maximum fractional recoil screen.

    Electron momenta follow an isotropic Maxwell-Juttner distribution in the
    plasma rest frame and are Lorentz-boosted along the scattering-vector
    direction by flow_projection_m_s. The scattered wavelength uses the exact
    elastic Thomson frequency ratio in the instantaneous electron rest frame:
    lambda_s/lambda_i=(1-n_s·beta)/(1-n_i·beta).

    The supplied incident polarization is Lorentz-transformed into each
    electron rest frame and the polarized Thomson differential cross-section
    is evaluated there, together with incident-flux and aberration Jacobians.
    By default recoil is screened but not included. With include_compton_recoil=True,
    single-scattering recoil and the polarized Klein-Nishina differential weight are included.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    angle = float(scattering_angle_rad)
    temperature = _positive(electron_temperature_eV, "electron_temperature_eV")
    flow = float(flow_projection_m_s)
    if (x.ndim != 1 or x.size < 7 or np.any(~np.isfinite(x))
            or np.any(np.diff(x) <= 0) or np.any(x <= 0)):
        raise ValueError("wavelength_nm must be a finite, positive, strictly increasing vector")
    if not np.isfinite(angle) or not 0 < angle <= np.pi:
        raise ValueError("scattering_angle_rad must lie in (0, pi]")
    if not np.isfinite(flow) or abs(flow) >= C:
        raise ValueError("flow_projection_m_s must be finite and strictly subluminal (|v| < c)")
    if int(radial_order) < 12 or int(angular_order) < 8:
        raise ValueError("radial_order >=12 and angular_order >=8 are required")
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    recoil_limit = _positive(maximum_recoil_correction_fraction, "maximum_recoil_correction_fraction")
    if recoil_limit >= 1.0:
        raise ValueError("maximum_recoil_correction_fraction must be below 1")

    theta = temperature/E_REST_EV
    beta_bulk = flow/C
    gamma_bulk = 1.0/np.sqrt(1.0-beta_bulk**2)
    sine, cosine = np.sin(angle/2.0), np.cos(angle/2.0)
    # Choose coordinates with q=k_s-k_i along +x. Then n_s-n_i=2 sin(theta/2) xhat.
    n_incident = np.array([-sine, cosine, 0.0])
    n_scattered = np.array([sine, cosine, 0.0])
    polarization = np.asarray(incident_polarization, dtype=float)
    if polarization.shape != (3,) or np.any(~np.isfinite(polarization)) or np.linalg.norm(polarization) == 0:
        raise ValueError("incident_polarization must be a finite non-zero 3-vector")
    if abs(float(np.dot(polarization/np.linalg.norm(polarization), n_incident))) > 1e-6:
        raise ValueError("incident_polarization must be transverse to incident_direction")
    polarization = polarization/np.linalg.norm(polarization)

    radial_nodes, radial_weights = leggauss(int(radial_order))
    umax = max(12.0*np.sqrt(theta)+12.0*theta, 1e-5)
    u = 0.5*umax*(radial_nodes+1.0)
    wu = 0.5*umax*radial_weights
    mu, wmu = leggauss(int(angular_order))
    nphi = max(16, 2*int(angular_order))
    phi = 2.0*np.pi*np.arange(nphi)/nphi
    U, MU_GRID, PHI = np.meshgrid(u, mu, phi, indexing="ij")
    GAMMA_REST = np.sqrt(1.0+U*U)
    beta_prime = U/GAMMA_REST
    beta_parallel = beta_prime*MU_GRID
    beta_perp = beta_prime*np.sqrt(np.maximum(1.0-MU_GRID*MU_GRID, 0.0))
    denominator = 1.0+beta_bulk*beta_parallel
    bx = (beta_parallel+beta_bulk)/denominator
    by = beta_perp*np.cos(PHI)/(gamma_bulk*denominator)
    bz = beta_perp*np.sin(PHI)/(gamma_bulk*denominator)
    beta = np.stack((bx, by, bz), axis=-1)
    beta_mag = np.sqrt(np.sum(beta*beta, axis=-1))
    gamma_lab = gamma_bulk*GAMMA_REST*denominator
    ni_dot = np.einsum("...j,j->...", beta, n_incident)
    ns_dot = np.einsum("...j,j->...", beta, n_scattered)
    lambda_scattered_thomson = laser*(1.0-ns_dot)/(1.0-ni_dot)

    beta_hat = np.divide(
        beta, beta_mag[..., None],
        out=np.zeros_like(beta), where=beta_mag[..., None] > 0,
    )
    def aberrated_direction(direction):
        mu_direction = np.einsum("...j,j->...", beta_hat, direction)
        denominator_direction = 1.0-beta_mag*mu_direction
        parallel = (mu_direction-beta_mag)/denominator_direction
        perpendicular = (
            direction-mu_direction[..., None]*beta_hat
        )/(gamma_lab[..., None]*denominator_direction[..., None])
        return parallel[..., None]*beta_hat+perpendicular

    ni_rest = aberrated_direction(n_incident)
    ns_rest = aberrated_direction(n_scattered)
    # Screen the Thomson/no-recoil approximation in each represented electron
    # rest frame. The Compton fractional correction is r/(1+r), where
    # r = (h nu' / m_e c^2) * (1 - cos(theta_scatter')).
    rest_angle_cosine = np.clip(
        np.einsum("...j,...j->...", ni_rest, ns_rest), -1.0, 1.0
    )
    incident_photon_energy_eV = 1239.8419843320026 / laser
    rest_photon_energy_eV = (
        incident_photon_energy_eV * gamma_lab * (1.0 - ni_dot)
    )
    recoil_parameter = (
        rest_photon_energy_eV / E_REST_EV * (1.0 - rest_angle_cosine)
    )
    recoil_ratio = 1.0/(1.0+recoil_parameter)
    recoil_correction_fraction = float(np.max(1.0-recoil_ratio))
    if (not np.isfinite(recoil_correction_fraction)
            or (not include_compton_recoil and recoil_correction_fraction > recoil_limit)):
        raise ValueError(
            "The Thomson/no-recoil approximation is outside the selected validity "
            f"limit: maximum represented Compton correction={recoil_correction_fraction:.6g}, "
            f"limit={recoil_limit:.6g}. Enable include_compton_recoil=True to use the "
            "single-scattering Klein-Nishina forward model."
        )
    lambda_scattered = lambda_scattered_thomson/recoil_ratio
    # Lorentz-transform the incident electric field to each electron rest frame.
    magnetic_direction_cross_e = np.cross(n_incident, polarization)
    e_rest = (
        gamma_lab[..., None]*(
            polarization+np.cross(beta, magnetic_direction_cross_e)
        )
        -(gamma_lab*gamma_lab/(gamma_lab+1.0))[..., None]
        *beta*np.einsum("...j,j->...", beta, polarization)[..., None]
    )
    e_rest -= np.einsum("...j,...j->...", e_rest, ni_rest)[..., None]*ni_rest
    e_rest_norm = np.linalg.norm(e_rest, axis=-1)
    e_rest = np.divide(
        e_rest, e_rest_norm[..., None],
        out=np.zeros_like(e_rest), where=e_rest_norm[..., None] > 0,
    )
    polarization_projection_squared = np.square(
        np.einsum("...j,...j->...", ns_rest, e_rest)
    )
    if include_compton_recoil:
        # Polarized Klein-Nishina cross-section, summed over final photon
        # polarization states. r=epsilon_s'/epsilon_i'; in the r->1 limit
        # this reduces exactly to the polarized Thomson factor.
        polarization_factor = 0.5*np.square(recoil_ratio)*(
            1.0/recoil_ratio + recoil_ratio
            - 2.0*polarization_projection_squared
        )
    else:
        polarization_factor = 1.0-polarization_projection_squared
    polarization_factor = np.maximum(polarization_factor, 0.0)
    differential_weight = (
        (1.0-ni_dot)*polarization_factor
        /(gamma_lab*gamma_lab*np.square(1.0-ns_dot))
    )
    radial_weight = (
        wu[:, None, None]*u[:, None, None]**2
        * np.exp(-(GAMMA_REST-1.0)/theta)
    )
    distribution_quadrature_weight = (
        radial_weight*wmu[None, :, None]*(2.0*np.pi/nphi)
    )
    distribution_normalization = float(np.sum(distribution_quadrature_weight))
    if not np.isfinite(distribution_normalization) or distribution_normalization <= 0.0:
        raise ValueError("Maxwell-Juttner momentum quadrature has a non-positive normalization")
    # Exact isotropic Maxwell-Juttner momentum-space normalization:
    # integral_0^inf u^2 exp[-(sqrt(1+u^2)-1)/theta] du
    #   = theta exp(1/theta) K_2(1/theta)
    #   = theta Kve_2(1/theta).
    # The angular integral contributes 4*pi.  Comparing the actual finite
    # quadrature against this analytic value detects a truncated high-energy
    # tail instead of silently renormalizing that missing population away.
    exact_distribution_normalization = (
        4.0*np.pi*theta*float(kve(2, 1.0/theta))
    )
    momentum_quadrature_to_analytic_ratio = (
        distribution_normalization/exact_distribution_normalization
    )
    if (not np.isfinite(momentum_quadrature_to_analytic_ratio)
            or momentum_quadrature_to_analytic_ratio <= 0.0
            or momentum_quadrature_to_analytic_ratio > 1.0 + 1e-4):
        raise ValueError(
            "Maxwell-Juttner momentum quadrature is inconsistent with the "
            f"analytic normalization (captured={momentum_quadrature_to_analytic_ratio:.8g}); "
            "increase radial_order or revise the momentum quadrature."
        )
    momentum_probability_coverage = min(1.0, momentum_quadrature_to_analytic_ratio)
    if momentum_probability_coverage < 0.999:
        raise ValueError(
            "Maxwell-Juttner momentum quadrature captures only "
            f"{momentum_probability_coverage:.6%} of the analytic momentum "
            "probability; increase radial_order or the momentum cutoff. "
            "The missing tail is not silently renormalized."
        )
    quadrature_weight = distribution_quadrature_weight * differential_weight

    edges = np.empty(x.size+1, dtype=float)
    edges[1:-1] = 0.5*(x[:-1]+x[1:])
    edges[0] = x[0]-0.5*(x[1]-x[0])
    edges[-1] = x[-1]+0.5*(x[-1]-x[-2])
    # Deposit quadrature weight with a linear (cloud-in-cell) spectral basis.
    # A hard histogram is piecewise constant in flow/temperature parameters and
    # gives least-squares optimizers a zero or discontinuous numerical Jacobian.
    # Linear deposition is continuous, conserves probability and represents the
    # finite wavelength-bin measurement without introducing a fitted line shape.
    centers = 0.5 * (edges[:-1] + edges[1:])
    sampled_wavelength = lambda_scattered.ravel()
    sampled_weight = quadrature_weight.ravel()
    total_weight = float(np.sum(sampled_weight))
    if not np.isfinite(total_weight) or total_weight <= 0.0:
        raise ValueError("Relativistic Thomson quadrature has a non-positive integrated scattering weight")
    effective_differential_cross_section = (
        ELECTRON_CLASSICAL_RADIUS_M**2 * total_weight / distribution_normalization
    )
    inside = (
        (sampled_wavelength >= edges[0])
        & (sampled_wavelength <= edges[-1])
        & np.isfinite(sampled_weight)
        & (sampled_weight >= 0.0)
    )
    captured = float(np.sum(sampled_weight[inside])) / total_weight if total_weight > 0 else 0.0
    counts = np.zeros(x.size, dtype=float)
    if np.any(inside):
        lam = sampled_wavelength[inside]
        weight = sampled_weight[inside]
        right = np.searchsorted(centers, lam, side="right")
        below = right == 0
        above = right == centers.size
        right = np.clip(right, 1, centers.size - 1)
        left = right - 1
        fraction = (lam - centers[left]) / (centers[right] - centers[left])
        left[below] = 0
        right[below] = 0
        fraction[below] = 0.0
        left[above] = centers.size - 1
        right[above] = centers.size - 1
        fraction[above] = 0.0
        np.add.at(counts, left, weight * (1.0 - fraction))
        np.add.at(counts, right, weight * fraction)
    if total_weight <= 0 or (require_full_spectrum and captured < 0.995):
        raise ValueError(
            "wavelength_nm must capture at least 99.5% of the relativistic Thomson spectrum; "
            f"captured fraction was {captured:.6g}"
        )
    widths = np.diff(edges)
    density = counts/widths
    area = float(np.sum(counts))
    if area <= 0:
        raise ValueError("relativistic Thomson quadrature produced zero spectral power")
    density /= area
    centers = 0.5*(edges[:-1]+edges[1:])
    centroid = float(np.sum(centers*density*widths))
    variance = float(np.sum((centers-centroid)**2*density*widths))
    return {
        "spectral_density_per_nm": density,
        "wavelength_bin_widths_nm": widths,
        "integrated_probability": 1.0,
        "momentum_probability_coverage": float(momentum_probability_coverage),
        "analytic_momentum_normalization": float(exact_distribution_normalization),
        "quadrature_momentum_normalization": float(distribution_normalization),
        "quadrature_to_analytic_normalization_ratio": float(momentum_quadrature_to_analytic_ratio),
        "momentum_tail_renormalization_performed": False,
        "effective_differential_cross_section_m2_sr": float(effective_differential_cross_section),
        "fitted_centroid_nm": centroid,
        "spectral_rms_width_nm": float(np.sqrt(max(variance, 0.0))),
        "captured_probability": captured,
        "relativistic_kinematics": True,
        "frequency_ratio": (
            "lambda_s/lambda_i=[(1-n_s dot beta)/(1-n_i dot beta)]/r_KN"
            if include_compton_recoil else
            "lambda_s/lambda_i=(1-n_s dot beta)/(1-n_i dot beta)"
        ),
        "distribution": "Lorentz-boosted isotropic Maxwell-Juttner",
        "recoil_included": bool(include_compton_recoil),
        "scattering_cross_section_model": (
            "polarized_single_scattering_Klein_Nishina_instantaneous_electron_rest_frame"
            if include_compton_recoil else "polarized_Thomson_instantaneous_electron_rest_frame"
        ),
        "compton_energy_ratio_epsilon_scattered_over_incident_rest_frame": recoil_ratio,
        "maximum_recoil_correction_fraction_on_quadrature_support": recoil_correction_fraction,
        "thomson_recoil_correction_fraction_max_on_quadrature_support": recoil_correction_fraction,
        "thomson_recoil_correction_fraction_limit": float(recoil_limit),
        "thomson_no_recoil_validity_status": (
            "Compton_recoil_included" if include_compton_recoil else
            "within_user_selected_fractional_correction_limit"
        ),
        "polarization_model": "supplied incident polarization Lorentz-transformed to the electron rest frame",
        "quadrature": {"radial_order": int(radial_order), "angular_order": int(angular_order), "gyrophase_points": nphi},
    }


def fit_relativistic_maxwell_juttner(
    wavelength_nm, signal, laser_wavelength_nm, scattering_angle_rad,
    response, uncertainty=None, *, incident_direction, scattered_direction,
    incident_polarization, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
):
    """Fit relativistic Maxwell-Juttner Te and q-parallel flow to one spectrum."""

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or x.size < 12:
        raise ValueError("wavelength_nm and signal must be matching vectors with >=12 samples")
    ni = np.array(incident_direction, dtype=float, copy=True)
    ns = np.array(scattered_direction, dtype=float, copy=True)
    if ni.shape != (3,) or ns.shape != (3,) or np.any(~np.isfinite(ni)) or np.any(~np.isfinite(ns)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(ni) == 0 or np.linalg.norm(ns) == 0:
        raise ValueError("incident_direction and scattered_direction must be non-zero")
    ni /= np.linalg.norm(ni)
    ns /= np.linalg.norm(ns)
    angle_from_vectors = float(np.arccos(np.clip(np.dot(ni, ns), -1.0, 1.0)))
    if not np.isclose(angle_from_vectors, scattering_angle_rad, rtol=0.0, atol=1e-5):
        raise ValueError("scattering_angle_rad is inconsistent with incident_direction and scattered_direction")
    polarization_lab = np.asarray(incident_polarization, dtype=float)
    if (polarization_lab.shape != (3,) or np.any(~np.isfinite(polarization_lab))
            or np.linalg.norm(polarization_lab) == 0):
        raise ValueError("incident_polarization must be a finite non-zero 3-vector")
    polarization_lab = polarization_lab/np.linalg.norm(polarization_lab)
    if abs(float(np.dot(polarization_lab, ni))) > 1e-6:
        raise ValueError("incident_polarization must be transverse to incident_direction")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    e3 = np.cross(ni, ns)
    if np.linalg.norm(e3) <= 1e-12:
        e3 = polarization_lab-np.dot(polarization_lab,qhat)*qhat
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(qhat,e3)
    e2 /= np.linalg.norm(e2)
    local_to_lab = np.column_stack((qhat,e2,e3))
    polarization_local = local_to_lab.T@polarization_lab
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y/scale
    unc = None
    covariance = None
    covariance_cholesky = None
    if uncertainty is not None:
        supplied_uncertainty = np.asarray(uncertainty, dtype=float)
        if supplied_uncertainty.shape == y.shape:
            unc = supplied_uncertainty/scale
            if np.any(~np.isfinite(unc)) or np.any(unc <= 0):
                raise ValueError("uncertainty must be positive and match signal")
        elif supplied_uncertainty.shape == (y.size, y.size):
            covariance = supplied_uncertainty/(scale*scale)
            if np.any(~np.isfinite(covariance)):
                raise ValueError("spectral_covariance must contain only finite values")
            covariance_scale = float(np.max(np.abs(covariance)))
            if covariance_scale <= 0 or not np.allclose(
                covariance, covariance.T, rtol=1e-9, atol=1e-12*covariance_scale
            ):
                raise ValueError("spectral_covariance must be symmetric and non-zero")
            try:
                covariance_cholesky = np.linalg.cholesky(covariance)
            except np.linalg.LinAlgError as exc:
                raise ValueError("spectral_covariance must be positive definite") from exc
        else:
            raise ValueError("uncertainty must be a positive pointwise vector or a full n_wavelength x n_wavelength covariance matrix")
    residual_scale = (
        np.full_like(ys, max(float(np.max(np.abs(ys))), np.finfo(float).tiny))
        if unc is None and covariance_cholesky is None else unc
    )
    def whiten(residual):
        if covariance_cholesky is not None:
            return np.linalg.solve(covariance_cholesky, residual)
        return residual/residual_scale
    edge = max(2, x.size//10)
    baseline = float(np.median(np.r_[ys[:edge], ys[-edge:]]))
    positive = np.maximum(ys-baseline, 0.0)
    if np.sum(positive) <= 0:
        raise ValueError("signal must contain a positive Thomson-scattering feature")
    centroid0 = float(np.sum(x*positive)/np.sum(positive))
    sigma0 = float(np.sqrt(np.sum((x-centroid0)**2*positive)/np.sum(positive)))
    sigma0 = max(sigma0, float(np.median(np.diff(x))))
    sine = np.sin(float(scattering_angle_rad)/2.0)
    te0 = E_REST_EV*(sigma0/(2.0*float(laser_wavelength_nm)*sine))**2
    if not np.isfinite(te0) or te0 <= 0:
        raise ValueError("The measured spectral width does not provide a positive relativistic temperature initial estimate")
    beta0 = -(centroid0-float(laser_wavelength_nm))/(2.0*float(laser_wavelength_nm)*sine)
    if not np.isfinite(beta0) or abs(beta0) >= 1.0:
        raise ValueError("The measured spectral centroid implies a non-physical initial flow projection |v| >= c")
    amp0 = max(float(np.max(positive)), np.finfo(float).tiny)
    p0 = np.array([np.log(te0), np.arctanh(beta0), np.log(amp0), baseline, 0.0])

    def evaluate(parameters, return_parts=False, *, require_full_spectrum=True):
        te = float(np.exp(np.clip(parameters[0], -700.0, 700.0)))
        beta_flow = float(np.tanh(parameters[1]))
        physical = relativistic_maxwell_juttner_spectrum(
            x, laser_wavelength_nm, scattering_angle_rad, te,
            flow_projection_m_s=beta_flow*C,
            incident_polarization=polarization_local,
            radial_order=radial_order, angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
            require_full_spectrum=require_full_spectrum,
        )
        shape = _apply_response(response, physical["spectral_density_per_nm"], x)
        amplitude = float(np.exp(np.clip(parameters[2], -700.0, 700.0)))
        offset, slope = float(parameters[3]), float(parameters[4])
        predicted = amplitude*shape+offset+slope*(x-float(laser_wavelength_nm))
        return (predicted, physical, shape) if return_parts else predicted

    def objective(parameters):
        try:
            return whiten(evaluate(parameters, require_full_spectrum=False)-ys)
        except ValueError as exc:
            message = str(exc)
            if "capture at least 99.5%" in message or "produced zero spectral power" in message:
                # A trial step can move the spectrum outside the measured window.
                # Penalize that trial; retain the strict full-window requirement
                # for the final reported physical fit.
                return np.full_like(ys, 1.0e6)
            raise

    # Keep the atanh(beta) coordinate away from floating-point saturation: tanh(±20)
    # rounds to exactly ±1 and would create a spurious |v|=c trial during optimization.
    fit_bounds = (
        np.array([-np.inf, -18.0, -np.inf, -np.inf, -np.inf]),
        np.array([np.inf, 18.0, np.inf, np.inf, np.inf]),
    )
    fit = least_squares(
        objective, p0, x_scale="jac", max_nfev=1200, bounds=fit_bounds,
    )
    # A broad baseline slope can partially trade against a small Doppler shift.
    # If refinement reverses the centroid-implied flow sign, retry from a
    # smaller centroid-guided shift and from zero flow, retaining the best fit.
    # These are numerical starts only, not priors or physical flow constraints.
    if abs(beta0) > 1e-8 and np.sign(np.tanh(fit.x[1])) != np.sign(beta0):
        for alternate_beta in (0.5 * beta0, 0.0):
            alternate_start = p0.copy()
            alternate_start[1] = np.arctanh(alternate_beta)
            alternate_fit = least_squares(
                objective, alternate_start, x_scale="jac", max_nfev=1200,
                bounds=fit_bounds,
            )
            if alternate_fit.success and (
                not fit.success or alternate_fit.cost < fit.cost
            ):
                fit = alternate_fit
    if not fit.success:
        raise ValueError(f"relativistic Maxwell-Juttner Thomson fit failed: {fit.message}")
    jacobian = np.asarray(fit.jac, dtype=float)
    column_norms = np.linalg.norm(jacobian, axis=0)
    scaled_jacobian = np.divide(
        jacobian, column_norms[None, :],
        out=np.zeros_like(jacobian),
        where=column_norms[None, :] > np.finfo(float).tiny,
    )
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    rank_tolerance = (
        np.finfo(float).eps * max(scaled_jacobian.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0 else float("inf")
    )
    if rank < 5 or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            "relativistic Maxwell-Juttner Thomson fit is not jointly identifiable "
            f"(scaled Jacobian rank {rank}/5, condition number {condition:.6g})"
        )
    predicted_scaled, physical, shape = evaluate(
        fit.x, return_parts=True, require_full_spectrum=True
    )
    dof = max(1, x.size-5)
    covariance_fit = np.linalg.pinv(fit.jac.T@fit.jac)
    if unc is None and covariance_cholesky is None:
        covariance_fit *= float(2.0*fit.cost/dof)
    te = float(np.exp(np.clip(fit.x[0], -700.0, 700.0)))
    beta_flow = float(np.tanh(fit.x[1]))
    amplitude_scaled = float(np.exp(np.clip(fit.x[2], -700.0, 700.0)))
    transform = np.diag([te, C*(1.0-beta_flow*beta_flow), amplitude_scaled*scale, scale, scale])
    covariance_physical = transform@covariance_fit@transform.T
    def derived_spectral_quantities(parameters):
        _, sampled_physical, sampled_shape = evaluate(
            parameters, return_parts=True, require_full_spectrum=False
        )
        sampled_window_area = (
            float(np.exp(np.clip(parameters[2], -700.0, 700.0)))
            * scale * float(np.sum(
                sampled_shape * sampled_physical["wavelength_bin_widths_nm"]
            ))
        )
        captured_probability = float(sampled_physical["captured_probability"])
        if not np.isfinite(captured_probability) or captured_probability <= 0.0:
            raise ValueError("The fitted relativistic spectrum has no finite captured probability")
        total_area = sampled_window_area / captured_probability
        return (
            float(sampled_physical["effective_differential_cross_section_m2_sr"]),
            captured_probability,
            total_area,
            sampled_window_area,
        )
    cross_section_gradient = np.zeros(5, dtype=float)
    area_gradient = np.zeros(5, dtype=float)
    captured_gradient = np.zeros(5, dtype=float)
    for parameter_index in range(3):
        step = np.cbrt(np.finfo(float).eps) * max(abs(float(fit.x[parameter_index])), 1.0)
        plus = np.asarray(fit.x, dtype=float).copy()
        minus = np.asarray(fit.x, dtype=float).copy()
        plus[parameter_index] += step
        minus[parameter_index] -= step
        cs_plus, capture_plus, area_plus, _ = derived_spectral_quantities(plus)
        cs_minus, capture_minus, area_minus, _ = derived_spectral_quantities(minus)
        cross_section_gradient[parameter_index] = (cs_plus-cs_minus)/(2.0*step)
        captured_gradient[parameter_index] = (capture_plus-capture_minus)/(2.0*step)
        area_gradient[parameter_index] = (area_plus-area_minus)/(2.0*step)
    cross_section, captured_probability, area, window_area = derived_spectral_quantities(fit.x)
    cross_section_variance = float(
        cross_section_gradient @ covariance_fit @ cross_section_gradient
    )
    cross_section_uncertainty = float(np.sqrt(max(cross_section_variance, 0.0)))
    area_variance = float(area_gradient @ covariance_fit @ area_gradient)
    area_sigma = float(np.sqrt(max(area_variance, 0.0)))
    captured_variance = float(captured_gradient @ covariance_fit @ captured_gradient)
    captured_uncertainty = float(np.sqrt(max(captured_variance, 0.0)))
    power_cross_section_covariance = float(
        area_gradient @ covariance_fit @ cross_section_gradient
    )
    power_cross_section_relative_covariance = (
        power_cross_section_covariance/(area*cross_section)
        if area > 0.0 and cross_section > 0.0 else 0.0
    )
    amplitude = amplitude_scaled*scale
    offset = float(fit.x[3]*scale)
    slope = float(fit.x[4]*scale)
    fitted = predicted_scaled*scale
    residual = y-fitted
    return {
        "electron_temperature_eV": te,
        "electron_temperature_uncertainty_eV": float(np.sqrt(max(covariance_physical[0, 0], 0.0))),
        "electron_velocity_projection_m_s": beta_flow*C,
        "electron_velocity_projection_uncertainty_m_s": float(np.sqrt(max(covariance_physical[1, 1], 0.0))),
        "fitted_center_nm": physical["fitted_centroid_nm"],
        "intrinsic_sigma_nm": physical["spectral_rms_width_nm"],
        "amplitude": amplitude,
        "background_intercept": offset,
        "background_slope_per_nm": slope,
        "fit_covariance": covariance_physical,
        "spectral_weighting_status": (
            "full_within_channel_generalized_least_squares"
            if covariance_cholesky is not None else
            "diagonal_pointwise_measurement_uncertainty_weighted"
            if unc is not None else
            "unweighted_least_squares_no_measurement_uncertainty_supplied"
        ),
        "fit_covariance_interpretation": (
            "local_parameter_covariance_from_supplied_full_spectral_covariance"
            if covariance_cholesky is not None else
            "local_parameter_covariance_from_supplied_pointwise_measurement_uncertainty"
            if unc is not None else
            "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"
        ),
        "fit_parameters": np.array([te, beta_flow*C, amplitude, offset, slope], dtype=float),
        "fit_parameter_names": ["electron_temperature_eV", "electron_velocity_projection_m_s", "amplitude", "background_intercept", "background_slope_per_nm"],
        "integrated_scattered_power_W": area,
        "integrated_scattered_power_uncertainty_W": area_sigma,
        "integrated_scattered_power_fit_uncertainty_W": area_sigma,
        "effective_differential_cross_section_m2_sr": cross_section,
        "scattering_cross_section_model": physical["scattering_cross_section_model"],
        "maximum_recoil_correction_fraction": float(physical["maximum_recoil_correction_fraction_on_quadrature_support"]),
        "effective_differential_cross_section_uncertainty_m2_sr": cross_section_uncertainty,
        "power_cross_section_relative_covariance": power_cross_section_relative_covariance,
        "captured_spectral_probability_uncertainty": captured_uncertainty,
        "integrated_scattered_power_window_W": window_area,
        "effective_cross_section_uncertainty_method": "finite-difference propagation of fitted Te/flow covariance through the Lorentz-transformed polarized differential cross-section (Thomson or Klein-Nishina, as selected)",
        "reduced_chi_square": (
            float(2.0*fit.cost/dof)
            if covariance_cholesky is not None or unc is not None else None
        ),
        "fitted_spectrum": fitted,
        "residual": residual,
        "optimizer_success": bool(fit.success),
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "model": "relativistic non-collective Thomson scattering with Lorentz-boosted Maxwell-Juttner electrons",
        "relativistic_kinematics": True,
        "recoil_included": bool(physical["recoil_included"]),
        "thomson_recoil_correction_fraction_max_on_quadrature_support": physical[
            "thomson_recoil_correction_fraction_max_on_quadrature_support"
        ],
        "thomson_recoil_correction_fraction_limit": physical["thomson_recoil_correction_fraction_limit"],
        "thomson_no_recoil_validity_status": physical["thomson_no_recoil_validity_status"],
        "quadrature_diagnostics": physical["quadrature"],
        "momentum_probability_coverage": physical["momentum_probability_coverage"],
        "quadrature_to_analytic_normalization_ratio": physical["quadrature_to_analytic_normalization_ratio"],
        "momentum_tail_renormalization_performed": physical["momentum_tail_renormalization_performed"],
        "captured_spectral_probability": captured_probability,
        "assumptions": [
            "Single-scattering non-collective Thomson regime.",
            "Isotropic Maxwell-Juttner electron distribution in its rest frame.",
            "Bulk electron flow is parallel to the scattering vector.",
            "The incident polarization is explicitly supplied and Lorentz-transformed into each electron rest frame.",
            "Exact relativistic photon-frequency ratio and Lorentz aberration are used.",
            ("Single-scattering Compton recoil and polarized Klein-Nishina weighting are included." if physical["recoil_included"] else "Photon recoil/Compton corrections are neglected; the elastic Thomson limit is assumed."),
            "Measured instrument response is applied to the calculated spectral density."
        ],
    }


def _kappa_pdf_and_derivatives(velocity, standard_deviation, kappa):
    nu = 2.0*kappa-1.0
    if nu <= 2.0:
        raise ValueError("kappa must be > 1.5 for finite variance")
    scale = standard_deviation*np.sqrt((nu-2.0)/nu)
    log_norm = (
        gammaln((nu+1.0)/2.0)-gammaln(nu/2.0)
        -0.5*np.log(nu*np.pi)-np.log(scale)
    )
    denominator = nu*scale*scale+velocity*velocity
    pdf = np.exp(log_norm)*(1.0+velocity*velocity/(nu*scale*scale))**(-(nu+1.0)/2.0)
    first = -(nu+1.0)*velocity/denominator*pdf
    second = (-(nu+1.0)*nu*scale*scale+(nu+1.0)*(nu+2.0)*velocity*velocity)/(denominator*denominator)*pdf
    return pdf, first, second


def _maxwellian_pdf_and_derivatives(velocity, standard_deviation):
    pdf = np.exp(-0.5*(velocity/standard_deviation)**2)/(np.sqrt(2.0*np.pi)*standard_deviation)
    first = -velocity/(standard_deviation**2)*pdf
    second = ((velocity**2)/(standard_deviation**4)-1.0/(standard_deviation**2))*pdf
    return pdf, first, second


def _landau_susceptibility(phase_velocity, plasma_frequency_squared, wavenumber,
                           standard_deviation, distribution="maxwellian", kappa=None):
    """Longitudinal susceptibility with the Landau contour on the full velocity axis.

    The real principal value is evaluated symmetrically:
      PV int f'(v)/(v-v_phi) dv
        = int_0^inf [f'(v_phi+x)-f'(v_phi-x)]/x dx.
    This removes the pole analytically and avoids truncating algebraic Kappa
    tails at an arbitrary number of thermal speeds. The retarded contour adds
    +i*pi*f'(v_phi), consistent with exp(-i omega t).
    """
    if distribution == "kappa":
        pdf_fn = lambda v: _kappa_pdf_and_derivatives(v, standard_deviation, kappa)
    elif distribution == "maxwellian":
        pdf_fn = lambda v: _maxwellian_pdf_and_derivatives(v, standard_deviation)
    else:
        raise ValueError("distribution must be 'maxwellian' or 'kappa'")
    sigma = _positive(standard_deviation, "standard_deviation")
    vphi = float(phase_velocity)
    if not np.isfinite(vphi):
        raise ValueError("phase_velocity must be finite")
    omega_p2 = _positive(plasma_frequency_squared, "plasma_frequency_squared")
    k_wave = _positive(wavenumber, "wavenumber")

    # Map x in [0, infinity) to t in [0, 1): x=sigma*t/(1-t).
    # Gauss-Legendre quadrature resolves the resonant neighbourhood and the
    # slowly decaying algebraic tails without a finite velocity cutoff.
    # Kappa tails become exceptionally broad as kappa approaches 3/2.
    # Increase quadrature order in that near-boundary regime; otherwise a
    # finite-node rule can miss the remote algebraic tail and even corrupt the
    # sign of the principal-value response.
    if distribution == "kappa" and float(kappa) < 1.5002:
        quadrature_order = 1536
    elif distribution == "kappa" and float(kappa) < 1.501:
        quadrature_order = 768
    else:
        quadrature_order = 384
    nodes, weights = _FULL_AXIS_LANDAU_QUADRATURE[quadrature_order]
    t = 0.5*(nodes+1.0)
    wt = 0.5*weights
    one_minus_t = 1.0-t
    offset = sigma*t/one_minus_t
    jacobian = sigma/(one_minus_t*one_minus_t)
    _, derivative_plus, _ = pdf_fn(vphi+offset)
    _, derivative_minus, _ = pdf_fn(vphi-offset)
    principal = float(np.sum(
        wt*jacobian*(derivative_plus-derivative_minus)/offset
    ))
    _, derivative_at, _ = pdf_fn(np.asarray(vphi))
    landau_integral = principal+1j*np.pi*float(derivative_at)
    return -(omega_p2/(k_wave*k_wave))*landau_integral


def _resolve_ion_species(ion_species):
    result = []
    for index, item in enumerate(ion_species):
        if isinstance(item, Mapping):
            name = str(item.get("name", f"ion_{index}"))
            charge_state = float(item["charge_state"])
            if "mass_kg" in item:
                mass = _positive(item["mass_kg"], f"ion_species[{index}].mass_kg")
            else:
                mass = _positive(item["mass_amu"], f"ion_species[{index}].mass_amu")*AMU
        else:
            try:
                from plasmapy.particles import Particle
            except ImportError as exc:
                raise ImportError("String ion species require the optional PlasmaPy particle package; alternatively provide explicit name, charge_state and mass_amu/mass_kg mappings.") from exc
            particle = item if hasattr(item, "mass") and hasattr(item, "charge_number") else Particle(item)
            name = str(particle)
            mass = float(particle.mass.to_value("kg"))
            charge_state = float(particle.charge_number)
        if not np.isfinite(charge_state) or charge_state <= 0:
            raise ValueError(f"ion_species[{index}] must have a positive charge state")
        result.append({"name": name, "mass_kg": mass, "charge_state": charge_state})
    return result


def collective_kappa_spectral_density(
    wavelength_nm, laser_wavelength_nm, scattering_angle_rad, *,
    electron_density_m3, electron_temperature_eV, kappa,
    ion_species, ion_fractions, ion_temperature_eV,
    electron_velocity_m_s, ion_velocity_m_s,
    incident_direction, scattered_direction, electron_populations=None,
    ion_populations=None,
):
    """Unmagnetized collective spectrum with Maxwellian/Kappa electron and ion mixtures.

    electron_populations optionally supplies independently drifting electron
    populations with density_fraction (summing to one), temperature_ratio,
    distribution ('kappa' or 'maxwellian'), Kappa index when selected, and
    flow_projection_m_s. ion_populations optionally supplies one mapping per
    ion species with distribution, Kappa index when selected, and projected
    flow; absent ion_populations preserves the Maxwellian-ion default.

    Every species contributes its own Landau-contour susceptibility. The
    species responses are summed before the common dielectric screening and
    the electron/ion fluctuation source terms are assembled. This is a kinetic
    mixture, not a sum of separately fitted spectra. All Kappa populations
    require finite variance (kappa > 3/2); mixture fractions and flows are
    validated rather than clipped or silently renormalized.

    This is a collisionless, unmagnetized, electrostatic model. It does not
    imply collisional, relativistic, magnetized or Full-EM Kappa-ion physics.
    """
    x = np.asarray(wavelength_nm, dtype=float)
    ne = _positive(electron_density_m3, "electron_density_m3")
    te = _positive(electron_temperature_eV, "electron_temperature_eV")
    kap = float(kappa)
    if not np.isfinite(kap) or kap <= 1.5:
        raise ValueError("kappa must be finite and > 1.5")
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    angle = float(scattering_angle_rad)
    if x.ndim != 1 or x.size < 7 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be finite and strictly increasing")
    if not 0 < angle <= np.pi:
        raise ValueError("scattering_angle_rad must lie in (0, pi]")
    ni = np.asarray(incident_direction, dtype=float)
    ns = np.asarray(scattered_direction, dtype=float)
    if ni.shape != (3,) or ns.shape != (3,) or np.any(~np.isfinite(ni)) or np.any(~np.isfinite(ns)):
        raise ValueError("incident_direction and scattered_direction must be finite 3-vectors")
    if np.linalg.norm(ni) == 0 or np.linalg.norm(ns) == 0:
        raise ValueError("incident_direction and scattered_direction must be non-zero")
    ni = ni/np.linalg.norm(ni)
    ns = ns/np.linalg.norm(ns)
    if not np.isclose(np.arccos(np.clip(np.dot(ni, ns), -1.0, 1.0)), angle, atol=1e-5, rtol=0.0):
        raise ValueError("scattering_angle_rad is inconsistent with incident/scattered directions")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    ve = np.asarray(electron_velocity_m_s, dtype=float)
    vi = np.asarray(ion_velocity_m_s, dtype=float)
    if ve.shape != (3,) or np.any(~np.isfinite(ve)) or np.linalg.norm(ve) >= C:
        raise ValueError("electron_velocity_m_s must be a finite subluminal 3-vector")
    species = _resolve_ion_species(ion_species)
    fractions = np.asarray(ion_fractions, dtype=float)
    temperatures = np.asarray(np.atleast_1d(ion_temperature_eV), dtype=float)
    if (fractions.shape != (len(species),) or temperatures.shape != (len(species),)
            or np.any(~np.isfinite(fractions)) or np.any(fractions <= 0)
            or not np.isclose(np.sum(fractions), 1.0, rtol=1e-6, atol=1e-8)
            or np.any(~np.isfinite(temperatures)) or np.any(temperatures <= 0)):
        raise ValueError("ion fractions and temperatures must provide one positive value per species; fractions must sum to one")
    if vi.shape == (3,) and len(species) == 1:
        vi = vi.reshape(1, 3)
    if vi.shape != (len(species), 3) or np.any(~np.isfinite(vi)) or np.any(np.linalg.norm(vi, axis=1) >= C):
        raise ValueError("ion_velocity_m_s must contain one finite subluminal 3-vector per species")
    ue = float(np.dot(ve, qhat))
    ui = vi@qhat
    k = 4.0*np.pi*np.sin(angle/2.0)/(laser*1e-9)
    vphase = -C*(x-laser)/(2.0*laser*np.sin(angle/2.0))
    sigma_e = np.sqrt(te*E_CHARGE/ME)
    omega_pe2 = ne*E_CHARGE**2/(EPS0*ME)
    if electron_populations is None:
        populations = [{"density_fraction": 1.0, "temperature_ratio": 1.0,
                        "kappa": kap, "distribution": "kappa",
                        "flow_projection_m_s": ue}]
    else:
        if not isinstance(electron_populations, Sequence) or isinstance(electron_populations, (str, bytes)) or len(electron_populations) < 1:
            raise ValueError("electron_populations must be a non-empty sequence of population mappings")
        populations = []
        for index, population in enumerate(electron_populations):
            if not isinstance(population, Mapping):
                raise ValueError(f"electron_populations[{index}] must be a mapping")
            fraction = _positive(population.get("density_fraction"), f"electron_populations[{index}].density_fraction")
            ratio = _positive(population.get("temperature_ratio", 1.0), f"electron_populations[{index}].temperature_ratio")
            distribution = str(population.get("distribution", "kappa")).lower()
            population_kappa = float(population.get("kappa", kap))
            flow = float(population.get("flow_projection_m_s", ue))
            if distribution not in ("kappa", "maxwellian"):
                raise ValueError(f"electron_populations[{index}].distribution must be 'kappa' or 'maxwellian'")
            if distribution == "kappa" and (not np.isfinite(population_kappa) or population_kappa <= 1.5):
                raise ValueError(f"electron_populations[{index}].kappa must be finite and > 1.5")
            if not np.isfinite(flow) or abs(flow) >= C:
                raise ValueError(f"electron_populations[{index}].flow_projection_m_s must be finite and subluminal")
            populations.append({"density_fraction": fraction, "temperature_ratio": ratio,
                                "kappa": population_kappa, "distribution": distribution,
                                "flow_projection_m_s": flow})
        if not np.isclose(sum(item["density_fraction"] for item in populations), 1.0, rtol=1e-6, atol=1e-8):
            raise ValueError("electron population density_fraction values must sum to one")
    electron_pdf = np.zeros(x.size, dtype=float)
    chi_e = np.zeros(x.size, dtype=complex)
    population_pdfs, population_susceptibilities = [], []
    for population in populations:
        fraction = population["density_fraction"]
        sigma_population = sigma_e*np.sqrt(population["temperature_ratio"])
        shifted_phase = vphase-population["flow_projection_m_s"]
        if population["distribution"] == "kappa":
            pdf_population, _, _ = _kappa_pdf_and_derivatives(
                shifted_phase, sigma_population, population["kappa"])
        else:
            pdf_population, _, _ = _maxwellian_pdf_and_derivatives(
                shifted_phase, sigma_population)
        chi_population = np.empty(x.size, dtype=complex)
        for j in range(x.size):
            chi_population[j] = _landau_susceptibility(
                shifted_phase[j], fraction*omega_pe2, k, sigma_population,
                distribution=population["distribution"],
                kappa=population["kappa"] if population["distribution"] == "kappa" else None)
        electron_pdf += fraction*pdf_population
        chi_e += chi_population
        population_pdfs.append(pdf_population)
        population_susceptibilities.append(chi_population)
    if ion_populations is None:
        resolved_ion_populations = [
            {"distribution": "maxwellian", "kappa": None,
             "flow_projection_m_s": float(ui[index])}
            for index in range(len(species))
        ]
    else:
        if (not isinstance(ion_populations, Sequence)
                or isinstance(ion_populations, (str, bytes))
                or len(ion_populations) != len(species)):
            raise ValueError("ion_populations must provide one distribution mapping per ion species")
        resolved_ion_populations = []
        for index, population in enumerate(ion_populations):
            if not isinstance(population, Mapping):
                raise ValueError(f"ion_populations[{index}] must be a mapping")
            distribution = str(population.get("distribution", "maxwellian")).lower()
            kappa_i = population.get("kappa")
            flow_i = float(population.get("flow_projection_m_s", ui[index]))
            if distribution not in ("maxwellian", "kappa"):
                raise ValueError(f"ion_populations[{index}].distribution must be 'maxwellian' or 'kappa'")
            if distribution == "kappa":
                kappa_i = float(kappa_i)
                if not np.isfinite(kappa_i) or kappa_i <= 1.5:
                    raise ValueError(f"ion_populations[{index}].kappa must be finite and > 1.5")
            elif kappa_i is not None:
                raise ValueError(f"ion_populations[{index}].kappa is only valid for a Kappa ion distribution")
            if not np.isfinite(flow_i) or abs(flow_i) >= C:
                raise ValueError(f"ion_populations[{index}].flow_projection_m_s must be finite and subluminal")
            resolved_ion_populations.append({
                "distribution": distribution, "kappa": kappa_i,
                "flow_projection_m_s": flow_i,
            })
    ion_chi = []
    ion_pdf = []
    for index, item in enumerate(species):
        density = fractions[index]*ne/item["charge_state"]
        sigma_i = np.sqrt(temperatures[index]*E_CHARGE/item["mass_kg"])
        omega_pi2 = density*(item["charge_state"]*E_CHARGE)**2/(EPS0*item["mass_kg"])
        ion_model = resolved_ion_populations[index]
        ion_phase = vphase-ion_model["flow_projection_m_s"]
        if ion_model["distribution"] == "kappa":
            ion_pdf_i, _, _ = _kappa_pdf_and_derivatives(
                ion_phase, sigma_i, ion_model["kappa"])
        else:
            ion_pdf_i, _, _ = _maxwellian_pdf_and_derivatives(ion_phase, sigma_i)
        ion_pdf.append(ion_pdf_i)
        chi_i = np.empty(x.size, dtype=complex)
        for j in range(x.size):
            chi_i[j] = _landau_susceptibility(
                ion_phase[j], omega_pi2, k, sigma_i,
                distribution=ion_model["distribution"],
                kappa=ion_model["kappa"])
        ion_chi.append(chi_i)
    ion_sum = np.sum(np.asarray(ion_chi), axis=0)
    dielectric = 1.0+chi_e+ion_sum
    denominator = np.maximum(np.abs(dielectric)**2, np.finfo(float).tiny)
    electron_term = np.abs(1.0+ion_sum)**2/denominator*electron_pdf
    ion_term = np.zeros(x.size, dtype=float)
    for index, item in enumerate(species):
        ion_term += (
            item["charge_state"]**2*fractions[index]/item["charge_state"]
            *np.abs(chi_e)**2/denominator*ion_pdf[index]
        )
    dynamic_structure = (2.0*np.pi/k)*(electron_term+ion_term)
    omega_per_nm = 2.0*np.pi*C*1e9/(x*x)
    spectral = np.maximum(dynamic_structure, 0.0)*omega_per_nm
    if np.any(~np.isfinite(spectral)) or not np.any(spectral > 0):
        raise ValueError("collective kappa dielectric model produced a non-finite or zero spectrum")
    return {
        "spectral_density_per_nm": spectral,
        "electron_susceptibility": chi_e,
        "ion_susceptibility": np.asarray(ion_chi),
        "dielectric_function": dielectric,
        "dynamic_structure_factor": dynamic_structure,
        "electron_flow_projection_m_s": ue,
        "ion_flow_projections_m_s": np.asarray(
            [item["flow_projection_m_s"] for item in resolved_ion_populations],
            dtype=float,
        ),
        "ion_species": species,
        "ion_fractions": fractions,
        "ion_temperature_eV": temperatures,
        "kappa_index": kap,
        "electron_populations": [dict(item) for item in populations],
        "ion_populations": [dict(item) for item in resolved_ion_populations],
        "electron_population_susceptibilities": np.asarray(population_susceptibilities),
        "electron_population_projected_vdfs_s_m": np.asarray(population_pdfs),
        "wavenumber_m_inverse": k,
        "phase_velocity_m_s": vphase,
        "model": "unmagnetized collective Thomson scattering with kinetic electron and ion mixtures",
    }


def fit_collective_kappa(
    wavelength_nm, signal, laser_wavelength_nm, scattering_angle_rad,
    response, uncertainty=None, *, incident_direction, scattered_direction,
    ion_species, ion_fractions, ion_temperature_eV,
    electron_velocity_m_s, ion_velocity_m_s,
    initial_density_m3, initial_temperature_eV, initial_kappa,
    density_bounds_m3=None, temperature_bounds_eV=None, kappa_bounds=None,
    electron_populations=None, ion_populations=None,
    electron_population_parameter_covariance=None,
    population_mc_samples=100, population_mc_seed=84623,
):
    """Fit ne, Te and electron Kappa while retaining explicit ion kinetic models."""
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    if x.shape != y.shape or x.ndim != 1 or x.size < 12:
        raise ValueError("wavelength_nm and signal must be matching vectors with >=12 samples")
    scale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y/scale
    unc = None if uncertainty is None else np.asarray(uncertainty, dtype=float)/scale
    if unc is not None and (unc.shape != y.shape or np.any(~np.isfinite(unc)) or np.any(unc <= 0)):
        raise ValueError("uncertainty must be positive and match signal")
    residual_scale = np.full_like(ys, max(float(np.max(np.abs(ys))), np.finfo(float).tiny)) if unc is None else unc
    ne0 = _positive(initial_density_m3, "initial_density_m3")
    te0 = _positive(initial_temperature_eV, "initial_temperature_eV")
    kap0 = _positive(initial_kappa-1.5, "initial_kappa_minus_1p5")+1.5
    lower = np.full(3, -np.inf, dtype=float)
    upper = np.full(3, np.inf, dtype=float)
    for bounds, name, index, transform in (
        (density_bounds_m3, "density_bounds_m3", 0, np.log10),
        (temperature_bounds_eV, "temperature_bounds_eV", 1, np.log10),
        (kappa_bounds, "kappa_bounds", 2, lambda value: np.log(value-1.5)),
    ):
        if bounds is None:
            continue
        pair = np.asarray(bounds, dtype=float)
        if pair.shape != (2,) or np.any(~np.isfinite(pair)) or pair[0] >= pair[1]:
            raise ValueError(f"{name} must be a finite increasing pair")
        if index == 2 and pair[0] <= 1.5:
            raise ValueError("kappa_bounds lower bound must be >1.5")
        if index in (0,1) and pair[0] <= 0:
            raise ValueError(f"{name} must be positive")
        if index == 0 and not pair[0] <= ne0 <= pair[1]:
            raise ValueError("initial_density_m3 must lie within density_bounds_m3")
        if index == 1 and not pair[0] <= te0 <= pair[1]:
            raise ValueError("initial_temperature_eV must lie within temperature_bounds_eV")
        if index == 2 and not pair[0] <= kap0 <= pair[1]:
            raise ValueError("initial_kappa must lie within kappa_bounds")
        lower[index], upper[index] = transform(pair[0]), transform(pair[1])
    edge = max(2, x.size//10)
    baseline = float(np.median(np.r_[ys[:edge], ys[-edge:]]))
    amp0 = max(float(np.max(ys)-baseline), np.finfo(float).tiny)
    p0 = np.array([np.log10(ne0), np.log10(te0), np.log(kap0-1.5)])

    def unpack(p):
        return (
            10.0**np.clip(p[0], -300.0, 300.0),
            10.0**np.clip(p[1], -300.0, 300.0),
            1.5+np.exp(np.clip(p[2], -700.0, 700.0)),
        )

    def spectrum(p):
        ne, te, kap = unpack(p)
        physical = collective_kappa_spectral_density(
            x, laser_wavelength_nm, scattering_angle_rad,
            electron_density_m3=ne, electron_temperature_eV=te, kappa=kap,
            ion_species=ion_species, ion_fractions=ion_fractions,
            ion_temperature_eV=ion_temperature_eV,
            electron_velocity_m_s=electron_velocity_m_s,
            ion_velocity_m_s=ion_velocity_m_s,
            incident_direction=incident_direction, scattered_direction=scattered_direction,
            electron_populations=electron_populations, ion_populations=ion_populations,
        )["spectral_density_per_nm"]
        return _apply_response(response, physical, x)

    def nuisance(shape):
        shape_scale=max(float(np.max(np.abs(shape))),np.finfo(float).tiny)
        design=np.column_stack((shape/shape_scale,np.ones_like(shape),x-float(laser_wavelength_nm)))
        coefficients=np.linalg.lstsq(design/residual_scale[:,None],ys/residual_scale,rcond=None)[0]
        if coefficients[0] < 0:
            bg=np.linalg.lstsq(design[:,1:]/residual_scale[:,None],ys/residual_scale,rcond=None)[0]
            coefficients=np.array([0.0,bg[0],bg[1]])
        coefficients[0]/=shape_scale
        return coefficients

    def residual(p, return_fit=False):
        shape=spectrum(p)
        amp,offset,slope=nuisance(shape)
        predicted=amp*shape+offset+slope*(x-float(laser_wavelength_nm))
        value=(predicted-ys)/residual_scale
        return (value, predicted, shape, amp, offset, slope) if return_fit else value

    # `lower` and `upper` were already assembled above in the optimizer's
    # mixed transformed coordinates: log10(ne), log10(Te), and log(kappa-1.5).
    # Rebuilding them here from undefined aliases both raised NameError and
    # discarded the optional/unbounded parameter contract.
    fit=least_squares(residual,p0,bounds=(lower,upper),x_scale="jac",max_nfev=3000)
    if not fit.success:
        raise ValueError(f"collective kappa Thomson fit failed: {fit.message}")
    rank=int(np.linalg.matrix_rank(fit.jac))
    condition=float(np.linalg.cond(fit.jac))
    if rank<3 or not np.isfinite(condition) or condition>1e12:
        raise ValueError("collective kappa Thomson density, temperature and kappa are not jointly identifiable")
    r,predicted,shape,amp,offset,slope=residual(fit.x,return_fit=True)
    ne,te,kap=unpack(fit.x)
    dof=max(1,x.size-6)
    covariance_log=np.linalg.pinv(fit.jac.T@fit.jac)
    if unc is None:
        covariance_log*=float(r@r/dof)
    transform=np.diag([np.log(10.0)*ne,np.log(10.0)*te,kap-1.5])
    covariance=transform@covariance_log@transform.T
    conditional_fit_covariance = covariance.copy()
    population_parameter_names = []
    population_joint_covariance = None
    population_successful_samples = 0
    if electron_population_parameter_covariance is not None:
        if electron_populations is None or not isinstance(electron_populations, (list, tuple)) or len(electron_populations) < 2:
            raise ValueError("electron_population_parameter_covariance requires at least two explicit electron_populations")
        if isinstance(population_mc_samples, (bool, np.bool_)) or not isinstance(population_mc_samples, (int, np.integer)) or population_mc_samples < 20:
            raise ValueError("population_mc_samples must be an integer >=20")
        if isinstance(population_mc_seed, (bool, np.bool_)) or not isinstance(population_mc_seed, (int, np.integer)):
            raise ValueError("population_mc_seed must be an integer")
        population_fields = ("density_fraction", "temperature_ratio", "kappa", "flow_projection_m_s")
        population_mean = []
        for index, population in enumerate(electron_populations):
            if not isinstance(population, Mapping):
                raise ValueError("each electron population must be a mapping")
            missing_fields = [field for field in population_fields if field not in population]
            if missing_fields:
                raise ValueError(f"electron population {index} is missing covariance parameters: {missing_fields}")
            values = np.asarray([population[field] for field in population_fields], dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError("electron population covariance parameters must be finite")
            population_mean.extend(values.tolist())
            population_parameter_names.extend(
                f"population_{index}_{field}" for field in population_fields
            )
        population_mean = np.asarray(population_mean, dtype=float)
        population_covariance = np.asarray(electron_population_parameter_covariance, dtype=float)
        expected_shape = (population_mean.size, population_mean.size)
        if population_covariance.shape != expected_shape or np.any(~np.isfinite(population_covariance)):
            raise ValueError(f"electron_population_parameter_covariance must be finite with shape {expected_shape}")
        covariance_scale = max(float(np.max(np.abs(population_covariance))), np.finfo(float).tiny)
        if not np.allclose(population_covariance, population_covariance.T, rtol=1e-10, atol=1e-12*covariance_scale):
            raise ValueError("electron_population_parameter_covariance must be symmetric")
        population_covariance = 0.5*(population_covariance+population_covariance.T)
        population_eigenvalues = np.linalg.eigvalsh(population_covariance)
        if population_eigenvalues[0] < -1e-10*covariance_scale:
            raise ValueError("electron_population_parameter_covariance must be positive semidefinite")
        if np.any(population_mean[0::4] <= 0.0) or not np.isclose(np.sum(population_mean[0::4]), 1.0, rtol=0.0, atol=1e-8):
            raise ValueError("nominal electron population fractions must be positive and sum to one")
        if np.any(population_mean[1::4] <= 0.0) or np.any(population_mean[2::4] <= 1.5) or np.any(np.abs(population_mean[3::4]) >= C):
            raise ValueError("nominal population temperature ratios, Kappa indices or projected flows are outside their physical domains")
        rng = np.random.default_rng(int(population_mc_seed))
        fit_draws, input_draws = [], []
        attempts = 0
        maximum_attempts = 10*int(population_mc_samples)
        while len(fit_draws) < int(population_mc_samples) and attempts < maximum_attempts:
            attempts += 1
            draw = rng.multivariate_normal(population_mean, population_covariance, check_valid="raise")
            fractions, ratios, kappas, flows = draw[0::4], draw[1::4], draw[2::4], draw[3::4]
            # Do not renormalize fractions or clip physical parameters: rejected
            # draws preserve the stated covariance model without hidden repair.
            if (np.any(fractions <= 0.0) or not np.isclose(np.sum(fractions), 1.0, rtol=0.0, atol=1e-7)
                    or np.any(ratios <= 0.0) or np.any(kappas <= 1.5) or np.any(np.abs(flows) >= C)):
                continue
            sampled_populations = []
            for j, nominal in enumerate(electron_populations):
                sampled = dict(nominal)
                sampled.update({
                    "density_fraction": float(fractions[j]),
                    "temperature_ratio": float(ratios[j]),
                    "kappa": float(kappas[j]),
                    "flow_projection_m_s": float(flows[j]),
                })
                sampled_populations.append(sampled)
            try:
                sampled_fit = fit_collective_kappa(
                    wavelength_nm=x, signal=y, laser_wavelength_nm=laser_wavelength_nm,
                    scattering_angle_rad=scattering_angle_rad, response=response,
                    uncertainty=uncertainty, incident_direction=incident_direction,
                    scattered_direction=scattered_direction, ion_species=ion_species,
                    ion_fractions=ion_fractions, ion_temperature_eV=ion_temperature_eV,
                    electron_velocity_m_s=electron_velocity_m_s, ion_velocity_m_s=ion_velocity_m_s,
                    initial_density_m3=ne, initial_temperature_eV=te, initial_kappa=kap,
                    density_bounds_m3=density_bounds_m3, temperature_bounds_eV=temperature_bounds_eV,
                    kappa_bounds=kappa_bounds, electron_populations=sampled_populations,
                    ion_populations=ion_populations,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            fit_draws.append(np.asarray(sampled_fit["fit_parameters"], dtype=float))
            input_draws.append(draw)
        minimum_successful = max(20, int(np.ceil(0.7*int(population_mc_samples))))
        if len(fit_draws) < minimum_successful:
            raise ValueError(
                f"electron population uncertainty propagation produced {len(fit_draws)} successful refits; "
                f"at least {minimum_successful} are required. Check the supplied covariance and physical bounds."
            )
        population_successful_samples = len(fit_draws)
        joint_draws = np.column_stack((np.asarray(fit_draws), np.asarray(input_draws)))
        population_joint_covariance = np.cov(joint_draws, rowvar=False, ddof=1)
        # Add the conditional spectral-fit covariance to the fitted-parameter
        # block while retaining MC cross-covariance with population inputs.
        population_joint_covariance[:3, :3] += conditional_fit_covariance
        covariance = population_joint_covariance[:3, :3].copy()
    sigma=np.sqrt(np.maximum(np.diag(covariance),0.0))
    incident_unit=np.asarray(incident_direction,dtype=float)
    scattered_unit=np.asarray(scattered_direction,dtype=float)
    incident_unit=incident_unit/np.linalg.norm(incident_unit)
    scattered_unit=scattered_unit/np.linalg.norm(scattered_unit)
    qhat=scattered_unit-incident_unit
    qhat=qhat/np.linalg.norm(qhat)
    ue=float(np.dot(np.asarray(electron_velocity_m_s,dtype=float),qhat))
    ion_properties=_resolve_ion_species(ion_species)
    ion_v=np.asarray(ion_velocity_m_s,dtype=float)
    if ion_v.shape==(3,) and len(ion_properties)==1:
        ion_v=ion_v.reshape(1,3)
    ui=ion_v@qhat
    if ion_populations is not None:
        ui=np.asarray(
            [float(item.get("flow_projection_m_s", ui[index]))
             for index, item in enumerate(ion_populations)],
            dtype=float,
        )
    k=4.0*np.pi*np.sin(float(scattering_angle_rad)/2.0)/(float(laser_wavelength_nm)*1e-9)
    debye=np.sqrt(EPS0*te/(ne*E_CHARGE))
    alpha=1.0/(k*debye)
    fitted=predicted*scale
    residual_signal=y-fitted
    return {
        "electron_density_m3":float(ne),
        "electron_density_uncertainty_m3":float(sigma[0]),
        "electron_temperature_eV":float(te),
        "electron_temperature_uncertainty_eV":float(sigma[1]),
        "kappa_index":float(kap),
        "kappa_index_uncertainty":float(sigma[2]),
        "electron_velocity_projection_m_s":ue,
        "electron_flow_source":"explicitly supplied electron velocity vector",
        "ion_velocity_projection_m_s":ui.tolist(),
        "ion_species":[item["name"] for item in ion_properties],
        "ion_fractions":np.asarray(ion_fractions,dtype=float).tolist(),
        "ion_temperature_eV":np.asarray(np.atleast_1d(ion_temperature_eV),dtype=float).tolist(),
        "ion_velocity_m_s":ion_v.tolist(),
        "electron_velocity_m_s":np.asarray(electron_velocity_m_s,dtype=float).tolist(),
        "fitted_spectrum":fitted,
        "residual":residual_signal,
        "amplitude":float(amp*scale),
        "background_intercept":float(offset*scale),
        "background_slope_per_nm":float(slope*scale),
        "fit_covariance":covariance,
        "fit_covariance_conditional_on_population_inputs":conditional_fit_covariance,
        "electron_population_parameter_names":population_parameter_names,
        "electron_population_parameter_covariance":None if electron_population_parameter_covariance is None else np.asarray(electron_population_parameter_covariance,dtype=float),
        "population_joint_fit_input_covariance":population_joint_covariance,
        "population_uncertainty_successful_refits":int(population_successful_samples),
        "population_uncertainty_seed":int(population_mc_seed) if electron_population_parameter_covariance is not None else None,
        "fit_parameters":np.array([ne,te,kap],dtype=float),
        "fit_parameter_names":["electron_density_m3","electron_temperature_eV","kappa_index"],
        "optimizer_success":bool(fit.success),
        "optimizer_message":str(fit.message),
        "collective_parameter_jacobian_rank":rank,
        "collective_parameter_jacobian_condition_number":condition,
        "reduced_chi_square":float(np.sum((residual_signal/np.asarray(uncertainty,dtype=float))**2)/dof) if uncertainty is not None else None,
        "scattering_parameter_alpha":float(alpha),
        "scattering_regime":"collective" if alpha>1.0 else "non_collective_or_weakly_collective",
        "model":"Plasma kinetic dielectric model with configurable electron and ion Kappa/Maxwellian populations",
        "distribution_model":"density-weighted electron populations and per-species ion populations; Maxwellian and finite-variance Kappa responses are supported; ion populations are fixed inputs during ne/Te/kappa inference",
        "electron_populations":None if electron_populations is None else [dict(item) for item in electron_populations],
        "ion_populations":None if ion_populations is None else [dict(item) for item in ion_populations],
    }


def fit_relativistic_maxwell_juttner_channels(
    wavelength_nm, measured_spectra, laser_wavelength_nm, scattering_angle_rad,
    response, *, incident_direction, scattered_direction, incident_polarization,
    joint_spectral_covariance, initial_temperature_eV,
    initial_flow_projection_m_s=0.0, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01, include_compton_recoil=False,
    background_model="linear", background_degree=1, background_interior_knots=3,
    background_templates=None, max_nfev=1600,
):
    """Joint relativistic MJ fit with full cross-channel GLS and structured backgrounds.

    Data/covariance use channel-major ordering. A shared Te and q-parallel flow
    are inferred jointly; each channel has an independent amplitude and an
    independently weighted detector-space background basis. Supported bases
    are constant, linear, polynomial, cubic_spline and researcher-supplied
    templates. Backgrounds are nuisance terms, not plasma emission.
    """
    from .thomson_background import _background_design

    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(measured_spectra, dtype=float)
    if x.ndim != 1 or x.size < 12 or np.any(~np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be finite, strictly increasing, with >=12 samples")
    if y.ndim != 2 or y.shape[1] != x.size or y.shape[0] < 2 or np.any(~np.isfinite(y)):
        raise ValueError("measured_spectra must be a finite (n_channels,n_wavelengths) array with >=2 channels")
    m, n = y.shape
    cov = np.asarray(joint_spectral_covariance, dtype=float)
    if cov.shape != (m*n, m*n) or np.any(~np.isfinite(cov)):
        raise ValueError("joint_spectral_covariance must be finite and shaped (n_channels*n_wavelengths)^2 in channel-major order")
    scale_cov = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12*scale_cov):
        raise ValueError("joint_spectral_covariance must be symmetric")
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("joint_spectral_covariance must be positive definite") from exc
    ni = np.asarray(incident_direction, dtype=float)
    ns = np.asarray(scattered_direction, dtype=float)
    pol = np.asarray(incident_polarization, dtype=float)
    if ni.shape != (3,) or ns.shape != (3,) or pol.shape != (3,) or any(np.any(~np.isfinite(v)) for v in (ni, ns, pol)):
        raise ValueError("incident_direction, scattered_direction and incident_polarization must be finite 3-vectors")
    if min(np.linalg.norm(ni), np.linalg.norm(ns), np.linalg.norm(pol)) <= 0:
        raise ValueError("direction and polarization vectors must be non-zero")
    ni = ni/np.linalg.norm(ni)
    ns = ns/np.linalg.norm(ns)
    pol = pol/np.linalg.norm(pol)
    if abs(float(np.dot(pol, ni))) > 1e-6:
        raise ValueError("incident_polarization must be transverse to incident_direction")
    angle = float(np.arccos(np.clip(np.dot(ni, ns), -1.0, 1.0)))
    if not np.isclose(angle, scattering_angle_rad, rtol=0.0, atol=1e-5):
        raise ValueError("scattering_angle_rad is inconsistent with direction vectors")
    qhat = ns-ni
    qhat /= np.linalg.norm(qhat)
    e3 = np.cross(ni, ns)
    if np.linalg.norm(e3) <= 1e-12:
        e3 = pol-np.dot(pol, qhat)*qhat
    e3 /= np.linalg.norm(e3)
    e2 = np.cross(qhat, e3)
    e2 /= np.linalg.norm(e2)
    local_to_lab = np.column_stack((qhat, e2, e3))
    pol_local = local_to_lab.T @ pol
    laser = _positive(laser_wavelength_nm, "laser_wavelength_nm")
    te0 = _positive(initial_temperature_eV, "initial_temperature_eV")
    flow0 = float(initial_flow_projection_m_s)
    if not np.isfinite(flow0) or abs(flow0) >= C:
        raise ValueError("initial_flow_projection_m_s must be finite and subluminal")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    if isinstance(response, (list, tuple)) and len(response) == m:
        responses = [np.asarray(item, dtype=float) for item in response]
    else:
        responses = [np.asarray(response, dtype=float)]*m
    for item in responses:
        if item.shape != (n, n) or np.any(~np.isfinite(item)) or np.any(item < 0):
            raise ValueError("response must be a finite non-negative (n_lambda,n_lambda) matrix per channel")
        if not np.allclose(item.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("response columns must already sum to one; calibrated response is not renormalized")
    if background_model not in {"constant", "linear", "polynomial", "cubic_spline", "templates"}:
        raise ValueError("background_model must be constant, linear, polynomial, cubic_spline or templates")
    if background_templates is not None and np.asarray(background_templates).ndim == 3:
        template_by_channel = [np.asarray(background_templates, dtype=float)[j] for j in range(m)]
    elif isinstance(background_templates, (list, tuple)):
        if len(background_templates) != m:
            raise ValueError("background_templates list must contain one matrix per channel")
        template_by_channel = list(background_templates)
    else:
        template_by_channel = [background_templates] * m
    designs = [
        _background_design(
            x, background_model, degree=background_degree,
            interior_knots=background_interior_knots,
            templates=template_by_channel[j],
        )
        for j in range(m)
    ]
    nbg = designs[0].shape[1]
    if any(d.shape[1] != nbg for d in designs):
        raise ValueError("all channel background bases must have the same number of columns")
    yscale = max(float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y/yscale
    chol_scaled = chol/yscale
    edge = max(2, n//10)
    baseline = np.median(np.concatenate((ys[:, :edge], ys[:, -edge:]), axis=1), axis=1)
    amplitudes = np.maximum(np.max(ys, axis=1)-baseline, np.finfo(float).tiny)
    bg0 = np.concatenate([
        np.linalg.lstsq(designs[j], np.full(n, baseline[j]), rcond=None)[0]
        for j in range(m)
    ])
    p0 = np.r_[np.log(te0), np.arctanh(flow0/C), np.log(amplitudes), bg0]
    def predict(p):
        te = float(np.exp(np.clip(p[0], -700, 700)))
        flow = float(np.tanh(p[1])*C)
        physical = relativistic_maxwell_juttner_spectrum(
            x, laser, angle, te, flow_projection_m_s=flow,
            incident_polarization=pol_local, radial_order=radial_order,
            angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
            require_full_spectrum=False,
        )
        shape = physical["spectral_density_per_nm"]
        out = np.empty_like(ys)
        amps = np.exp(np.clip(p[2:2+m], -700, 700))
        coefficients = p[2+m:].reshape(m, nbg)
        for j in range(m):
            convolved = _apply_response(responses[j], shape, x)
            out[j] = amps[j]*convolved + designs[j]@coefficients[j]
        return out
    def objective(p):
        try:
            residual = (ys-predict(p)).reshape(-1)
            return np.linalg.solve(chol_scaled, residual)
        except ValueError as exc:
            if "capture at least 99.5%" in str(exc) or "zero spectral power" in str(exc):
                return np.full(m*n, 1e6)
            raise
    bounds = (np.r_[-np.inf, -18.0, np.full(m + m*nbg, -np.inf)],
              np.r_[np.inf, 18.0, np.full(m + m*nbg, np.inf)])
    fit = least_squares(objective, p0, bounds=bounds, x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success:
        raise ValueError(f"joint relativistic Maxwell-Juttner fit failed: {fit.message}")
    jac = np.asarray(fit.jac, dtype=float)
    norms = np.linalg.norm(jac, axis=0)
    scaled_jac = np.divide(jac, norms[None, :], out=np.zeros_like(jac), where=norms[None, :] > np.finfo(float).tiny)
    singular = np.linalg.svd(scaled_jac, compute_uv=False)
    rank_tol = np.finfo(float).eps*max(scaled_jac.shape)*(singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > rank_tol))
    condition = float(singular[0]/singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    if rank < p0.size or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"joint relativistic Maxwell-Juttner fit is not identifiable (rank {rank}/{p0.size}, condition={condition:.6g})")
    covariance_p = np.linalg.pinv(jac.T@jac, rcond=1e-12)
    transform = np.eye(p0.size)
    transform[0, 0] = np.exp(fit.x[0])
    transform[1, 1] = C*(1.0-np.tanh(fit.x[1])**2)
    transform[2:2+m, 2:2+m] = np.diag(np.exp(fit.x[2:2+m])*yscale)
    transform[2+m:, 2+m:] *= yscale
    covariance_physical = transform@covariance_p@transform.T
    fitted = predict(fit.x)*yscale
    residual = y-fitted
    coefficients = fit.x[2+m:].reshape(m, nbg)*yscale
    names = ["electron_temperature_eV","electron_velocity_projection_m_s"]
    names += [f"channel_{j}_amplitude" for j in range(m)]
    names += [f"channel_{j}_background_coefficient_{k}" for j in range(m) for k in range(nbg)]
    background_intercepts = np.array([designs[j][np.argmin(np.abs(x-laser))]@coefficients[j] for j in range(m)])
    result = {
        "electron_temperature_eV": float(np.exp(fit.x[0])),
        "electron_temperature_uncertainty_eV": float(np.sqrt(max(covariance_physical[0,0],0.0))),
        "electron_velocity_projection_m_s": float(np.tanh(fit.x[1])*C),
        "electron_velocity_projection_uncertainty_m_s": float(np.sqrt(max(covariance_physical[1,1],0.0))),
        "channel_amplitudes": np.exp(fit.x[2:2+m])*yscale,
        "channel_background_intercepts": background_intercepts,
        "channel_background_coefficients": coefficients,
        "background_model": background_model,
        "background_basis_matrices": [d.copy() for d in designs],
        "fitted_spectra": fitted,
        "residuals": residual,
        "weighted_residual_sum_squares": float(2.0*fit.cost),
        "fit_covariance": covariance_physical,
        "fit_parameter_names": names,
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "spectral_weighting_status": "full_channel_major_generalized_least_squares_including_cross_channel_covariance",
        "covariance_ordering": "channel_major",
        "model": "relativistic non-collective Thomson scattering; shared Maxwell-Juttner Te and q-parallel flow across channels",
        "recoil_included": bool(include_compton_recoil),
        "scattering_cross_section_model": ("polarized_single_scattering_Klein_Nishina" if include_compton_recoil else "polarized_elastic_Thomson"),
        "density_inference": "not_inferred_from_normalized_spectral_shape",
        "assumptions": ["All channels share Te, q-parallel bulk flow, wavelength grid and scattering geometry.", "Channel amplitudes and structured detector-space background coefficients are independent nuisance parameters.", "The supplied full covariance is known, positive definite and includes cross-channel correlations.", "A common measured response is used unless a per-channel response list is supplied.", "Single-scattering photon recoil and polarized Klein-Nishina weighting are included only when include_compton_recoil=True; otherwise the configured Thomson-limit correction threshold is enforced.", "Background basis is researcher-selected and may be confounded with the plasma feature; Jacobian rank/conditioning is the identifiability check."]
    }
    if background_model == "linear":
        z = 2.0*(x-x.min())/np.ptp(x)-1.0
        result["channel_background_slopes_per_nm"] = coefficients[:, 1]*2.0/np.ptp(x)
        laser_z = 2.0*(laser-x.min())/np.ptp(x)-1.0
        result["channel_background_intercepts"] = coefficients[:, 0] + coefficients[:, 1]*laser_z
    return result



def project_gyrotropic_bi_kappa_electron_vdf(
    velocity_grid_m_s, *, temperature_parallel_eV,
    temperature_perpendicular_eV, kappa, magnetic_field_unit,
    scattering_vector_unit, flow_projection_m_s=0.0,
):
    r"""Return the exact 1-D marginal of a gyrotropic 3-D finite-variance Kappa EVDF.

    The parent distribution is proportional to
      [1 + v_parallel**2/(kappa*theta_parallel**2)
         + v_perp**2/(kappa*theta_perpendicular**2)]**(-(kappa+1)).
    For mu = b_hat.q_hat, the projected second-moment temperature is
      T_q = T_perp*(1-mu**2) + T_parallel*mu**2,
    and theta_q**2 = ((2*kappa-3)/kappa) * e*T_q/m_e.
    Its normalized marginal is
      p(v_q) = Gamma(kappa) /
        (sqrt(pi*kappa)*Gamma(kappa-1/2)*theta_q)
        * [1 + (v_q-u_q)**2/(kappa*theta_q**2)]**(-kappa).

    The finite-variance requirement kappa>3/2 is enforced. The finite supplied
    velocity grid is not renormalized; represented probability coverage is
    returned explicitly. A single view identifies T_q, not both anisotropy
    temperatures independently.
    """
    grid = np.asarray(velocity_grid_m_s)
    if grid.dtype.kind not in "iuf" or grid.ndim != 1 or grid.size < 3:
        raise ValueError("velocity_grid_m_s must be a real one-dimensional grid with at least three points")
    grid = grid.astype(float, copy=False)
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("velocity_grid_m_s must be finite and strictly increasing")
    t_parallel = _positive(temperature_parallel_eV, "temperature_parallel_eV")
    t_perpendicular = _positive(temperature_perpendicular_eV, "temperature_perpendicular_eV")
    kappa_value = _positive(kappa, "kappa")
    if kappa_value <= 1.5:
        raise ValueError("finite-variance gyrotropic Kappa requires kappa > 3/2")
    b = np.asarray(magnetic_field_unit)
    q = np.asarray(scattering_vector_unit)
    if b.dtype.kind not in "iuf" or q.dtype.kind not in "iuf" or b.shape != (3,) or q.shape != (3,):
        raise ValueError("magnetic_field_unit and scattering_vector_unit must be real three-component vectors")
    b, q = b.astype(float, copy=False), q.astype(float, copy=False)
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(q)):
        raise ValueError("direction vectors must be finite")
    if not np.isclose(np.linalg.norm(b), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("magnetic_field_unit must have unit norm")
    if not np.isclose(np.linalg.norm(q), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("scattering_vector_unit must have unit norm")
    if isinstance(flow_projection_m_s, (bool, np.bool_)) or not np.isscalar(flow_projection_m_s):
        raise ValueError("flow_projection_m_s must be a finite scalar")
    flow = float(flow_projection_m_s)
    if not np.isfinite(flow):
        raise ValueError("flow_projection_m_s must be finite")
    mu = float(np.clip(np.dot(b, q), -1.0, 1.0))
    directional_temperature = t_perpendicular * (1.0 - mu**2) + t_parallel * mu**2
    theta_squared = ((2.0*kappa_value - 3.0)/kappa_value) * E_CHARGE * directional_temperature / ME
    theta = float(np.sqrt(theta_squared))
    shifted = grid - flow
    log_normalization = (
        gammaln(kappa_value)
        - 0.5*np.log(np.pi*kappa_value)
        - gammaln(kappa_value - 0.5)
        - np.log(theta)
    )
    log_shape = -kappa_value * np.log1p(shifted**2/(kappa_value*theta_squared))
    distribution = np.exp(log_normalization + log_shape)
    degrees = 2.0*kappa_value - 1.0
    student_scale = theta*np.sqrt(kappa_value/degrees)
    coverage = float(
        stdtr(degrees, (grid[-1]-flow)/student_scale)
        - stdtr(degrees, (grid[0]-flow)/student_scale)
    )
    if not np.isfinite(coverage) or coverage <= 0.0:
        raise ValueError("velocity_grid_m_s does not cover a positive finite portion of the projected Kappa distribution")
    return {
        "velocity_grid_m_s": grid.copy(),
        "projected_vdf_s_m": distribution,
        "velocity_grid_probability_coverage": coverage,
        "projected_temperature_eV": float(directional_temperature),
        "projected_velocity_mean_m_s": flow,
        "projected_kappa_index": float(kappa_value),
        "projected_kappa_scale_m_s": theta,
        "degrees_of_freedom": float(degrees),
        "magnetic_pitch_cosine": mu,
        "anisotropy_identifiability": "only_the_directional_temperature_combination_is_identified_from_one_scattering_direction",
        "separate_parallel_perpendicular_temperatures_identifiable": False,
        "model": "gyrotropic_finite_variance_bi_kappa_exact_projected_marginal",
        "assumptions": [
            "Gyrotropic 3-D Kappa distribution with a common magnetic-field axis",
            "Zero heat flux and finite variance (kappa > 3/2)",
            "Non-relativistic elastic-Thomson projected velocity coordinate",
        ],
        "limitations": [
            "This is a forward projected-VDF kernel, not a measured-spectrum optimizer",
            "One scattering direction cannot separate parallel and perpendicular temperatures",
            "Finite-grid probability is reported without tail renormalization",
        ],
    }

"""Full symmetric 3-D electron-temperature tensor inversion from Thomson views.

Each calibrated Thomson spectrum supplies a projected second-moment temperature
T_q = (m_e/e) q^T <delta_v delta_v> q.  This module inverts those projections
without imposing gyrotropy about a prescribed magnetic-field direction.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

_E = 1.602176634e-19
_M_E = 9.1093837015e-31


def _finite_array(value, name, ndim):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != ndim:
        raise ValueError(f"{name} must be a real numeric {ndim}-D array")
    result = raw.astype(float, copy=False)
    if result.size == 0 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite and non-empty")
    return result


def _tensor_from_components(parameters):
    xx, yy, zz, xy, xz, yz = np.asarray(parameters, dtype=float)
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def infer_full_temperature_tensor_from_thomson_projections(
    projected_temperature_eV,
    scattering_vector_unit,
    *,
    projected_temperature_covariance_eV2=None,
    maximum_geometry_condition_number=1.0e6,
    minimum_consistency_p_value=1.0e-3,
):
    """Infer a general symmetric PSD 3-D temperature tensor from q-projections.

    Uses T_q = q^T T_eV q, where T_eV=(m_e/e)<delta_v delta_v>.
    At least six linearly independent view dyads q q^T are required. The
    covariance is channel-major over projected temperatures and is treated as
    an absolute measurement covariance when supplied. A PSD-constrained GLS
    solve prevents an unphysical negative principal temperature.
    """
    temperature = _finite_array(projected_temperature_eV, "projected_temperature_eV", 1)
    directions = _finite_array(scattering_vector_unit, "scattering_vector_unit", 2)
    n = temperature.size
    if directions.shape != (n, 3) or n < 6:
        raise ValueError("at least six paired projected temperatures and 3-D scattering directions are required")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("each scattering_vector_unit row must have unit norm")
    if np.any(temperature <= 0.0):
        raise ValueError("projected temperatures must be strictly positive")
    condition_limit = float(maximum_geometry_condition_number)
    p_limit = float(minimum_consistency_p_value)
    if not np.isfinite(condition_limit) or condition_limit <= 1.0:
        raise ValueError("maximum_geometry_condition_number must be finite and > 1")
    if not np.isfinite(p_limit) or not 0.0 <= p_limit < 1.0:
        raise ValueError("minimum_consistency_p_value must lie in [0,1)")

    qx, qy, qz = directions.T
    design = np.column_stack((qx*qx, qy*qy, qz*qz,
                              2.0*qx*qy, 2.0*qx*qz, 2.0*qy*qz))
    singular = np.linalg.svd(design, compute_uv=False)
    tolerance = np.finfo(float).eps * max(design.shape) * singular[0]
    rank = int(np.count_nonzero(singular > tolerance))
    if rank < 6:
        raise ValueError(
            f"full temperature tensor is not identifiable: view-dyad rank={rank}, required=6"
        )
    condition = float(singular[0] / singular[-1])
    if not np.isfinite(condition) or condition > condition_limit:
        raise ValueError(
            f"full temperature tensor geometry is ill-conditioned ({condition:.6g} > {condition_limit:.6g})"
        )

    if projected_temperature_covariance_eV2 is None:
        covariance = np.eye(n, dtype=float)
        covariance_source = "identity_metric_no_measurement_covariance_supplied"
    else:
        covariance = _finite_array(
            projected_temperature_covariance_eV2,
            "projected_temperature_covariance_eV2", 2
        )
        if covariance.shape != (n, n):
            raise ValueError("projected_temperature_covariance_eV2 must be (n_view,n_view)")
        covariance = 0.5 * (covariance + covariance.T)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("projected-temperature covariance must be positive definite") from exc
        covariance_source = "supplied_absolute_full_view_covariance"
    if projected_temperature_covariance_eV2 is None:
        chol = np.eye(n)
    whitened_design = np.linalg.solve(chol, design)
    whitened_data = np.linalg.solve(chol, temperature)
    information = whitened_design.T @ whitened_design
    if np.linalg.matrix_rank(information) < 6:
        raise ValueError("whitened temperature-tensor information matrix is rank deficient")

    unconstrained = np.linalg.solve(information, whitened_design.T @ whitened_data)
    scale = max(float(np.max(temperature)), 1.0)
    tensor_unconstrained = _tensor_from_components(unconstrained)
    psd_active = bool(np.min(np.linalg.eigvalsh(tensor_unconstrained)) < -1e-10 * scale)
    if psd_active:
        def objective(parameters):
            residual = whitened_design @ parameters - whitened_data
            return 0.5 * float(residual @ residual)

        initial = np.array([max(float(np.mean(temperature)), 1e-8)] * 3 + [0.0, 0.0, 0.0])
        constraints = [{
            "type": "ineq",
            "fun": lambda p: np.linalg.eigvalsh(_tensor_from_components(p)) / scale,
        }]
        fit = minimize(
            objective, initial, method="SLSQP", constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 2000},
        )
        if not fit.success:
            raise RuntimeError(f"PSD-constrained temperature tensor inversion failed: {fit.message}")
        parameters = np.asarray(fit.x, dtype=float)
        tensor = _tensor_from_components(parameters)
        psd_active = bool(np.min(np.linalg.eigvalsh(tensor)) <= 1e-8 * scale)
    else:
        parameters = unconstrained
        tensor = tensor_unconstrained

    predicted = design @ parameters
    residual = temperature - predicted
    whitened_residual = np.linalg.solve(chol, residual)
    chi_square = float(whitened_residual @ whitened_residual)
    dof = n - 6
    p_value = float(chi2.sf(chi_square, dof)) if dof > 0 else None
    if p_value is not None and p_value < p_limit:
        consistency = "inconsistent_with_single_common_temperature_tensor"
    else:
        consistency = "consistent_or_not_testable"

    # The usual local GLS covariance is valid only in the interior of the PSD
    # cone. At an active eigenvalue boundary, symmetric Gaussian errors are not
    # a valid confidence description and are deliberately not reported.
    parameter_covariance = None
    covariance_status = "not_reported_psd_boundary"
    if not psd_active:
        parameter_covariance = np.linalg.inv(information)
        if projected_temperature_covariance_eV2 is None:
            parameter_covariance *= chi_square / max(1, dof)
        covariance_status = "local_gls_covariance_interior_psd_cone"

    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    return {
        "electron_temperature_tensor_eV": tensor,
        "thermal_velocity_covariance_tensor_m2_s2": (_E / _M_E) * tensor,
        "principal_temperature_eigenvalues_eV": eigenvalues,
        "principal_axes_cartesian_columns": eigenvectors,
        "predicted_projected_temperature_eV": predicted,
        "projected_temperature_residual_eV": residual,
        "chi_square": chi_square,
        "degrees_of_freedom": dof,
        "consistency_p_value": p_value,
        "consistency_status": consistency,
        "view_dyad_rank": rank,
        "view_dyad_condition_number": condition,
        "parameter_order": ["T_xx_eV", "T_yy_eV", "T_zz_eV",
                            "T_xy_eV", "T_xz_eV", "T_yz_eV"],
        "parameter_covariance_eV2": parameter_covariance,
        "parameter_covariance_status": covariance_status,
        "positive_semidefinite_constraint_active": psd_active,
        "covariance_provenance": covariance_source,
        "projection_equation": "T_q_eV = q_hat.T @ T_eV @ q_hat",
        "physical_scope": (
            "general_symmetric_second_moment_tensor; no gyrotropy or magnetic-axis "
            "alignment is assumed; this does not identify higher EVDF moments"
        ),
        "reportable_as_inferred_parameter": bool(
            consistency != "inconsistent_with_single_common_temperature_tensor"
        ),
        "experimental_validation": "not_performed",
    }

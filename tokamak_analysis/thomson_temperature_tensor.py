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


def _finite_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
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
    electron_density_m3=None,
    electron_density_uncertainty_m3=None,
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

    # Remove only round-off-scale negative eigenvalues left by the constrained
    # optimizer. Recompute the six components so all reported projections and
    # residuals correspond to the returned PSD tensor.
    eigenvalues_psd, eigenvectors_psd = np.linalg.eigh(tensor)
    if eigenvalues_psd[0] < 0.0:
        if eigenvalues_psd[0] < -1e-7 * scale:
            raise ArithmeticError("temperature tensor solver returned a materially non-PSD tensor")
        tensor = (eigenvectors_psd * np.maximum(eigenvalues_psd, 0.0)) @ eigenvectors_psd.T
        parameters = np.array([
            tensor[0, 0], tensor[1, 1], tensor[2, 2],
            tensor[0, 1], tensor[0, 2], tensor[1, 2],
        ])
        psd_active = True

    predicted = design @ parameters
    residual = temperature - predicted
    whitened_residual = np.linalg.solve(chol, residual)
    chi_square = float(whitened_residual @ whitened_residual)
    dof = n - 6
    # A chi-square survival probability is meaningful only when the supplied
    # covariance is an absolute measurement-noise model. The identity metric
    # used for unweighted least squares is not a noise estimate.
    covariance_is_measured = projected_temperature_covariance_eV2 is not None
    p_value = float(chi2.sf(chi_square, dof)) if covariance_is_measured and dof > 0 else None
    if p_value is not None and p_value < p_limit:
        consistency = "inconsistent_with_single_common_temperature_tensor"
    elif p_value is None:
        consistency = "not_testable_without_absolute_covariance_or_redundant_views"
    else:
        consistency = "consistent_with_single_common_temperature_tensor"

    # The usual local GLS covariance is valid only in the interior of the PSD
    # cone. At an active eigenvalue boundary, symmetric Gaussian errors are not
    # a valid confidence description and are deliberately not reported.
    parameter_covariance = None
    covariance_status = "not_reported_psd_boundary"
    if not psd_active:
        if covariance_is_measured:
            parameter_covariance = np.linalg.inv(information)
            covariance_status = "absolute_measurement_covariance_gls_interior_psd_cone"
        elif dof > 0:
            # With no supplied absolute noise model, estimate a common residual
            # variance only when redundant views provide positive degrees of
            # freedom. Exactly six independent projections determine six tensor
            # components algebraically but cannot estimate their uncertainty.
            residual_variance = chi_square / dof
            if residual_variance > 0.0:
                parameter_covariance = np.linalg.inv(information) * residual_variance
                covariance_status = "residual_variance_estimated_from_redundant_views"
            else:
                covariance_status = "not_reported_zero_residual_variance_unresolved"
        else:
            covariance_status = "not_reported_without_absolute_covariance_or_redundant_views"

    eigenvalues, eigenvectors = np.linalg.eigh(tensor)

    # Optional thermodynamic closure: P_e = n_e e T_eV. Density is an
    # independently supplied measurement, never inferred from spectral width.
    pressure_tensor = None
    pressure_covariance = None
    pressure_covariance_status = "not_available_without_temperature_covariance"
    density_uncertainty = None
    if electron_density_m3 is not None:
        density = float(electron_density_m3)
        if not np.isfinite(density) or density <= 0.0:
            raise ValueError("electron_density_m3 must be finite and strictly positive")
        if electron_density_uncertainty_m3 is not None:
            density_uncertainty = float(electron_density_uncertainty_m3)
            if not np.isfinite(density_uncertainty) or density_uncertainty < 0.0:
                raise ValueError("electron_density_uncertainty_m3 must be finite and non-negative")
        tensor_components = np.array([
            tensor[0, 0], tensor[1, 1], tensor[2, 2],
            tensor[0, 1], tensor[0, 2], tensor[1, 2],
        ])
        pressure_tensor = density * _E * tensor
        if parameter_covariance is not None:
            pressure_covariance = (density * _E) ** 2 * parameter_covariance
            if density_uncertainty is not None:
                pressure_covariance += (_E * density_uncertainty) ** 2 * np.outer(
                    tensor_components, tensor_components
                )
                pressure_covariance_status = (
                    "independent_temperature_gls_and_density_uncertainty_delta_method"
                )
            else:
                pressure_covariance_status = "temperature_gls_only_density_treated_as_exact"
    elif electron_density_uncertainty_m3 is not None:
        raise ValueError("electron_density_uncertainty_m3 requires electron_density_m3")

    return {
        "electron_temperature_tensor_eV": tensor,
        "thermal_velocity_covariance_tensor_m2_s2": (_E / _M_E) * tensor,
        "electron_density_m3": None if electron_density_m3 is None else float(electron_density_m3),
        "electron_density_uncertainty_m3": density_uncertainty,
        "electron_pressure_tensor_Pa": pressure_tensor,
        "electron_pressure_parameter_covariance_Pa2": pressure_covariance,
        "electron_pressure_covariance_status": pressure_covariance_status,
        "electron_pressure_equation": "P_e_Pa = n_e_m-3 * elementary_charge_C * T_e_eV",
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

def _tensor_finite_bin_gaussian_density(wavelength_nm, centre_nm, sigma_nm):
    """Return exact Gaussian probability density averaged over wavelength bins."""
    wavelength = _finite_array(wavelength_nm, "wavelength_nm", 1)
    if wavelength.size < 3 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must contain at least three increasing bin centres")
    sigma, centre = float(sigma_nm), float(centre_nm)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(centre):
        raise ValueError("Gaussian centre must be finite and width strictly positive")
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    z = (edges - centre) / sigma
    from scipy.special import ndtr
    probability = ndtr(z[1:]) - ndtr(z[:-1])
    positive = z[:-1] > 0.0
    probability[positive] = ndtr(-z[:-1][positive]) - ndtr(-z[1:][positive])
    widths = np.diff(edges)
    return probability / widths, probability, widths


def _tensor_finite_bin_kappa_density(wavelength_nm, centre_nm, sigma_nm, kappa_index):
    """Exact finite-bin projected marginal of an ellipsoidal 3-D Kappa EVDF."""
    wavelength = _finite_array(wavelength_nm, "wavelength_nm", 1)
    sigma, centre, kappa = float(sigma_nm), float(centre_nm), float(kappa_index)
    if wavelength.size < 3 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must contain at least three increasing bin centres")
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(centre):
        raise ValueError("Kappa centre must be finite and projected standard deviation strictly positive")
    if not np.isfinite(kappa) or kappa <= 1.5:
        raise ValueError("Kappa index must be finite and >1.5 for a finite second moment")
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[1:-1] = 0.5*(wavelength[:-1]+wavelength[1:])
    edges[0] = wavelength[0]-0.5*(wavelength[1]-wavelength[0])
    edges[-1] = wavelength[-1]+0.5*(wavelength[-1]-wavelength[-2])
    widths = np.diff(edges)
    degrees = 2.0*kappa-1.0
    scale = sigma*np.sqrt((degrees-2.0)/degrees)
    z = (edges-centre)/scale
    from scipy.special import stdtr
    probability = stdtr(degrees,z[1:])-stdtr(degrees,z[:-1])
    positive = z[:-1] > 0.0
    probability[positive] = stdtr(degrees,-z[:-1][positive])-stdtr(degrees,-z[1:][positive])
    return probability/widths, probability, widths


def fit_general_temperature_tensor_thomson_spectra(
    wavelength_nm,
    measured_spectra,
    *,
    laser_wavelength_nm,
    scattering_angle_rad,
    scattering_vector_unit,
    instrument_response_matrix,
    measurement_covariance=None,
    signal_uncertainty=None,
    model_discrepancy_covariance=None,
    systematic_covariance=None,
    initial_temperature_tensor_eV=None,
    initial_electron_velocity_vector_m_s=None,
    distribution_model="maxwellian",
    initial_kappa_index=None,
    fit_kappa_index=False,
    kappa_bounds=None,
    maximum_projected_rms_beta=0.1,
    maximum_flow_beta=0.1,
    maximum_projected_superluminal_tail_probability=1.0e-6,
    maximum_jacobian_condition_number=1.0e12,
    electron_density_m3=None,
    electron_density_uncertainty_m3=None,
    electron_density_temperature_tensor_cross_covariance=None,
    scattering_vector_covariance=None,
    spectral_geometry_cross_covariance=None,
    max_nfev=5000,
):
    r"""Jointly infer a general PSD 3-D electron temperature tensor from raw TS spectra.

    Physical closure: T_q=q^T T_eV q and Var(v_q)=e*T_q/m_e. Maxwellian
    projections are Gaussian; Kappa projections are Student-t with nu=2*kappa-1
    and scale^2=sigma_v^2*(nu-2)/nu. The measured response acts on bin-integrated power.
    The selected Gaussian or Student-t probability is integrated exactly over
    finite wavelength bins, then the measured response acts on bin-integrated
    spectral power. This is a non-relativistic, elastic, single-scattering model
    family; the Kappa branch is an ellipsoidal parametric closure, not an
    unrestricted non-Gaussian 3-D EVDF reconstruction.
    """
    from scipy.optimize import least_squares

    wavelength = _finite_array(wavelength_nm, "wavelength_nm", 1)
    spectra = _finite_array(measured_spectra, "measured_spectra", 2)
    directions = _finite_array(scattering_vector_unit, "scattering_vector_unit", 2)
    n_view, n_wave = spectra.shape
    if wavelength.size != n_wave or wavelength.size < 12 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("measured_spectra must be (n_view,n_wavelength) on a strictly increasing wavelength grid")
    if directions.shape != (n_view, 3) or n_view < 6:
        raise ValueError("at least six measured views with paired 3-D scattering vectors are required")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms <= 0.0) or not np.allclose(norms, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("each scattering_vector_unit row must have unit norm")
    laser, angle = float(laser_wavelength_nm), float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0.0 or not np.isfinite(angle) or not 0.0 < angle < np.pi:
        raise ValueError("laser wavelength must be positive and scattering angle must lie in (0, pi)")
    if distribution_model not in {"maxwellian","kappa"}:
        raise ValueError("distribution_model must be 'maxwellian' or 'kappa'")
    if not isinstance(fit_kappa_index,(bool,np.bool_)):
        raise ValueError("fit_kappa_index must be boolean")
    kappa_seed=None
    kappa_lower=kappa_upper=None
    if distribution_model=="kappa":
        if initial_kappa_index is None:
            raise ValueError("Kappa tensor inference requires an explicit initial_kappa_index >1.5")
        kappa_seed=float(initial_kappa_index)
        if not np.isfinite(kappa_seed) or kappa_seed<=1.5:
            raise ValueError("initial_kappa_index must be finite and >1.5")
        if fit_kappa_index:
            if kappa_bounds is None:
                raise ValueError("fit_kappa_index=True requires explicit kappa_bounds")
            bounds=_finite_array(kappa_bounds,"kappa_bounds",1)
            if bounds.shape!=(2,) or bounds[0]<=1.5 or bounds[0]>=bounds[1]:
                raise ValueError("kappa_bounds must be an increasing pair with lower bound >1.5")
            if not bounds[0]<=kappa_seed<=bounds[1]:
                raise ValueError("initial_kappa_index must lie inside kappa_bounds")
            kappa_lower,kappa_upper=map(float,bounds)
    elif initial_kappa_index is not None or fit_kappa_index or kappa_bounds is not None:
        raise ValueError("Kappa controls are valid only when distribution_model='kappa'")
    q_factor = 2.0*np.sin(angle/2.0)
    dyads = np.column_stack((
        directions[:,0]**2, directions[:,1]**2, directions[:,2]**2,
        2*directions[:,0]*directions[:,1], 2*directions[:,0]*directions[:,2],
        2*directions[:,1]*directions[:,2],
    ))
    geom_singular = np.linalg.svd(dyads, compute_uv=False)
    geom_tol = np.finfo(float).eps*max(dyads.shape)*geom_singular[0]
    geom_rank = int(np.count_nonzero(geom_singular > geom_tol))
    geom_condition = float(geom_singular[0]/geom_singular[-1]) if geom_singular[-1] > 0 else float("inf")
    if geom_rank < 6:
        raise ValueError(f"general 3-D temperature tensor is not identifiable: view-dyad rank {geom_rank}/6")
    if not np.isfinite(geom_condition) or geom_condition > 1.0e6:
        raise ValueError(f"temperature-tensor view geometry is ill-conditioned ({geom_condition:.6g})")

    geometry_covariance = None
    geometry_cross_covariance = None
    if scattering_vector_covariance is not None:
        geometry_covariance = _finite_array(
            scattering_vector_covariance, "scattering_vector_covariance", 2
        )
        if geometry_covariance.shape != (3*n_view, 3*n_view):
            raise ValueError("scattering_vector_covariance must be (3*n_view,3*n_view) in view-major Cartesian order")
        if not np.allclose(geometry_covariance,geometry_covariance.T,rtol=1e-10,atol=1e-14):
            raise ValueError("scattering_vector_covariance must be symmetric")
        geometry_covariance = 0.5*(geometry_covariance+geometry_covariance.T)
        geometry_eigenvalues = np.linalg.eigvalsh(geometry_covariance)
        geometry_scale = max(float(np.max(np.abs(geometry_eigenvalues))), 1e-300)
        if geometry_eigenvalues[0] < -1e-10*geometry_scale:
            raise ValueError("scattering_vector_covariance must be positive semidefinite")
        tangent_projector = np.zeros((3*n_view,3*n_view))
        for view, direction in enumerate(directions):
            tangent = np.eye(3)-np.outer(direction,direction)
            tangent_projector[3*view:3*view+3,3*view:3*view+3] = tangent
        tangent_residual = geometry_covariance - tangent_projector@geometry_covariance@tangent_projector
        if np.linalg.norm(tangent_residual) > 1e-7*max(np.linalg.norm(geometry_covariance),1e-300):
            raise ValueError("scattering_vector_covariance must describe tangent-plane uncertainty of unit scattering directions")
    if spectral_geometry_cross_covariance is not None:
        if geometry_covariance is None:
            raise ValueError("spectral_geometry_cross_covariance requires scattering_vector_covariance")
        geometry_cross_covariance = _finite_array(
            spectral_geometry_cross_covariance, "spectral_geometry_cross_covariance", 2
        )
        if geometry_cross_covariance.shape != (n_view*n_wave,3*n_view):
            raise ValueError("spectral_geometry_cross_covariance must have shape (n_view*n_wave,3*n_view), channel-major by view-major Cartesian")
        if measurement_covariance is None and signal_uncertainty is None:
            raise ValueError("spectral_geometry_cross_covariance requires an absolute measurement covariance or signal_uncertainty")
        for view, direction in enumerate(directions):
            block = geometry_cross_covariance[:,3*view:3*view+3]
            if np.linalg.norm(block@direction) > 1e-7*max(np.linalg.norm(block),1e-300):
                raise ValueError("spectral_geometry_cross_covariance must lie in each unit-direction tangent plane")
    response = np.asarray(instrument_response_matrix, dtype=float)
    if response.shape == (n_wave, n_wave):
        response = np.broadcast_to(response, (n_view,n_wave,n_wave))
    if response.shape != (n_view,n_wave,n_wave) or not np.all(np.isfinite(response)) or np.any(response < 0.0):
        raise ValueError("instrument_response_matrix must be (n_wave,n_wave) or (n_view,n_wave,n_wave), finite and non-negative")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("each response matrix must conserve bin-integrated spectral power (columns sum to one)")
    edges = np.empty(n_wave+1, dtype=float)
    edges[1:-1] = 0.5*(wavelength[:-1]+wavelength[1:])
    edges[0] = wavelength[0]-0.5*(wavelength[1]-wavelength[0])
    edges[-1] = wavelength[-1]+0.5*(wavelength[-1]-wavelength[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength bins must have positive widths")

    n_data = n_view*n_wave
    covariance_parts = []
    covariance_components = {}
    measured_covariance_for_geometry = None
    if measurement_covariance is not None and signal_uncertainty is not None:
        raise ValueError("supply measurement_covariance or signal_uncertainty, not both")
    if measurement_covariance is not None:
        measured_covariance = _finite_array(measurement_covariance,"measurement_covariance",2)
        if measured_covariance.shape != (n_data,n_data):
            raise ValueError("measurement_covariance must be square in flattened channel-major order")
        scale = max(float(np.max(np.abs(measured_covariance))), 1e-300)
        if not np.allclose(measured_covariance, measured_covariance.T, rtol=1e-10, atol=1e-12*scale):
            raise ValueError("measurement_covariance must be symmetric")
        measured_covariance_for_geometry = 0.5*(measured_covariance+measured_covariance.T)
        covariance_parts.append(measured_covariance_for_geometry)
        covariance_components["measurement"] = measured_covariance_for_geometry
    if signal_uncertainty is not None:
        uncertainty = _finite_array(signal_uncertainty,"signal_uncertainty")
        if uncertainty.shape not in {(n_view,n_wave),(n_data,)} or np.any(uncertainty <= 0.0):
            raise ValueError("signal_uncertainty must be positive with shape (n_view,n_wave) or flattened channel-major")
        measured_covariance_for_geometry = np.diag(uncertainty.ravel()**2)
        covariance_parts.append(measured_covariance_for_geometry)
        covariance_components["measurement"] = measured_covariance_for_geometry
    if geometry_cross_covariance is not None:
        if measured_covariance_for_geometry is None or measured_covariance_for_geometry.shape != (n_data,n_data):
            raise ValueError("spectral_geometry_cross_covariance requires a valid absolute (n_view*n_wavelength)^2 measurement covariance")
        joint_measurement_geometry = np.block([
            [measured_covariance_for_geometry, geometry_cross_covariance],
            [geometry_cross_covariance.T, geometry_covariance],
        ])
        joint_measurement_geometry = 0.5*(
            joint_measurement_geometry+joint_measurement_geometry.T
        )
        joint_eigenvalues = np.linalg.eigvalsh(joint_measurement_geometry)
        joint_scale = max(float(np.max(np.abs(joint_eigenvalues))),1e-300)
        if joint_eigenvalues[0] < -1e-10*joint_scale:
            raise ValueError(
                "joint measured-spectrum/scattering-geometry covariance must be positive semidefinite"
            )
    for name, component in (("model_discrepancy_covariance", model_discrepancy_covariance),
                            ("systematic_covariance", systematic_covariance)):
        if component is None:
            continue
        matrix = _finite_array(component, name, 2)
        if matrix.shape != (n_data,n_data):
            raise ValueError(f"{name} must be square in flattened channel-major order")
        scale = max(float(np.max(np.abs(matrix))), 1e-300)
        if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12*scale):
            raise ValueError(f"{name} must be symmetric")
        matrix = 0.5*(matrix+matrix.T)
        eigenvalues = np.linalg.eigvalsh(matrix)
        if eigenvalues[0] < -1e-10*max(float(np.max(np.abs(eigenvalues))), 1e-300):
            raise ValueError(f"{name} must be positive semidefinite")
        covariance_parts.append(matrix)
        covariance_components["model_discrepancy" if name == "model_discrepancy_covariance" else "systematic"] = matrix
    if covariance_parts:
        if any(part.shape != (n_data,n_data) for part in covariance_parts):
            raise ValueError("every supplied covariance component must be (n_view*n_wavelength)^2 channel-major")
        covariance = np.sum(covariance_parts, axis=0)
        covariance = 0.5*(covariance+covariance.T)
        try:
            chol = np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("combined spectral covariance must be positive definite") from exc
        covariance_source = "measurement_plus_supplied_discrepancy_and_systematic_components"
    else:
        chol = np.eye(n_data)
        covariance_source = "identity_metric_no_absolute_spectral_uncertainty_supplied"

    if initial_temperature_tensor_eV is None:
        initial_tensor = np.eye(3)*max(float(np.median(np.ptp(spectra,axis=1))),100.0)
    else:
        initial_tensor = _finite_array(initial_temperature_tensor_eV,"initial_temperature_tensor_eV",2)
        if initial_tensor.shape != (3,3) or not np.allclose(initial_tensor,initial_tensor.T,rtol=1e-9,atol=1e-9):
            raise ValueError("initial_temperature_tensor_eV must be symmetric 3x3")
    try:
        initial_cholesky = np.linalg.cholesky(initial_tensor)
    except np.linalg.LinAlgError as exc:
        raise ValueError("initial_temperature_tensor_eV must be strictly positive definite") from exc
    tensor_seed = np.array([
        np.log(initial_cholesky[0,0]), initial_cholesky[1,0],
        np.log(initial_cholesky[1,1]), initial_cholesky[2,0],
        initial_cholesky[2,1], np.log(initial_cholesky[2,2]),
    ])
    if initial_electron_velocity_vector_m_s is None:
        velocity_seed = np.zeros(3, dtype=float)
    else:
        velocity_seed = _finite_array(
            initial_electron_velocity_vector_m_s,
            "initial_electron_velocity_vector_m_s", 1,
        )
        if velocity_seed.shape != (3,):
            raise ValueError("initial_electron_velocity_vector_m_s must be a Cartesian 3-vector")
    nuisance_seed=[]
    for row in spectra:
        edge_level=float(np.median(np.r_[row[:max(2,n_wave//12)],row[-max(2,n_wave//12):]]))
        peak=max(float(np.max(row)-edge_level),1e-12)
        nuisance_seed.extend([np.log(peak),edge_level,0.0])
    p0=np.r_[tensor_seed, velocity_seed, np.asarray(nuisance_seed).reshape(n_view,3).ravel()]
    if distribution_model=="kappa" and fit_kappa_index:
        p0=np.r_[p0,np.log(kappa_seed-1.5)]

    def unpack(parameters):
        p=np.asarray(parameters,dtype=float)
        l=np.array([[np.exp(p[0]),0.0,0.0],[p[1],np.exp(p[2]),0.0],[p[3],p[4],np.exp(p[5])]])
        return l@l.T

    def forward(parameters, view_directions=None):
        view_directions = directions if view_directions is None else np.asarray(view_directions,dtype=float)
        tensor=unpack(parameters)
        result=np.empty_like(spectra)
        projected=np.einsum("vi,ij,vj->v",view_directions,tensor,view_directions)
        if np.any(projected <= 0.0):
            raise ValueError("projected temperature tensor must be positive definite")
        velocity_vector = parameters[6:9]
        flow_projections = view_directions @ velocity_vector
        for view in range(n_view):
            log_amp, offset, slope = parameters[9+3*view:12+3*view]
            centre=laser*(1.0-q_factor*flow_projections[view]/299792458.0)
            sigma=laser*q_factor*np.sqrt(_E*projected[view]/_M_E)/299792458.0
            if distribution_model=="maxwellian":
                density, probability, _ = _tensor_finite_bin_gaussian_density(wavelength,centre,sigma)
            else:
                kappa_value = 1.5+np.exp(parameters[-1]) if fit_kappa_index else kappa_seed
                density, probability, _ = _tensor_finite_bin_kappa_density(wavelength,centre,sigma,kappa_value)
            captured=float(np.sum(probability))
            if captured <= 0.0 or not np.isfinite(captured):
                raise ValueError("spectral wavelength window captures no physical Thomson signal")
            # The measured operator acts on finite-bin integrated spectral power.
            convolved=(response[view]@(density*widths))/widths
            baseline=offset+slope*(wavelength-laser)/max(float(np.ptp(wavelength)),1e-12)
            result[view]=np.exp(log_amp)*convolved+baseline
        return result

    observed=spectra.ravel()
    def residual(parameters):
        delta=(forward(parameters)-spectra).ravel()
        return np.linalg.solve(chol,delta)

    flow_limit=float(maximum_flow_beta)
    beta_limit=float(maximum_projected_rms_beta)
    tail_probability_limit=float(maximum_projected_superluminal_tail_probability)
    if not 0.0<beta_limit<1.0 or not 0.0<flow_limit<1.0:
        raise ValueError("relativistic validity limits must lie strictly between zero and one")
    if not np.isfinite(tail_probability_limit) or not 0.0<tail_probability_limit<1.0:
        raise ValueError("maximum_projected_superluminal_tail_probability must lie strictly between zero and one")
    if np.any(np.abs(velocity_seed) > flow_limit*299792458.0):
        raise ValueError("initial_electron_velocity_vector_m_s exceeds maximum_flow_beta")
    lower=np.full(p0.size,-np.inf); upper=np.full(p0.size,np.inf)
    lower[0]=lower[2]=lower[5]=-10.0
    upper[0]=upper[2]=upper[5]=10.0
    lower[6:9]=-flow_limit*299792458.0
    upper[6:9]= flow_limit*299792458.0
    for view in range(n_view):
        lower[9+3*view]=-50.0; upper[9+3*view]=50.0
    if distribution_model=="kappa" and fit_kappa_index:
        lower[-1]=np.log(kappa_lower-1.5)
        upper[-1]=np.log(kappa_upper-1.5)
    fit=least_squares(residual,p0,bounds=(lower,upper),method="trf",x_scale="jac",max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"general temperature-tensor spectral fit failed: {fit.message}")
    jac=np.asarray(fit.jac,dtype=float)
    singular=np.linalg.svd(jac,compute_uv=False)
    tolerance=np.finfo(float).eps*max(jac.shape)*(singular[0] if singular.size else 0.0)
    rank=int(np.count_nonzero(singular>tolerance))
    condition=float(singular[0]/singular[-1]) if singular.size and singular[-1]>0 else float("inf")
    if rank<p0.size or not np.isfinite(condition) or condition>float(maximum_jacobian_condition_number):
        raise ValueError(f"general temperature-tensor spectral fit is not identifiable (rank {rank}/{p0.size}, condition={condition:.6g})")
    tensor=unpack(fit.x)
    eigenvalues,eigenvectors=np.linalg.eigh(tensor)
    if eigenvalues[0] <= 0.0:
        raise ValueError("fitted temperature tensor reached a singular PSD boundary; symmetric covariance is withheld")
    fitted=forward(fit.x)
    raw_residual=spectra-fitted
    chi_square=float(fit.fun@fit.fun)
    dof=n_data-p0.size
    parameter_covariance=np.linalg.inv(jac.T@jac)
    covariance_status="absolute_GLS_covariance_conditional_on_supplied_spectral_covariance"
    # Decompose the spectral contribution to parameter uncertainty by its
    # physical source.  For the whitened GLS estimator
    # A=(J_w^T J_w)^-1 J_w^T, each independent covariance component C_k
    # contributes A L^-1 C_k L^-T A^T.  The components sum to the
    # measurement+discrepancy+systematic GLS covariance before geometry terms.
    parameter_covariance_components = {}
    if covariance_parts:
        estimator = np.linalg.solve(jac.T @ jac, jac.T)
        for component_name, component_covariance in covariance_components.items():
            whitened_component = np.linalg.solve(chol, component_covariance)
            whitened_component = np.linalg.solve(chol, whitened_component.T).T
            contribution = estimator @ whitened_component @ estimator.T
            contribution = 0.5 * (contribution + contribution.T)
            eig = np.linalg.eigvalsh(contribution)
            scale = max(float(np.max(np.abs(eig))), 1e-300)
            if eig[0] < -1e-9 * scale:
                raise ArithmeticError(
                    f"{component_name} parameter-covariance contribution is not positive semidefinite"
                )
            parameter_covariance_components[component_name] = contribution
    geometry_parameter_covariance = None
    geometry_cross_parameter_covariance = None
    geometry_jacobian = None
    if geometry_covariance is not None:
        # Differentiate the physical forward model with respect to tangent
        # perturbations of each measured unit scattering vector. The geometry
        # covariance is propagated through the same whitened GLS estimator as
        # the spectral data; it is not folded into a fictitious temperature error.
        geometry_jacobian = np.zeros((n_data,3*n_view),dtype=float)
        for view in range(n_view):
            direction = directions[view]
            for component in range(3):
                tangent_step = np.eye(3)[component]-direction[component]*direction
                norm_step = float(np.linalg.norm(tangent_step))
                if norm_step <= 1e-14:
                    continue
                step = np.cbrt(np.finfo(float).eps)
                plus_directions = directions.copy()
                minus_directions = directions.copy()
                plus_directions[view] = direction + step*tangent_step
                minus_directions[view] = direction - step*tangent_step
                plus_directions[view] /= np.linalg.norm(plus_directions[view])
                minus_directions[view] /= np.linalg.norm(minus_directions[view])
                geometry_jacobian[:,3*view+component] = (
                    (forward(fit.x,plus_directions)-forward(fit.x,minus_directions)).ravel()
                    /(2.0*step)
                )
        whitened_geometry_jacobian = np.linalg.solve(chol,geometry_jacobian)
        estimator = np.linalg.solve(jac.T@jac,jac.T)
        geometry_parameter_sensitivity = -estimator@whitened_geometry_jacobian
        geometry_parameter_covariance = (
            geometry_parameter_sensitivity@geometry_covariance
            @geometry_parameter_sensitivity.T
        )
        parameter_covariance += geometry_parameter_covariance
        covariance_status += "+tangent_plane_scattering_geometry_covariance"
        if geometry_cross_covariance is not None:
            # The supplied cross block is Cov(measured spectrum, q geometry).
            # Whiten it with the same total spectral covariance used by GLS.
            whitened_cross = np.linalg.solve(chol,geometry_cross_covariance)
            geometry_cross_parameter_covariance = (
                estimator@whitened_cross@geometry_parameter_sensitivity.T
            )
            geometry_cross_parameter_covariance += geometry_cross_parameter_covariance.T
            parameter_covariance += geometry_cross_parameter_covariance
            covariance_status += "+measured_spectrum_geometry_cross_covariance"
        parameter_covariance = 0.5*(parameter_covariance+parameter_covariance.T)
        covariance_eigenvalues = np.linalg.eigvalsh(parameter_covariance)
        covariance_scale = max(float(np.max(np.abs(covariance_eigenvalues))),1e-300)
        if covariance_eigenvalues[0] < -1e-9*covariance_scale:
            raise ValueError("propagated spectral/geometry parameter covariance is not positive semidefinite")
        if covariance_eigenvalues[0] < 0.0:
            values,vectors=np.linalg.eigh(parameter_covariance)
            parameter_covariance=(vectors*np.maximum(values,0.0))@vectors.T
    if not covariance_parts:
        if dof<=0:
            parameter_covariance=None
            covariance_status="not_reported_without_absolute_covariance_or_redundant_spectral_bins"
        else:
            parameter_covariance*=chi_square/dof
            covariance_status="residual_scaled_covariance_not_absolute_measurement_uncertainty"
    tensor_covariance=None
    tensor_flow_joint_covariance=None
    flow_vector_covariance=None
    if parameter_covariance is not None:
        tensor_jac=np.empty((6,6),dtype=float)
        for j in range(6):
            step=np.cbrt(np.finfo(float).eps)*max(1.0,abs(float(fit.x[j])))
            plus,minus=fit.x.copy(),fit.x.copy()
            plus[j]+=step; minus[j]-=step
            def components(mat):
                return np.array([mat[0,0],mat[1,1],mat[2,2],mat[0,1],mat[0,2],mat[1,2]])
            tensor_jac[:,j]=(components(unpack(plus))-components(unpack(minus)))/(2*step)
        transform=np.zeros((9,p0.size),dtype=float)
        transform[:6,:6]=tensor_jac
        transform[6:9,6:9]=np.eye(3)
        tensor_flow_joint_covariance=transform@parameter_covariance@transform.T
        tensor_covariance=tensor_flow_joint_covariance[:6,:6]
        flow_vector_covariance=tensor_flow_joint_covariance[6:9,6:9]
    projected=np.einsum("vi,ij,vj->v",directions,tensor,directions)
    density=None
    density_uncertainty=None
    pressure_tensor=None
    pressure_covariance=None
    pressure_covariance_status="not_available_without_independent_electron_density"
    tensor_components=np.array([tensor[0,0],tensor[1,1],tensor[2,2],tensor[0,1],tensor[0,2],tensor[1,2]])
    if electron_density_m3 is None and (
        electron_density_uncertainty_m3 is not None
        or electron_density_temperature_tensor_cross_covariance is not None
    ):
        raise ValueError("density uncertainty/cross-covariance requires electron_density_m3")
    if electron_density_m3 is not None:
        density=float(electron_density_m3)
        if not np.isfinite(density) or density<=0.0:
            raise ValueError("electron_density_m3 must be finite and strictly positive")
        if electron_density_uncertainty_m3 is not None:
            density_uncertainty=float(electron_density_uncertainty_m3)
            if not np.isfinite(density_uncertainty) or density_uncertainty<0.0:
                raise ValueError("electron_density_uncertainty_m3 must be finite and non-negative")
        density_tensor_cross = None
        if electron_density_temperature_tensor_cross_covariance is not None:
            density_tensor_cross = _finite_array(
                electron_density_temperature_tensor_cross_covariance,
                "electron_density_temperature_tensor_cross_covariance", 1,
            )
            if density_tensor_cross.shape != (6,):
                raise ValueError(
                    "electron_density_temperature_tensor_cross_covariance must contain "
                    "six covariances ordered as T_xx,T_yy,T_zz,T_xy,T_xz,T_yz"
                )
            if density_uncertainty is None or tensor_covariance is None:
                raise ValueError(
                    "density-temperature cross-covariance requires both density uncertainty "
                    "and an available temperature-tensor covariance"
                )
        pressure_tensor=density*_E*tensor
        if tensor_covariance is not None:
            pressure_covariance=(density*_E)**2*tensor_covariance
            if density_uncertainty is not None:
                density_variance=density_uncertainty**2
                cross=np.zeros(6) if density_tensor_cross is None else density_tensor_cross
                joint_input_covariance=np.block([
                    [np.array([[density_variance]]), cross[None,:]],
                    [cross[:,None], tensor_covariance],
                ])
                joint_eigenvalues=np.linalg.eigvalsh(
                    0.5*(joint_input_covariance+joint_input_covariance.T)
                )
                joint_scale=max(float(np.max(np.abs(joint_eigenvalues))),1e-300)
                if joint_eigenvalues[0] < -1e-10*joint_scale:
                    raise ValueError(
                        "joint density-temperature covariance must be positive semidefinite"
                    )
                pressure_covariance += (_E**2)*density_variance*np.outer(tensor_components,tensor_components)
                pressure_covariance += (_E**2)*density*(
                    np.outer(tensor_components,cross)+np.outer(cross,tensor_components)
                )
                pressure_covariance_status=(
                    "joint_density_temperature_cross_covariance_delta_method"
                    if density_tensor_cross is not None
                    else "independent_tensor_GLS_and_density_uncertainty_delta_method"
                )
            else:
                pressure_covariance_status="tensor_GLS_only_density_treated_as_exact"
    thermal_beta=float(np.sqrt(_E*np.max(eigenvalues)/_M_E)/299792458.0)
    velocity_vector=fit.x[6:9].copy()
    flows=directions@velocity_vector
    flow_beta=float(np.linalg.norm(velocity_vector)/299792458.0)
    projected_superluminal_tail_probability=np.zeros(n_view,dtype=float)
    if distribution_model=="kappa":
        from scipy.special import stdtr
        fitted_kappa=float(1.5+np.exp(fit.x[-1]) if fit_kappa_index else kappa_seed)
        degrees=2.0*fitted_kappa-1.0
        for view in range(n_view):
            projected_sigma_v=np.sqrt(_E*projected[view]/_M_E)
            projected_scale_v=projected_sigma_v*np.sqrt((degrees-2.0)/degrees)
            flow_projection=float(flows[view])
            projected_superluminal_tail_probability[view]=(
                stdtr(degrees,(-299792458.0-flow_projection)/projected_scale_v)
                +stdtr(degrees,(flow_projection-299792458.0)/projected_scale_v)
            )
    maximum_superluminal_tail_probability=float(np.max(projected_superluminal_tail_probability))
    within_validity=(
        thermal_beta<beta_limit and flow_beta<flow_limit
        and (distribution_model!="kappa" or maximum_superluminal_tail_probability<=tail_probability_limit)
    )
    return {
        "model":f"nonrelativistic_non_gyrotropic_3d_{distribution_model}_thomson_joint_spectral_fit",
        "distribution_model":distribution_model,
        "kappa_index":(1.5+np.exp(fit.x[-1]) if distribution_model=="kappa" and fit_kappa_index else kappa_seed),
        "kappa_index_uncertainty":(float(np.exp(fit.x[-1])*np.sqrt(max(parameter_covariance[-1,-1],0.0))) if distribution_model=="kappa" and fit_kappa_index and parameter_covariance is not None else None),
        "kappa_fit_status":("fitted_with_full_spectral_jacobian" if distribution_model=="kappa" and fit_kappa_index else "fixed_explicit_input" if distribution_model=="kappa" else "not_applicable"),
        "electron_temperature_tensor_eV":tensor,
        "principal_temperature_eigenvalues_eV":eigenvalues,
        "principal_axes_cartesian_columns":eigenvectors,
        "projected_temperature_eV":projected,
        "projected_temperature_equation":"T_q_eV = q_hat.T @ T_e_eV @ q_hat",
        "electron_velocity_vector_m_s":velocity_vector,
        "electron_velocity_vector_covariance_m2_s2":flow_vector_covariance,
        "electron_temperature_flow_joint_covariance":tensor_flow_joint_covariance,
        "electron_temperature_flow_joint_covariance_order":[
            "T_xx_eV","T_yy_eV","T_zz_eV","T_xy_eV","T_xz_eV","T_yz_eV",
            "u_x_m_s","u_y_m_s","u_z_m_s",
        ],
        "electron_velocity_projection_m_s":flows,
        "flow_projection_equation":"u_q_i_m_s = dot(scattering_vector_unit_i, electron_velocity_vector_m_s)",
        "fitted_spectra":fitted,
        "residual_spectra":raw_residual,
        "chi_square":chi_square,
        "degrees_of_freedom":dof,
        "reduced_chi_square":chi_square/dof if dof>0 else None,
        "jacobian_rank":rank,
        "jacobian_parameter_count":int(p0.size),
        "jacobian_condition_number":condition,
        "view_dyad_rank":geom_rank,
        "view_dyad_condition_number":geom_condition,
        "parameter_covariance":parameter_covariance,
        "spectral_parameter_covariance_components":parameter_covariance_components,
        "spectral_parameter_covariance_component_order":list(parameter_covariance_components),
        "spectral_parameter_covariance_decomposition_status":(
            "separated_GLS_sandwich_contributions_before_geometry_terms"
            if parameter_covariance_components else "not_available_without_supplied_spectral_covariance_components"
        ),
        "model_discrepancy_parameter_covariance":parameter_covariance_components.get("model_discrepancy"),
        "systematic_parameter_covariance":parameter_covariance_components.get("systematic"),
        "measurement_parameter_covariance":parameter_covariance_components.get("measurement"),
        "model_selection_uncertainty_included":False,
        "model_selection_uncertainty_status":"not_included; requires evidence-weighted comparison across distinct physical model families",
        "scattering_vector_covariance_used":geometry_covariance is not None,
        "spectral_geometry_cross_covariance_used":geometry_cross_covariance is not None,
        "geometry_parameter_covariance":geometry_parameter_covariance,
        "geometry_cross_parameter_covariance":geometry_cross_parameter_covariance,
        "geometry_covariance_order":[f"q{axis}_view_{view}" for view in range(n_view) for axis in ("x","y","z")],
        "geometry_covariance_method":"finite_difference_forward_sensitivity_in_unit_vector_tangent_planes_with_GLS_Jacobian_propagation" if geometry_covariance is not None else "not_supplied",
        "parameter_covariance_order":[
            "L_xx_log","L_yx","L_yy_log","L_zx","L_zy","L_zz_log",
            "u_x_m_s","u_y_m_s","u_z_m_s",
        ]+[name for i in range(n_view) for name in (f"log_amplitude_{i}",f"background_offset_{i}",f"background_slope_{i}")]+(["log_kappa_minus_1p5"] if distribution_model=="kappa" and fit_kappa_index else []),
        "temperature_tensor_parameter_covariance_eV2":tensor_covariance,
        "electron_density_m3":density,
        "electron_density_uncertainty_m3":density_uncertainty,
        "electron_pressure_tensor_Pa":pressure_tensor,
        "electron_pressure_parameter_covariance_Pa2":pressure_covariance,
        "electron_pressure_covariance_status":pressure_covariance_status,
        "electron_density_temperature_tensor_cross_covariance":density_tensor_cross if density is not None else None,
        "electron_pressure_covariance_equation":"Cov(P_i,P_j)=e^2[n_e^2 Cov(T_i,T_j)+sigma_n^2 T_i T_j+n_e(T_i Cov(n,T_j)+Cov(n,T_i) T_j)]",
        "electron_pressure_equation":"P_e_Pa = n_e_m-3 * elementary_charge_C * T_e_eV",
        "covariance_provenance":covariance_source,
        "covariance_status":covariance_status,
        "thermal_rms_beta":thermal_beta,
        "maximum_flow_beta":flow_beta,
        "projected_superluminal_tail_probability_per_view":projected_superluminal_tail_probability,
        "maximum_projected_superluminal_tail_probability":maximum_superluminal_tail_probability,
        "flow_model":"one_shared_cartesian_3d_bulk_velocity_vector_projected onto each measured scattering direction",
        "model_validity_status":"within_declared_nonrelativistic_domain" if within_validity else "outside_declared_nonrelativistic_domain",
        "model_validity_limits":{"maximum_principal_thermal_rms_beta":beta_limit,"maximum_flow_beta":flow_limit,"maximum_projected_superluminal_tail_probability":tail_probability_limit,"relativistic_corrections":False},
        "reportable_as_inferred_parameter":bool(within_validity),
        "primary_result_status":"identified_from_measurement" if within_validity else "withheld_outside_nonrelativistic_validity_domain",
        "physical_scope":f"raw measured multi-view spectra; general symmetric positive-definite temperature tensor; {distribution_model} projected closure; no gyrotropy assumption; not an unrestricted measured 3-D EVDF",
        "spectral_forward_model":("exact finite-wavelength-bin Gaussian probability" if distribution_model=="maxwellian" else "exact finite-wavelength-bin Student-t probability with nu=2*kappa-1 and second-moment-normalized scale")+"; measured response on bin-integrated power; view-specific flow, amplitude and affine background",
        "experimental_validation":"synthetic_physics_recovery_only",
        "optimizer":{"success":bool(fit.success),"message":str(fit.message),"nfev":int(fit.nfev),"cost":float(fit.cost)},
    }


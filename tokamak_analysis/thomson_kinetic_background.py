"""Native structured-background fitting for non-collective kinetic Thomson models.

The physical forward model, calibrated response, and detector-space nuisance
basis are fitted simultaneously. This helper is deliberately shape-only:
absolute density is not inferred from normalized or freely-amplitude-scaled
spectra. It complements, rather than silently changes, the legacy channel-wise
fitters until their UI contracts are migrated.
"""
from __future__ import annotations

import numpy as np

_C = 299_792_458.0
_E_REST_EV = 510_998.95069


def _real_vector(value, name, minimum=7):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size < minimum:
        raise ValueError(f"{name} must be a real one-dimensional vector with at least {minimum} entries")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite values")
    return out


def _apply_response(response, spectrum, axis):
    x = np.asarray(axis, dtype=float)
    y = np.asarray(spectrum, dtype=float)
    r = np.asarray(response, dtype=float)
    if r.shape != (x.size, x.size) or y.shape != x.shape or np.any(~np.isfinite(r)) or np.any(r < 0):
        raise ValueError("instrument_response must be a finite non-negative square matrix on the measured wavelength grid")
    if not np.allclose(r.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument-response columns must already sum to one")
    edges = np.empty(x.size + 1)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("wavelength grid must define positive bin widths")
    return (r @ (y * widths)) / widths


def fit_noncollective_kinetic_thomson_with_background(
    wavelength_nm, measured_spectrum, *, physical_model,
    laser_wavelength_nm, scattering_angle_rad, instrument_response,
    physical_initial, background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    measurement_uncertainty=None, measurement_covariance=None,
    incident_polarization=None, radial_order=36, angular_order=18,
    maximum_recoil_correction_fraction=0.01, include_compton_recoil=False,
    physical_bounds=None, max_nfev=5000,
):
    """Fit a Kappa or relativistic Maxwell-Juttner optical spectrum plus structured background.

    physical_initial order is [Te_eV, flow_projection_m_s, kappa] for Kappa,
    and [Te_eV, flow_projection_m_s] for Maxwell-Juttner. Amplitude is profiled
    linearly with the background coefficients. Kappa is the finite-variance
    isotropic Student-t projection; it is not the generalized Skolar plasma
    dispersion-function method. MJ uses the polarized single-particle forward
    model and its explicit Thomson/Compton recoil contract.
    """
    from scipy.optimize import least_squares
    from .thomson_background import _background_design
    from .thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    x = _real_vector(wavelength_nm, "wavelength_nm")
    y = _real_vector(measured_spectrum, "measured_spectrum")
    if y.shape != x.shape or np.any(np.diff(x) <= 0) or np.any(x <= 0) or np.any(y < 0):
        raise ValueError("measured spectrum must be non-negative and match a positive increasing wavelength grid")
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0 or not np.isfinite(angle) or not 0 < angle < np.pi:
        raise ValueError("laser_wavelength_nm must be positive and scattering_angle_rad must lie in (0, pi)")
    if physical_model not in {"kappa", "relativistic_maxwell_juttner"}:
        raise ValueError("physical_model must be 'kappa' or 'relativistic_maxwell_juttner'")
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    p0 = _real_vector(physical_initial, "physical_initial", minimum=2)
    expected = 3 if physical_model == "kappa" else 2
    if p0.size != expected:
        raise ValueError(f"physical_initial must contain {expected} values for {physical_model}")
    if p0[0] <= 0 or abs(p0[1]) >= _C:
        raise ValueError("initial Te must be positive and flow projection subluminal")
    if physical_model == "kappa" and p0[2] <= 1.5:
        raise ValueError("finite-variance Kappa index must exceed 1.5")
    if physical_model == "relativistic_maxwell_juttner" and incident_polarization is None:
        raise ValueError("incident_polarization is required for relativistic Maxwell-Juttner inference")
    response = np.asarray(instrument_response, dtype=float)
    # Validate response independently before optimization.
    _apply_response(response, np.ones_like(x), x)
    basis = _background_design(
        x, background_model, degree=background_degree,
        interior_knots=background_interior_knots, templates=background_templates,
    )

    if physical_bounds is None:
        lower = np.r_[1e-4, -0.95 * _C, 1.500001] if physical_model == "kappa" else np.array([1e-4, -0.95 * _C])
        upper = np.r_[2e6, 0.95 * _C, 100.0] if physical_model == "kappa" else np.array([2e6, 0.95 * _C])
    else:
        lower, upper = (np.asarray(v, dtype=float) for v in physical_bounds)
        if lower.shape != p0.shape or upper.shape != p0.shape or np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("physical_bounds must be two finite vectors matching physical_initial")
    if np.any(lower >= upper) or np.any(p0 < lower) or np.any(p0 > upper):
        raise ValueError("physical initial values must lie within strictly ordered physical_bounds")

    if measurement_uncertainty is not None and measurement_covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    if measurement_uncertainty is not None:
        sigma = _real_vector(measurement_uncertainty, "measurement_uncertainty")
        if sigma.shape != y.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match measured_spectrum")
        whiten = lambda a: np.asarray(a) / sigma
        noise_model = "supplied_pointwise_measurement_uncertainty"
    elif measurement_covariance is not None:
        cov = np.asarray(measurement_covariance, dtype=float)
        if cov.shape != (x.size, x.size) or np.any(~np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-13):
            raise ValueError("measurement_covariance must be finite, symmetric and wavelength-sized")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement_covariance must be positive definite") from exc
        whiten = lambda a: np.linalg.solve(chol, np.asarray(a))
        noise_model = "supplied_full_spectral_covariance"
    else:
        whiten = lambda a: np.asarray(a)
        noise_model = "unweighted_no_measurement_uncertainty_supplied"

    q = 2.0 * np.sin(angle / 2.0)

    def intrinsic(parameters):
        te, flow = float(parameters[0]), float(parameters[1])
        if physical_model == "kappa":
            kappa = float(parameters[2])
            center = laser * (1.0 - q * flow / _C)
            sigma_nm = laser * np.sqrt(2.0) * np.sin(angle / 2.0) * np.sqrt(te / _E_REST_EV)
            df = 2.0 * kappa - 1.0
            scale = sigma_nm * np.sqrt((df - 2.0) / df)
            return (1.0 + ((x - center) / scale) ** 2 / df) ** (-kappa)
        result = relativistic_maxwell_juttner_spectrum(
            x, laser, angle, te, flow_projection_m_s=flow,
            incident_polarization=np.asarray(incident_polarization, dtype=float),
            radial_order=radial_order, angular_order=angular_order,
            maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
            include_compton_recoil=include_compton_recoil,
            require_full_spectrum=False,
        )
        return np.asarray(result["spectral_density_per_nm"], dtype=float)

    def profiled(parameters):
        shape = _apply_response(response, intrinsic(parameters), x)
        design = np.column_stack((shape, basis))
        a = whiten(design)
        b = whiten(y)
        coefficients = np.linalg.lstsq(a, b, rcond=None)[0]
        if coefficients[0] < 0:
            coefficients[0] = 0.0
            coefficients[1:] = np.linalg.lstsq(whiten(basis), b, rcond=None)[0]
        fitted = design @ coefficients
        return coefficients, fitted, shape

    def residual(parameters):
        _, fitted, _ = profiled(parameters)
        return whiten(fitted - y)

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf", x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise ValueError(f"kinetic Thomson/background fit failed: {fit.message}")
    jac = np.asarray(fit.jac, dtype=float)
    singular = np.linalg.svd(jac, compute_uv=False)
    tolerance = np.finfo(float).eps * max(jac.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tolerance))
    condition = float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    if rank < p0.size or not np.isfinite(condition) or condition > 1e12:
        raise ValueError(f"{physical_model} physical parameters are not jointly identifiable after profiling background (rank {rank}/{p0.size}, condition={condition:.6g})")
    coefficients, fitted, shape = profiled(fit.x)
    # The variable-projection Jacobian is the profiled physical Jacobian.
    physical_covariance = np.linalg.pinv(jac.T @ jac, rcond=1e-12)
    dof = max(1, x.size - p0.size - basis.shape[1] - 1)
    if noise_model == "unweighted_no_measurement_uncertainty_supplied":
        physical_covariance *= float(fit.fun @ fit.fun / dof)
        covariance_status = "residual_scaled_unweighted_conditional_covariance_not_measurement_uncertainty"
    else:
        covariance_status = "local_profiled_GLS_covariance_conditional_on_supplied_measurement_covariance"
    amplitude = float(coefficients[0])
    background_coefficients = np.asarray(coefficients[1:], dtype=float)
    background = basis @ background_coefficients
    fitted_signal = amplitude * shape
    return {
        "model": f"non_collective_{physical_model}_with_structured_detector_background",
        "physical_model": physical_model,
        "physical_parameter_names": ["electron_temperature_eV", "electron_velocity_projection_m_s"] + (["kappa_index"] if physical_model == "kappa" else []),
        "physical_parameters": fit.x.copy(),
        "electron_temperature_eV": float(fit.x[0]),
        "electron_velocity_projection_m_s": float(fit.x[1]),
        "kappa_index": float(fit.x[2]) if physical_model == "kappa" else None,
        "amplitude": amplitude,
        "background_model": background_model,
        "background_coefficients": background_coefficients,
        "fitted_thomson_spectrum": fitted_signal,
        "fitted_background": background,
        "fitted_spectrum": fitted,
        "residual": y - fitted,
        "physical_parameter_covariance": physical_covariance,
        "physical_parameter_standard_uncertainty": np.sqrt(np.maximum(np.diag(physical_covariance), 0.0)),
        "jacobian_rank": rank,
        "jacobian_condition_number": condition,
        "optimizer_success": bool(fit.success),
        "optimizer_message": str(fit.message),
        "noise_model": noise_model,
        "covariance_status": covariance_status,
        "uncertainty_semantics": "Profiled physical-parameter covariance conditional on the selected background basis and fixed instrument response; background/physical cross-covariance is not separately returned by this variable-projection API.",
        "absolute_density_inferred": False,
        "recoil_included": bool(include_compton_recoil) if physical_model == "relativistic_maxwell_juttner" else False,
        "cross_section_model": ("polarized_single_scattering_Klein_Nishina" if include_compton_recoil else "polarized_elastic_Thomson") if physical_model == "relativistic_maxwell_juttner" else "not_applicable_nonrelativistic_kappa_projection",
        "experimental_validation": "not_performed",
    }


def fit_noncollective_kinetic_thomson_with_joint_background(*args, **kwargs):
    """Joint physical, amplitude and structured-background fit with full covariance.

    This entry point uses the same physical forward models and input contract
    as fit_noncollective_kinetic_thomson_with_background, but optimizes all
    nuisance coefficients explicitly so the returned covariance contains the
    physical/background cross blocks.
    """
    from scipy.optimize import least_squares
    from .thomson_background import _background_design
    from .thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    (wavelength_nm, measured_spectrum) = args[:2] if len(args) >= 2 else (kwargs.pop("wavelength_nm"), kwargs.pop("measured_spectrum"))
    kw = dict(kwargs)
    model = kw.pop("physical_model")
    laser = float(kw.pop("laser_wavelength_nm"))
    angle = float(kw.pop("scattering_angle_rad"))
    response = np.asarray(kw.pop("instrument_response"), dtype=float)
    p0 = np.asarray(kw.pop("physical_initial"), dtype=float)
    bg_model = kw.pop("background_model", "linear")
    bg_degree = kw.pop("background_degree", 2)
    bg_knots = kw.pop("background_interior_knots", 3)
    bg_templates = kw.pop("background_templates", None)
    uncertainty = kw.pop("measurement_uncertainty", None)
    covariance = kw.pop("measurement_covariance", None)
    polarization = kw.pop("incident_polarization", None)
    radial_order = kw.pop("radial_order", 36)
    angular_order = kw.pop("angular_order", 18)
    recoil_limit = kw.pop("maximum_recoil_correction_fraction", 0.01)
    recoil = kw.pop("include_compton_recoil", False)
    bounds = kw.pop("physical_bounds", None)
    max_nfev = kw.pop("max_nfev", 5000)
    if kw:
        raise TypeError("unexpected keyword arguments: " + ", ".join(sorted(kw)))
    x = _real_vector(wavelength_nm, "wavelength_nm")
    y = _real_vector(measured_spectrum, "measured_spectrum")
    if y.shape != x.shape or np.any(np.diff(x) <= 0) or np.any(x <= 0) or np.any(y < 0):
        raise ValueError("measured spectrum must be non-negative and match a positive increasing wavelength grid")
    if model not in {"kappa", "relativistic_maxwell_juttner"}:
        raise ValueError("physical_model must be kappa or relativistic_maxwell_juttner")
    nphys = 3 if model == "kappa" else 2
    if p0.shape != (nphys,) or np.any(~np.isfinite(p0)) or p0[0] <= 0 or abs(p0[1]) >= _C:
        raise ValueError("physical_initial must contain finite positive Te and subluminal flow, plus Kappa index when selected")
    if model == "kappa" and p0[2] <= 1.5:
        raise ValueError("Kappa index must exceed 1.5")
    if model == "relativistic_maxwell_juttner" and polarization is None:
        raise ValueError("incident_polarization is required for relativistic Maxwell-Juttner")
    if not isinstance(recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if uncertainty is not None and covariance is not None:
        raise ValueError("supply measurement_uncertainty or measurement_covariance, not both")
    if uncertainty is not None:
        sigma = _real_vector(uncertainty, "measurement_uncertainty")
        if sigma.shape != y.shape or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match the spectrum")
        whiten = lambda z: np.asarray(z) / sigma
        noise = "supplied_pointwise_measurement_uncertainty"
    elif covariance is not None:
        cov = np.asarray(covariance, dtype=float)
        if cov.shape != (x.size, x.size) or np.any(~np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-13):
            raise ValueError("measurement_covariance must be finite symmetric and wavelength-sized")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement_covariance must be positive definite") from exc
        whiten = lambda z: np.linalg.solve(chol, np.asarray(z))
        noise = "supplied_full_spectral_covariance"
    else:
        whiten = lambda z: np.asarray(z)
        noise = "unweighted_no_measurement_uncertainty_supplied"
    basis = _background_design(x, bg_model, degree=bg_degree, interior_knots=bg_knots, templates=bg_templates)
    if response.shape != (x.size, x.size) or np.any(~np.isfinite(response)) or np.any(response < 0) or not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("instrument_response must be a measured non-negative column-normalized square matrix")
    if bounds is None:
        lo = np.r_[1e-4, -0.95*_C, 1.500001] if model == "kappa" else np.array([1e-4, -0.95*_C])
        hi = np.r_[2e6, 0.95*_C, 100.0] if model == "kappa" else np.array([2e6, 0.95*_C])
    else:
        lo, hi = (np.asarray(v, dtype=float) for v in bounds)
    if lo.shape != p0.shape or hi.shape != p0.shape or np.any(lo >= hi) or np.any(p0 < lo) or np.any(p0 > hi):
        raise ValueError("physical_bounds must match physical_initial and contain the initial values")
    q = 2*np.sin(angle/2)
    def shape(p):
        if model == "kappa":
            te, flow, kap = p
            center = laser*(1-q*flow/_C)
            sigma_nm = laser*np.sqrt(2)*np.sin(angle/2)*np.sqrt(te/_E_REST_EV)
            df = 2*kap-1
            scale = sigma_nm*np.sqrt((df-2)/df)
            intrinsic = (1+((x-center)/scale)**2/df)**(-kap)
        else:
            physical = relativistic_maxwell_juttner_spectrum(
                x, laser, angle, float(p[0]), flow_projection_m_s=float(p[1]),
                incident_polarization=np.asarray(polarization, dtype=float),
                radial_order=radial_order, angular_order=angular_order,
                maximum_recoil_correction_fraction=recoil_limit,
                include_compton_recoil=recoil, require_full_spectrum=False)
            intrinsic = np.asarray(physical["spectral_density_per_nm"], dtype=float)
        edges = np.empty(x.size+1)
        edges[1:-1] = 0.5*(x[:-1]+x[1:])
        edges[0] = x[0]-0.5*(x[1]-x[0])
        edges[-1] = x[-1]+0.5*(x[-1]-x[-2])
        widths = np.diff(edges)
        return (response@(intrinsic*widths))/widths
    initial_shape = shape(p0)
    linear = np.linalg.lstsq(whiten(np.column_stack((initial_shape,basis))),whiten(y),rcond=None)[0]
    linear[0] = max(float(linear[0]), np.finfo(float).eps)
    pinit = np.r_[p0, linear]
    lower = np.r_[lo, 0.0, np.full(basis.shape[1], -np.inf)]
    upper = np.r_[hi, np.inf, np.full(basis.shape[1], np.inf)]
    def predict(p):
        return p[nphys]*shape(p[:nphys])+basis@p[nphys+1:]
    def residual(p):
        return whiten(predict(p)-y)
    fit = least_squares(residual,pinit,bounds=(lower,upper),x_scale="jac",method="trf",max_nfev=int(max_nfev))
    if not fit.success or np.any(~np.isfinite(fit.x)):
        raise ValueError("joint kinetic Thomson/background optimizer failed: "+str(fit.message))
    jac = np.asarray(fit.jac,dtype=float)
    singular = np.linalg.svd(jac,compute_uv=False)
    tol = np.finfo(float).eps*max(jac.shape)*(singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular>tol))
    cond = float(singular[0]/singular[-1]) if singular.size and singular[-1]>0 else float("inf")
    if rank < fit.x.size or not np.isfinite(cond) or cond > 1e12:
        raise ValueError(f"joint physical/background fit is non-identifiable (rank {rank}/{fit.x.size}, condition={cond})")
    dof = max(1,y.size-fit.x.size)
    joint_cov = np.linalg.pinv(jac.T@jac,rcond=1e-12)
    if noise == "unweighted_no_measurement_uncertainty_supplied":
        joint_cov *= float(fit.fun@fit.fun/dof)
    background = basis@fit.x[nphys+1:]
    fitted = predict(fit.x)
    if model == "kappa":
        theta_e = float(fit.x[0] / _E_REST_EV)
        beta_flow = float(abs(fit.x[1]) / _C)
        if theta_e >= 0.01 or beta_flow >= 0.1:
            raise ValueError("non-relativistic Kappa model rejected by its declared theta_e < 0.01 and |v|/c < 0.1 screening policy")
    else:
        theta_e = float(fit.x[0] / _E_REST_EV)
        beta_flow = float(abs(fit.x[1]) / _C)
    return {
        "physical_model": model, "physical_parameters": fit.x[:nphys].copy(),
        "physical_parameter_names": ["electron_temperature_eV","electron_velocity_projection_m_s"]+(["kappa_index"] if model=="kappa" else []),
        "amplitude": float(fit.x[nphys]), "background_model": bg_model,
        "background_coefficients": fit.x[nphys+1:].copy(),
        "fitted_thomson_spectrum": fitted-background, "fitted_background": background,
        "fitted_spectrum": fitted, "residual": y-fitted,
        "joint_parameter_covariance": joint_cov,
        "physical_parameter_covariance": joint_cov[:nphys,:nphys],
        "physical_background_cross_covariance": joint_cov[:nphys,nphys+1:],
        "joint_parameter_order": ["electron_temperature_eV","electron_velocity_projection_m_s"]+(["kappa_index"] if model=="kappa" else [])+["spectral_amplitude"]+[f"background_coefficient_{i}" for i in range(basis.shape[1])],
        "jacobian_rank": rank, "jacobian_parameter_count": int(fit.x.size),
        "jacobian_condition_number": cond, "optimizer_success": bool(fit.success),
        "noise_model": noise,
        "covariance_status": "residual_scaled_unweighted_joint_covariance_not_measurement_uncertainty" if noise.startswith("unweighted") else "local_joint_GLS_covariance_conditional_on_supplied_measurement_covariance",
        "absolute_density_inferred": False, "theta_e": theta_e, "bulk_flow_beta": beta_flow,
        "recoil_included": bool(recoil) if model=="relativistic_maxwell_juttner" else False,
        "cross_section_model": ("polarized_single_scattering_Klein_Nishina" if recoil else "polarized_elastic_Thomson") if model=="relativistic_maxwell_juttner" else "isotropic_finite_variance_Kappa_projection",
        "experimental_validation": "not_performed"
    }


# Public canonical name resolves to the full joint-covariance implementation.
fit_noncollective_kinetic_thomson_with_background = fit_noncollective_kinetic_thomson_with_joint_background

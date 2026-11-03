"""Joint Thomson spectral nuisance/physics and multi-profile inversion closures.

These routines operate on calibrated measured spectra.  They deliberately
separate data identifiability from regularized numerical stability and never
promote a regularization-selected null-space component to a measured result.
"""
from __future__ import annotations

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
        raise ValueError(f"{name} must contain finite values")
    return out


def _whitener(n, uncertainty=None, covariance=None):
    if uncertainty is not None and covariance is not None:
        raise ValueError("supply uncertainty or covariance, not both")
    if uncertainty is not None:
        sigma = _real(uncertainty, "measurement_uncertainty", 1)
        if sigma.shape != (n,) or np.any(sigma <= 0):
            raise ValueError("measurement_uncertainty must be positive and match flattened measurements")
        return lambda r: r / sigma, "supplied_pointwise_measurement_uncertainty"
    if covariance is not None:
        cov = _real(covariance, "measurement_covariance", 2)
        if cov.shape != (n, n) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-13):
            raise ValueError("measurement_covariance must be symmetric and match flattened measurements")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement_covariance must be positive definite") from exc
        return lambda r: np.linalg.solve(chol, r), "supplied_full_measurement_covariance"
    return lambda r: r, "unweighted_no_measurement_uncertainty_supplied"


def _identifiability(jacobian, rtol=None):
    j = np.asarray(jacobian, dtype=float)
    singular = np.linalg.svd(j, compute_uv=False)
    threshold = (np.finfo(float).eps * max(j.shape) if rtol is None else float(rtol)) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > threshold))
    return {
        "rank": rank, "parameter_count": int(j.shape[1]),
        "identifiable": rank == j.shape[1],
        "singular_values": singular,
        "rank_tolerance": float(threshold),
        "condition_number": float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf"),
    }


def compton_recoil_contract(laser_wavelength_nm, *, include_compton_recoil,
                            maximum_recoil_correction_fraction=0.01):
    """Return explicit recoil/cross-section provenance for the MJ forward model."""
    laser = float(laser_wavelength_nm)
    limit = float(maximum_recoil_correction_fraction)
    if not np.isfinite(laser) or laser <= 0 or not np.isfinite(limit) or not 0 < limit < 1:
        raise ValueError("laser wavelength and recoil screening fraction must be positive and finite")
    photon_energy_eV = 1239.8419843320026 / laser
    recoil_scale = photon_energy_eV / 510998.95069
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if not include_compton_recoil and recoil_scale > limit:
        raise ValueError("Thomson-limit model rejected: estimated photon recoil exceeds the declared screening fraction; enable Compton recoil")
    return {
        "include_compton_recoil": bool(include_compton_recoil),
        "cross_section": "polarized_Klein_Nishina" if include_compton_recoil else "polarized_Thomson",
        "photon_recoil_scale_fraction": float(recoil_scale),
        "maximum_recoil_correction_fraction": limit,
        "model_regime": "single_scattering_Compton" if include_compton_recoil else "elastic_Thomson_limit",
        "warning": None if include_compton_recoil or recoil_scale <= limit else "recoil outside selected model validity",
        "density_from_normalized_shape": False,
    }


def fit_thomson_with_structured_background(
    wavelength_nm, measured_spectrum, *, forward_model, physical_initial,
    background_design, physical_bounds=(-np.inf, np.inf),
    background_bounds=(-np.inf, np.inf), measurement_uncertainty=None,
    measurement_covariance=None, max_nfev=5000,
):
    """Joint nonlinear physical-spectrum + linear detector-background GLS fit.

    forward_model(physical_parameters, wavelength_nm) must return the Thomson
    signal in detector units before addition of background.  background_design
    has shape (n_wavelength, n_background_coefficients).  Both parameter
    blocks are optimized together; nuisance/physics cross-covariance is kept.
    """
    x = _real(wavelength_nm, "wavelength_nm", 1)
    y = _real(measured_spectrum, "measured_spectrum", 1)
    p0 = _real(physical_initial, "physical_initial", 1)
    bmat = _real(background_design, "background_design", 2)
    if x.shape != y.shape or x.size < 7 or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength and measured spectrum must match on a strictly increasing grid")
    if bmat.shape[0] != x.size or bmat.shape[1] < 1 or np.linalg.matrix_rank(bmat) < bmat.shape[1]:
        raise ValueError("background_design must have full column rank and one row per wavelength")
    if not callable(forward_model):
        raise ValueError("forward_model must be callable")
    p_lo, p_hi = (np.broadcast_to(np.asarray(v, dtype=float), p0.shape).copy() for v in physical_bounds)
    b0 = np.zeros(bmat.shape[1])
    b_lo, b_hi = (np.broadcast_to(np.asarray(v, dtype=float), b0.shape).copy() for v in background_bounds)
    lower, upper = np.r_[p_lo, b_lo], np.r_[p_hi, b_hi]
    if np.any(lower >= upper) or np.any(p0 < p_lo) or np.any(p0 > p_hi):
        raise ValueError("initial parameters must lie inside strictly ordered bounds")
    whiten, noise_model = _whitener(x.size, measurement_uncertainty, measurement_covariance)
    def prediction(z):
        physical = np.asarray(forward_model(z[:p0.size], x), dtype=float)
        if physical.shape != x.shape or not np.all(np.isfinite(physical)):
            raise ValueError("forward_model must return a finite vector matching wavelength_nm")
        return physical + bmat @ z[p0.size:]
    def residual(z):
        return whiten(prediction(z) - y)
    result = least_squares(residual, np.r_[p0, b0], bounds=(lower, upper), method="trf", max_nfev=int(max_nfev), x_scale="jac")
    jac = np.asarray(result.jac, dtype=float)
    ident = _identifiability(jac)
    dof = max(1, y.size - jac.shape[1])
    covariance = None
    covariance_status = "not_reported_non_identifiable"
    if ident["identifiable"]:
        covariance = np.linalg.pinv(jac.T @ jac)
        if noise_model == "unweighted_no_measurement_uncertainty_supplied":
            covariance *= float(result.fun @ result.fun / dof)
            covariance_status = "residual_scaled_unweighted_covariance_not_measurement_uncertainty"
        else:
            covariance_status = "local_GLS_parameter_covariance_conditional_on_supplied_measurement_covariance"
    z = result.x
    fitted = prediction(z)
    nphys = p0.size
    return {
        "model": "joint_measured_Thomson_signal_plus_structured_detector_background",
        "physical_parameters": z[:nphys], "background_coefficients": z[nphys:],
        "fitted_spectrum": fitted, "fitted_thomson_spectrum": fitted - bmat @ z[nphys:],
        "fitted_background": bmat @ z[nphys:], "residual": y - fitted,
        "measurement_origin": "user_supplied_calibrated_measured_spectrum",
        "noise_model": noise_model, "parameter_order": ["physical"] * nphys + ["background"] * bmat.shape[1],
        "joint_parameter_covariance": covariance, "covariance_status": covariance_status,
        "physical_background_cross_covariance_retained": covariance is not None,
        "identifiability": ident, "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message), "cost": float(result.cost),
        "degrees_of_freedom": int(dof),
        "uncertainty_semantics": "No measurement uncertainty was inferred; unweighted covariance, if available, is residual-scaled only.",
    }


def reconstruct_joint_thomson_profiles(
    wavelength_nm, measured_spectra, radius_m, *, forward_model,
    initial_profiles, profile_bounds, measurement_uncertainty=None,
    measurement_covariance=None, regularization_strength=0.0, max_nfev=10000,
):
    """Jointly infer ne(r), Te(r), Ti(r) from all measured channel spectra.

    forward_model(channel_index, wavelength_nm, ne, Te, Ti) returns one
    detector-space spectrum. Regularization penalizes second differences in
    each profile. Data-only Jacobian rank is reported separately: a smooth
    regularized solution is never labelled data-identifiable solely because
    the penalty stabilized it.
    """
    x = _real(wavelength_nm, "wavelength_nm", 1)
    y = _real(measured_spectra, "measured_spectra", 2)
    r = _real(radius_m, "radius_m", 1)
    p0 = _real(initial_profiles, "initial_profiles", 2)
    if y.shape != (r.size, x.size) or p0.shape != (3, r.size):
        raise ValueError("spectra must be (n_radius,n_wavelength); initial_profiles must be (3,n_radius) ordered ne,Te,Ti")
    if np.any(np.diff(x) <= 0) or np.any(np.diff(r) <= 0) or np.any(r < 0):
        raise ValueError("wavelength and radius coordinates must be strictly increasing; radius must be non-negative")
    if not callable(forward_model):
        raise ValueError("forward_model must be callable")
    lam = float(regularization_strength)
    if not np.isfinite(lam) or lam < 0:
        raise ValueError("regularization_strength must be finite and non-negative")
    lo, hi = (_real(v, "profile_bound", 2) for v in profile_bounds)
    if lo.shape != p0.shape or hi.shape != p0.shape or np.any(lo >= hi) or np.any(p0 < lo) or np.any(p0 > hi):
        raise ValueError("profile bounds must match (3,n_radius), be ordered, and contain initial profiles")
    whiten, noise_model = _whitener(y.size, measurement_uncertainty, measurement_covariance)
    nchan = r.size
    def predict(flat):
        profiles = flat.reshape(3, nchan)
        rows = []
        for i in range(nchan):
            row = np.asarray(forward_model(i, x, *profiles[:, i]), dtype=float)
            if row.shape != x.shape or not np.all(np.isfinite(row)):
                raise ValueError("forward_model must return one finite wavelength vector per channel")
            rows.append(row)
        return np.asarray(rows)
    def data_residual(flat):
        return whiten((predict(flat) - y).ravel())
    def second_difference(flat):
        p = flat.reshape(3, nchan)
        if nchan < 3:
            return np.empty(0)
        spacing = np.diff(r)
        left_slope = np.diff(p, axis=1)[:, :-1] / spacing[:-1]
        right_slope = np.diff(p, axis=1)[:, 1:] / spacing[1:]
        curvature = 2.0 * (right_slope - left_slope) / (spacing[:-1] + spacing[1:])
        # Radius-aware second derivative; unlike an unscaled second difference,
        # this remains meaningful on nonuniform diagnostic-channel spacing.
        return curvature.ravel()
    def objective(flat):
        data = data_residual(flat)
        return np.r_[data, np.sqrt(lam) * second_difference(flat)] if lam > 0 else data
    result = least_squares(objective, p0.ravel(), bounds=(lo.ravel(), hi.ravel()), method="trf", max_nfev=int(max_nfev), x_scale="jac")
    fitted_profiles = result.x.reshape(3, nchan)
    fitted_spectra = predict(result.x)
    data_jac = np.asarray(result.jac[:y.size], dtype=float)
    # For a full covariance whitener, residual vector dimension remains y.size.
    data_ident = _identifiability(data_jac)
    augmented_ident = _identifiability(result.jac)
    dof = max(1, y.size - data_jac.shape[1])
    cov = None
    covariance_status = "not_reported_data_non_identifiable"
    if result.success and data_ident["identifiable"]:
        cov = np.linalg.pinv(result.jac.T @ result.jac)
        if noise_model == "unweighted_no_measurement_uncertainty_supplied":
            cov *= float(np.sum(data_residual(result.x)**2) / dof)
            covariance_status = "regularized_unweighted_local_covariance_residual_scaled"
        else:
            covariance_status = "regularized_local_covariance_conditional_on_supplied_measurement_covariance"
    if not result.success:
        status = "optimizer_failed"
        covariance_status = "not_reported_optimizer_failed"
    elif data_ident["identifiable"]:
        status = "primary_data_identifiable"
    else:
        status = "non_identifiable_regularized_solution"
    return {
        "profiles": {"electron_density_m3": fitted_profiles[0], "electron_temperature_eV": fitted_profiles[1], "ion_temperature_eV": fitted_profiles[2]},
        "profile_order": ["electron_density_m3", "electron_temperature_eV", "ion_temperature_eV"],
        "fitted_spectra": fitted_spectra, "residual_spectra": y - fitted_spectra,
        "measurement_origin": "user_supplied_calibrated_measured_channel_spectra",
        "status": status, "data_identifiability": data_ident,
        "regularized_jacobian_identifiability": augmented_ident,
        "regularization_strength": lam, "regularization_is_not_measurement": True,
        "joint_profile_covariance": cov, "covariance_status": covariance_status,
        "covariance_parameter_order": [f"{name}[{i}]" for name in ("ne", "Te", "Ti") for i in range(nchan)],
        "noise_model": noise_model, "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message), "degrees_of_freedom": int(dof),
        "limitations": [
            "Ion temperature is identifiable only if the supplied forward model contains an ion-sensitive spectral feature.",
            "Single-channel data do not identify arbitrary radial profiles.",
            "A regularized profile in a data-Jacobian null space is conditional on the regularization prior and is not a primary measurement.",
            "No equilibrium/flux-surface mapping, PSF, channel cross-talk or geometry uncertainty is inferred by this routine.",
        ],
    }

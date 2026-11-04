"""Joint multi-view optical Thomson inference for a gyrotropic bi-Maxwellian.

The model fits calibrated measured spectra directly. Each channel measures the
one-dimensional marginal along its own scattering vector; independent magnetic
pitch angles are required to separate parallel and perpendicular temperatures.
This is a non-relativistic elastic-Thomson small-beta forward model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


_C = 299792458.0
_E = 1.602176634e-19
_ME = 9.1093837015e-31


def _real_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must contain real numeric values")
    out = raw.astype(float, copy=False)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite and non-empty")
    return out


def _bin_widths(axis):
    edges = np.empty(axis.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0.0):
        raise ValueError("wavelength_nm must define positive finite-volume bins")
    return widths


def fit_gyrotropic_bi_maxwellian_thomson(
    wavelength_nm, measured_spectra, *, laser_wavelength_nm,
    scattering_angle_rad, magnetic_field_unit, scattering_vector_unit,
    instrument_response_matrix, measurement_covariance=None,
    signal_uncertainty=None, model_discrepancy_covariance=None,
    initial_temperature_parallel_eV=1000.0,
    initial_temperature_perpendicular_eV=1000.0,
    temperature_bounds_eV=(1.0e-3, 1.0e7),
    flow_bounds_m_s=(-0.05 * _C, 0.05 * _C),
    max_nfev=4000,
):
    """Jointly infer T_parallel, T_perpendicular and per-view q-flow projections.

    Parameters
    ----------
    wavelength_nm : (n_bin,) array
        Calibrated wavelength-bin centers, strictly increasing.
    measured_spectra : (n_view, n_bin) array
        Calibrated spectral density per wavelength.
    magnetic_field_unit : (3,) or (n_view, 3) array
        Supplied unit magnetic-field direction for each view.
    scattering_vector_unit : (n_view, 3) array
        Supplied unit q directions, q = k_scattered - k_incident.
    instrument_response_matrix : (n_bin,n_bin) or (n_view,n_bin,n_bin)
        Measured response; columns must sum to one. No renormalization is done.
    measurement_covariance : optional (n_view*n_bin, n_view*n_bin) array
        Full channel-major measurement covariance.
    signal_uncertainty : optional (n_view,n_bin) array
        Pointwise standard uncertainties; mutually exclusive with covariance.
    model_discrepancy_covariance : optional full channel-major covariance
        Independent additive PSD model-discrepancy covariance.

    The wavelength mapping uses lambda_s/lambda_0 ~= 1 - q_hat.v/c and is
    therefore a non-relativistic elastic-Thomson approximation. The projected
    thermal variance is e*T_q/m_e with
    T_q=T_perp+(T_parallel-T_perp)(b_hat.q_hat)^2.
    """
    wavelength = _real_array(wavelength_nm, "wavelength_nm", 1)
    spectra = _real_array(measured_spectra, "measured_spectra", 2)
    if wavelength.size < 12 or np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must be positive, increasing, and contain >=12 bins")
    n_view, n_bin = spectra.shape
    if n_view < 1 or n_bin != wavelength.size:
        raise ValueError("measured_spectra must have shape (n_view, n_wavelength)")
    laser = float(laser_wavelength_nm)
    angle = float(scattering_angle_rad)
    if not np.isfinite(laser) or laser <= 0.0 or not np.isfinite(angle) or not 0.0 < angle < np.pi:
        raise ValueError("laser_wavelength_nm must be positive and scattering_angle_rad must lie in (0, pi)")
    b = _real_array(magnetic_field_unit, "magnetic_field_unit")
    q = _real_array(scattering_vector_unit, "scattering_vector_unit", 2)
    if b.shape == (3,):
        b = np.broadcast_to(b, (n_view, 3)).copy()
    if b.shape != (n_view, 3) or q.shape != (n_view, 3):
        raise ValueError("magnetic_field_unit and scattering_vector_unit must have shape (3,) or (n_view,3)")
    if not np.allclose(np.linalg.norm(b, axis=1), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("each magnetic_field_unit row must have unit norm")
    if not np.allclose(np.linalg.norm(q, axis=1), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("each scattering_vector_unit row must have unit norm")
    mu = np.einsum("ij,ij->i", b, q)
    mu2 = np.clip(mu * mu, 0.0, 1.0)
    design = np.column_stack((1.0 - mu2, mu2))
    geometry_rank = int(np.linalg.matrix_rank(design))
    response = _real_array(instrument_response_matrix, "instrument_response_matrix")
    if response.ndim == 2:
        response = np.broadcast_to(response, (n_view, n_bin, n_bin)).copy()
    if response.shape != (n_view, n_bin, n_bin) or np.any(response < 0.0):
        raise ValueError("instrument_response_matrix must be non-negative with shape (n_bin,n_bin) or (n_view,n_bin,n_bin)")
    column_sums = response.sum(axis=1)
    if np.any(column_sums <= 0.0) or not np.allclose(column_sums, 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured response columns must sum to one; response is never silently renormalized")
    if measurement_covariance is not None and signal_uncertainty is not None:
        raise ValueError("supply measurement_covariance or signal_uncertainty, not both")
    size = n_view * n_bin
    if measurement_covariance is not None:
        cov = _real_array(measurement_covariance, "measurement_covariance", 2)
        if cov.shape != (size, size) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-12 * max(np.max(np.abs(cov)), 1e-300)):
            raise ValueError("measurement_covariance must be symmetric and channel-major")
        cov = 0.5 * (cov + cov.T)
    elif signal_uncertainty is not None:
        sigma = _real_array(signal_uncertainty, "signal_uncertainty", 2)
        if sigma.shape != spectra.shape or np.any(sigma <= 0.0):
            raise ValueError("signal_uncertainty must be positive and match measured_spectra")
        cov = np.diag(sigma.ravel() ** 2)
    else:
        cov = np.eye(size)
    measurement_covariance_only = cov.copy()
    if model_discrepancy_covariance is not None:
        discrepancy = _real_array(model_discrepancy_covariance, "model_discrepancy_covariance", 2)
        if discrepancy.shape != (size, size) or not np.allclose(discrepancy, discrepancy.T, rtol=1e-10, atol=1e-12 * max(np.max(np.abs(discrepancy)), 1e-300)):
            raise ValueError("model_discrepancy_covariance must be symmetric and channel-major")
        discrepancy = 0.5 * (discrepancy + discrepancy.T)
        eig = np.linalg.eigvalsh(discrepancy)
        if eig[0] < -1e-10 * max(float(np.max(np.abs(eig))), 1e-300):
            raise ValueError("model_discrepancy_covariance must be positive semidefinite")
        cov = cov + discrepancy
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("effective spectral covariance must be positive definite") from exc
    whiten = lambda value: np.linalg.solve(chol, np.asarray(value, dtype=float).ravel())
    widths = _bin_widths(wavelength)
    t0p = float(initial_temperature_parallel_eV)
    t0t = float(initial_temperature_perpendicular_eV)
    tlo, thi = map(float, temperature_bounds_eV)
    if not (np.isfinite(t0p) and np.isfinite(t0t) and 0.0 < tlo < thi and tlo <= t0p <= thi and tlo <= t0t <= thi):
        raise ValueError("positive initial temperatures must lie inside ordered temperature_bounds_eV")
    flow_lo, flow_hi = map(float, flow_bounds_m_s)
    if not np.isfinite(flow_lo + flow_hi) or flow_lo >= flow_hi or flow_lo <= -_C or flow_hi >= _C:
        raise ValueError("flow_bounds_m_s must be ordered and strictly subluminal")
    if isinstance(max_nfev, (bool, np.bool_)) or int(max_nfev) < 1:
        raise ValueError("max_nfev must be a positive integer")
    qmag_factor = 2.0 * np.sin(angle / 2.0)
    # Parameters: log(T_parallel), log(T_perp), per-view projected flow,
    # positive amplitude, constant background and linear background.
    edge = max(2, n_bin // 10)
    p0 = [np.log(t0p), np.log(t0t)]
    lower = [np.log(tlo), np.log(tlo)]
    upper = [np.log(thi), np.log(thi)]
    for i in range(n_view):
        y = spectra[i]
        span = max(float(np.ptp(y)), np.finfo(float).tiny)
        baseline = float(np.median(np.r_[y[:edge], y[-edge:]]))
        p0.extend([0.0, np.log(max(float(np.max(y) - baseline), span * 1e-6)), baseline, 0.0])
        lower.extend([flow_lo, -50.0, -np.inf, -np.inf])
        upper.extend([flow_hi, 50.0, np.inf, np.inf])
    p0 = np.asarray(p0, dtype=float)
    lower, upper = np.asarray(lower), np.asarray(upper)

    def predict(parameters):
        tpar, tperp = np.exp(parameters[:2])
        predictions = np.empty_like(spectra)
        directional = tperp + (tpar - tperp) * mu2
        if np.any(directional <= 0.0):
            raise ValueError("directional bi-Maxwellian temperature became non-positive")
        for i in range(n_view):
            flow, log_amp, offset, slope = parameters[2 + 4*i: 6 + 4*i]
            sigma_v = np.sqrt(_E * directional[i] / _ME)
            center = laser * (1.0 - qmag_factor * flow / _C)
            sigma_lambda = laser * qmag_factor * sigma_v / _C
            if sigma_lambda <= np.finfo(float).tiny:
                raise ValueError("predicted Thomson spectral width is numerically zero")
            intrinsic = np.exp(-0.5 * ((wavelength - center) / sigma_lambda) ** 2)
            intrinsic /= np.sqrt(2.0 * np.pi) * sigma_lambda
            detector = response[i] @ intrinsic
            predictions[i] = np.exp(log_amp) * detector + offset + slope * (wavelength - laser)
        return predictions, directional

    def residual(parameters):
        return whiten(predict(parameters)[0] - spectra)

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"gyrotropic bi-Maxwellian Thomson fit failed: {fit.message}")
    fitted, directional = predict(fit.x)
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    tol = np.finfo(float).eps * max(fit.jac.shape) * (singular[0] if singular.size else 0.0)
    full_rank = int(np.count_nonzero(singular > tol)) == fit.x.size
    parameter_covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-12) if (full_rank and geometry_rank == 2) else None
    covariance_scale_method = "supplied_absolute_covariance"
    if parameter_covariance is not None and measurement_covariance is None and signal_uncertainty is None:
        dof = max(1, size - fit.x.size)
        parameter_covariance *= (2.0 * float(fit.cost) / dof)
        covariance_scale_method = "unweighted_residual_scaled"
    if parameter_covariance is not None:
        transform = np.eye(fit.x.size)
        transform[0, 0] = np.exp(fit.x[0])
        transform[1, 1] = np.exp(fit.x[1])
        parameter_covariance = transform @ parameter_covariance @ transform.T
    reportable_anisotropy = geometry_rank == 2 and full_rank
    tpar = float(np.exp(fit.x[0])) if reportable_anisotropy else None
    tperp = float(np.exp(fit.x[1])) if reportable_anisotropy else None
    return {
        "model": "gyrotropic_bi_maxwellian_nonrelativistic_optical_thomson",
        "model_validity": {
            "kinematics": "non-relativistic elastic Thomson mapping lambda_s/lambda_0 ~= 1 - q_hat.v/c",
            "distribution": "gyrotropic bi-Maxwellian with zero heat flux",
            "relativistic_corrections": False,
            "validity_note": "Use a relativistic forward model when thermal or bulk electron speeds make the small-beta approximation inadequate.",
        },
        "electron_temperature_parallel_eV": tpar,
        "electron_temperature_perpendicular_eV": tperp,
        "directional_temperature_eV": directional,
        "magnetic_pitch_cosine": mu,
        "magnetic_pitch_cosine_squared": mu2,
        "geometry_design_rank": geometry_rank,
        "anisotropy_identifiability": "identified" if reportable_anisotropy else "non_identifiable",
        "reportable_as_inferred_parameter": bool(reportable_anisotropy),
        "non_identifiability_reason": None if reportable_anisotropy else "Distinct squared magnetic pitch angles and a full-rank spectral fit are required to identify both T_parallel and T_perpendicular.",
        "electron_velocity_projection_m_s": fit.x[2::4].copy(),
        "fitted_spectra": fitted,
        "residual_spectra": spectra - fitted,
        "measurement_covariance": measurement_covariance_only,
        "model_discrepancy_covariance": None if model_discrepancy_covariance is None else discrepancy,
        "effective_covariance": cov,
        "parameter_covariance": parameter_covariance,
        "parameter_order": ["T_parallel_eV", "T_perpendicular_eV"] + [
            item for i in range(n_view) for item in (
                f"flow_projection_view_{i}_m_s", f"log_amplitude_view_{i}",
                f"background_offset_view_{i}", f"background_slope_view_{i}_per_nm"
            )
        ],
        "jacobian_singular_values": singular,
        "optimizer": {"success": bool(fit.success), "message": str(fit.message),
                      "nfev": int(fit.nfev), "cost": float(fit.cost)},
        "covariance_provenance": {
            "measurement": "supplied_full_channel_major" if measurement_covariance is not None else (
                "supplied_pointwise_diagonal" if signal_uncertainty is not None else "unweighted_identity_for_fit"
            ),
            "parameter_covariance_scale": covariance_scale_method,
            "model_discrepancy_supplied": model_discrepancy_covariance is not None,
            "combination": "C_effective = C_measurement + C_model_discrepancy; added once",
        },
        "experimental_validation": "not_performed",
    }

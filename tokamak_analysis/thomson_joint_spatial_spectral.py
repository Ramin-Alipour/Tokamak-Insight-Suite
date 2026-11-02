"""Joint wavelength-resolved spatial Thomson inverse with a fixed mixing operator.

This module fits calibrated measured channel-by-wavelength spectra directly.
The spatial mixing operator and spectral response are supplied fixed forward
operators: this code does not calibrate instruments or estimate a PSF.
The selected physical model is the isotropic, non-relativistic Maxwellian
single-particle Thomson line shape with a scalar scattering-vector flow.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


_C_M_S = 299792458.0
_MEC2_EV = 510998.95069


def _bin_widths(axis):
    edges = np.empty(axis.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (axis[:-1] + axis[1:])
    edges[0] = axis[0] - 0.5 * (axis[1] - axis[0])
    edges[-1] = axis[-1] + 0.5 * (axis[-1] - axis[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0):
        raise ValueError("wavelength_nm must define positive bin widths")
    return widths


def _real_array(values, name):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans, strings or complex values")
    result = raw.astype(float, copy=False)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _finite_real_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _response_stack(response, n_channels, n_wavelengths):
    if response is None:
        return np.repeat(np.eye(n_wavelengths)[None, :, :], n_channels, axis=0)
    value = _real_array(response, "instrument_response")
    if value.ndim == 2:
        value = np.repeat(value[None, :, :], n_channels, axis=0)
    if value.shape != (n_channels, n_wavelengths, n_wavelengths):
        raise ValueError("instrument_response must be (n_lambda,n_lambda) or (n_channels,n_lambda,n_lambda)")
    if not np.all(np.isfinite(value)) or np.any(value < 0):
        raise ValueError("instrument_response must be finite and non-negative")
    # Each channel stores a wavelength-output x wavelength-input operator.
    # Sum over output rows (axis=1) so every input-wavelength column is
    # checked independently for every channel. Summing axis=0 would instead
    # mix channels and response rows, rejecting valid shared/channel matrices.
    column_sums = value.sum(axis=1)
    if np.any(column_sums <= 0) or not np.allclose(
        column_sums, 1.0, rtol=1e-6, atol=1e-9
    ):
        raise ValueError("instrument_response columns must sum to one for every channel")
    return value


def _curvature(axis):
    matrix = np.zeros((axis.size - 2, axis.size), dtype=float)
    for i in range(axis.size - 2):
        left = axis[i + 1] - axis[i]
        right = axis[i + 2] - axis[i + 1]
        matrix[i, i] = 2.0 / (left * (left + right))
        matrix[i, i + 1] = -2.0 / (left * right)
        matrix[i, i + 2] = 2.0 / (right * (left + right))
    return matrix


def reconstruct_joint_thomson_spatial_spectra(
    wavelength_nm, measured_spectra, radius_m, spatial_mixing_matrix, *,
    laser_wavelength_nm, scattering_angle_rad, instrument_response=None,
    signal_uncertainty=None, joint_spectral_covariance=None,
    regularization_strength=0.0, positive_amplitude=True, max_nfev=3000,
):
    """Jointly infer Te(r), projected flow(r), and spectral amplitude(r).

    Data layout is channel x wavelength; covariance is flattened in
    channel-major order. The supplied mixing matrix maps spatial nodes to
    detector channels. Background offset and slope are detector-channel
    nuisance parameters and are not mixed as plasma emission.
    """
    x = _real_array(wavelength_nm, "wavelength_nm")
    y = _real_array(measured_spectra, "measured_spectra")
    r = _real_array(radius_m, "radius_m")
    h = _real_array(spatial_mixing_matrix, "spatial_mixing_matrix")
    if x.ndim != 1 or x.size < 9 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0):
        raise ValueError("wavelength_nm must be finite, strictly increasing, and contain >=9 samples")
    if y.ndim != 2 or y.shape[1] != x.size or not np.all(np.isfinite(y)):
        raise ValueError("measured_spectra must be a finite channel-by-wavelength array")
    if r.ndim != 1 or r.size < 3 or not np.all(np.isfinite(r)) or np.any(np.diff(r) <= 0):
        raise ValueError("radius_m must be finite, strictly increasing, and contain >=3 spatial nodes")
    m, nlam = y.shape
    if h.shape != (m, r.size) or not np.all(np.isfinite(h)) or np.any(h < 0):
        raise ValueError("spatial_mixing_matrix must be finite, non-negative, and shaped (n_channels,n_nodes)")
    if np.linalg.matrix_rank(h) < min(h.shape):
        raise ValueError("spatial_mixing_matrix is rank deficient")
    laser = _finite_real_scalar(laser_wavelength_nm, "laser_wavelength_nm")
    angle = _finite_real_scalar(scattering_angle_rad, "scattering_angle_rad")
    lam_reg = _finite_real_scalar(regularization_strength, "regularization_strength")
    if not np.isfinite(laser) or laser <= 0 or not np.isfinite(angle) or not 0 < angle < np.pi:
        raise ValueError("laser_wavelength_nm must be positive and scattering_angle_rad must lie strictly between 0 and pi")
    if not np.isfinite(lam_reg) or lam_reg < 0:
        raise ValueError("regularization_strength must be finite and non-negative")
    if signal_uncertainty is not None and joint_spectral_covariance is not None:
        raise ValueError("supply signal_uncertainty or joint_spectral_covariance, not both")
    if joint_spectral_covariance is not None:
        cov = _real_array(joint_spectral_covariance, "joint_spectral_covariance")
        if cov.shape != (m*nlam, m*nlam) or not np.all(np.isfinite(cov)) or not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-13):
            raise ValueError("joint_spectral_covariance must be finite, symmetric, and channel-major")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("joint_spectral_covariance must be positive definite") from exc
        whiten = lambda v: np.linalg.solve(chol, np.asarray(v).reshape(-1))
        noise_model = "full_channel_major_joint_spectral_covariance"
    elif signal_uncertainty is not None:
        sigma = _real_array(signal_uncertainty, "signal_uncertainty")
        if sigma.shape != y.shape or not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("signal_uncertainty must be positive, finite, and match measured_spectra")
        whiten = lambda v: (np.asarray(v) / sigma).reshape(-1)
        noise_model = "independent_pointwise_spectral_uncertainty"
    else:
        scale = max(float(np.std(y)), float(np.max(np.abs(y))) * 1e-12, np.finfo(float).tiny)
        whiten = lambda v: np.asarray(v).reshape(-1) / scale
        noise_model = "unweighted_residual_scaled"

    widths = _bin_widths(x)
    response = _response_stack(instrument_response, m, nlam)
    q = 2.0 * np.sin(angle / 2.0)
    d2 = _curvature(r)
    span = float(x[-1] - x[0])
    spectral_spacing = float(np.min(np.diff(x)))
    peak_scale = max(float(np.max(y) - np.median(y)), np.finfo(float).tiny)
    initial_sigma = max(span / 3.0, spectral_spacing)
    initial_te = max(_MEC2_EV * (initial_sigma / (2.0 * laser * np.sin(angle/2.0)))**2, 0.05)
    initial_te = float(np.clip(initial_te, 0.05, 1e5))
    initial_amp = peak_scale / max(float(np.max(h.sum(axis=1))), 1.0)

    # Parameter vector: log amplitude[n], log Te[n], flow[1e6 m/s,n],
    # then per-detector offset and linear spectral slope.
    p0 = np.r_[np.full(r.size, np.log(initial_amp)),
               np.full(r.size, np.log(initial_te)),
               np.zeros(r.size),
               np.full(m, float(np.median(y[:, :max(2,nlam//10)]))),
               np.zeros(m)]
    lower = np.r_[np.full(r.size, -700.0), np.full(r.size, np.log(0.01)),
                  np.full(r.size, -299.0),
                  np.full(m, -np.inf), np.full(m, -np.inf)]
    upper = np.r_[np.full(r.size, 700.0), np.full(r.size, np.log(1e7)),
                  np.full(r.size, 299.0),
                  np.full(m, np.inf), np.full(m, np.inf)]
    if not isinstance(positive_amplitude, (bool, np.bool_)):
        raise ValueError("positive_amplitude must be boolean")
    if not positive_amplitude:
        raise ValueError("signed plasma amplitudes are unsupported; use positive_amplitude=True")
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")

    def predict(p):
        amp = np.exp(p[:r.size])
        te = np.exp(p[r.size:2*r.size])
        flow = p[2*r.size:3*r.size] * 1e6
        offset = p[3*r.size:3*r.size+m]
        slope = p[3*r.size+m:3*r.size+2*m]
        sigma = 2.0 * laser * np.sqrt(te / _MEC2_EV) * np.sin(angle/2.0)
        center = laser * (1.0 - q * flow / _C_M_S)
        node_spectra = amp[:, None] * np.exp(
            -0.5 * ((x[None, :] - center[:, None]) / sigma[:, None])**2
        )
        mixed = h @ node_spectra
        predicted = np.empty_like(y)
        for channel in range(m):
            # Response acts on spectral density through bin-integrated power.
            convolved = (response[channel] @ (mixed[channel] * widths)) / widths
            predicted[channel] = convolved + offset[channel] + slope[channel] * (x-laser)
        return predicted, te, flow, amp

    def residual(p):
        predicted, te, flow, amp = predict(p)
        data = whiten(y - predicted)
        if lam_reg > 0:
            penalties = np.r_[np.sqrt(lam_reg) * (d2 @ np.log(te)),
                              np.sqrt(lam_reg) * (d2 @ flow / 1e6),
                              np.sqrt(lam_reg) * (d2 @ np.log(amp))]
            return np.r_[data, penalties]
        return data

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=int(max_nfev))
    predicted, te, flow, amp = predict(fit.x)
    data_residual = whiten(y - predicted)
    # Assess identifiability from measured spectral data alone.  The optimizer
    # Jacobian also contains optional regularization rows; counting those rows
    # as independent measurements would let smoothing manufacture rank.
    def data_residual_only(parameters):
        return whiten(y - predict(parameters)[0])

    parameter_count = int(fit.x.size)
    data_jacobian = np.empty((data_residual.size, parameter_count), dtype=float)
    for column in range(parameter_count):
        step = 1.0e-5 * max(1.0, abs(float(fit.x[column])))
        plus = fit.x.copy()
        minus = fit.x.copy()
        plus[column] += step
        minus[column] -= step
        data_jacobian[:, column] = (
            data_residual_only(plus) - data_residual_only(minus)
        ) / (2.0 * step)
    column_norms = np.linalg.norm(data_jacobian, axis=0)
    scaled_data_jacobian = data_jacobian / np.maximum(
        column_norms, np.finfo(float).tiny
    )[None, :]
    singular_values = np.linalg.svd(scaled_data_jacobian, compute_uv=False)
    rank_tolerance = (
        np.finfo(float).eps * max(scaled_data_jacobian.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    data_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    data_condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > rank_tolerance
        else float("inf")
    )
    optimizer_jacobian = np.asarray(fit.jac, dtype=float)
    optimizer_rank = int(np.linalg.matrix_rank(optimizer_jacobian))
    optimizer_condition = float(np.linalg.cond(optimizer_jacobian))
    data_identifiable = bool(fit.success and data_rank == parameter_count)
    dof = max(1, data_residual.size - parameter_count)
    covariance_p = np.linalg.pinv(optimizer_jacobian.T @ optimizer_jacobian)
    if joint_spectral_covariance is None and signal_uncertainty is None:
        covariance_p *= float(data_residual @ data_residual / dof)
    transform = np.diag(np.r_[amp, te, np.full(r.size, 1e6), np.ones(2*m)])
    physical_covariance = transform @ covariance_p @ transform.T
    diagonal = np.maximum(np.diag(physical_covariance), 0.0)
    uncertainties = np.sqrt(diagonal)
    n = r.size
    return {
        "analysis_status": "completed" if fit.success else "optimizer_not_converged",
        "primary_result_status": "identified_from_measured_data" if data_identifiable else "not_identifiable_from_measured_data",
        "primary_outputs_reportable_as_identified": data_identifiable,
        "model": "isotropic_nonrelativistic_maxwellian_single_particle_thomson",
        "radius_m": r.copy(),
        "electron_temperature_uncertainty_profile_eV": uncertainties[n:2*n] if data_identifiable else None,
        "electron_density_profile_m3": np.full(n, np.nan, dtype=float),
        "electron_density_uncertainty_profile_m3": np.full(n, np.nan, dtype=float),
        "electron_velocity_projection_uncertainty_profile_m_s": uncertainties[2*n:3*n] if data_identifiable else None,
        "electron_temperature_profile_eV": te if data_identifiable else None,
        "electron_temperature_uncertainty_eV": uncertainties[n:2*n] if data_identifiable else None,
        "electron_velocity_projection_profile_m_s": flow if data_identifiable else None,
        "electron_velocity_projection_uncertainty_m_s": uncertainties[2*n:3*n] if data_identifiable else None,
        "relative_spectral_amplitude_profile": amp if data_identifiable else None,
        "relative_spectral_amplitude_uncertainty": uncertainties[:n] if data_identifiable else None,
        "optimizer_candidate": {
            "role": "numerical_candidate_not_a_measured_result",
            "electron_temperature_profile_eV": te,
            "electron_velocity_projection_profile_m_s": flow,
            "relative_spectral_amplitude_profile": amp,
            "regularized_local_parameter_covariance": physical_covariance,
        },
        "fitted_spectra": predicted,
        "residuals": y-predicted,
        "parameter_covariance": physical_covariance if data_identifiable else None,
        "parameter_order": ([f"amplitude_{i}" for i in range(n)] +
                            [f"Te_{i}_eV" for i in range(n)] +
                            [f"flow_projection_{i}_m_s" for i in range(n)] +
                            [f"background_offset_channel_{i}" for i in range(m)] +
                            [f"background_slope_channel_{i}" for i in range(m)]),
        "detector_background_offset": fit.x[3*n:3*n+m].copy(),
        "detector_background_slope_per_nm": fit.x[3*n+m:3*n+2*m].copy(),
        "spatial_mixing_operator": "researcher_supplied_fixed_forward_operator",
        "instrument_response": "researcher_supplied_fixed_forward_operator",
        "regularization": {"kind": "nonuniform_radius_curvature_on_log_Te_flow_and_log_amplitude",
                           "strength": lam_reg, "user_selected": True},
        "noise_model": noise_model,
        "optimizer": {"success": bool(fit.success), "message": str(fit.message),
                      "nfev": int(fit.nfev), "cost": float(fit.cost),
                      "jacobian_rank": optimizer_rank, "parameter_count": parameter_count,
                      "jacobian_condition_number": optimizer_condition,
                      "data_only_jacobian_rank": data_rank,
                      "data_only_scaled_jacobian_condition_number": data_condition,
                      "regularization_rows_excluded_from_identifiability": True,
                      "degrees_of_freedom": dof},
        "identifiability": {
            "status": "identified_from_measured_data" if data_identifiable else "rank_deficient_or_optimizer_not_converged",
            "data_only_jacobian_rank": data_rank,
            "parameter_count": parameter_count,
            "primary": ["joint finite-dimensional Te(r), scattering-vector flow projection(r), and relative amplitude(r) under the supplied Maxwellian model and fixed operators"],
            "conditional": ["absolute density only if independent absolute-power and complete optical-geometry inputs are subsequently supplied; this function does not infer it"],
            "non_identifiable": ["flow components perpendicular to the scattering vector",
                                 "spatial structures in the null space of the supplied mixing operator",
                                 "sub-resolution spatial structure not encoded by the measured mixing operator",
                                 "unrestricted anisotropic or non-Maxwellian electron distributions"]},
        "uncertainty_semantics": "local linearized covariance conditional on the selected Maxwellian model, fixed spatial/spectral response operators, supplied noise model, and user-selected regularization; inspect primary_result_status because regularization is excluded from the data-only identifiability rank",
        "parameter_identifiability": {
            "primary": ["electron_temperature_profile_eV", "electron_velocity_projection_profile_m_s", "relative_spectral_amplitude_profile"] if data_identifiable else [],
            "conditional": ["absolute electron density requires a separate absolute-power and optical-collection inference with its complete auxiliary inputs"],
            "non_identifiable": ["perpendicular flow components", "spatial null-space and sub-resolution structures", "unrestricted anisotropic or non-Maxwellian electron distributions"],
            "status": "identified_from_measured_data" if data_identifiable else "not_identifiable_from_measured_data",
        },
        "experimental_validation": "not_performed; numerical software inference only",
    }

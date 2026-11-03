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
from scipy.special import gammaln


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
        raise ValueError(
            "A measured instrument_response is required. Supply the calibrated response operator, "
            "or explicitly supply an identity matrix only when the input spectra are already "
            "corrected for the instrument response."
        )
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


def _bound_aware_jacobian(function, parameters, lower, upper):
    """Finite-difference Jacobian that never evaluates outside fit bounds.

    Central differences are used in the interior and second-order accuracy is
    retained where both directions are available. At an active bound, a
    one-sided difference is used; a parameter fixed at identical lower/upper
    bounds has a zero derivative column.
    """
    point = np.asarray(parameters, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    if point.ndim != 1 or lo.shape != point.shape or hi.shape != point.shape:
        raise ValueError("parameters and finite-difference bounds must be matching vectors")
    if (not np.all(np.isfinite(point)) or np.any(np.isnan(lo)) or np.any(np.isnan(hi))
            or np.any(lo > hi) or np.any(point < lo) or np.any(point > hi)):
        raise ValueError("finite-difference point must be finite and lie within ordered bounds")
    base = np.asarray(function(point), dtype=float).reshape(-1)
    if not np.all(np.isfinite(base)):
        raise ValueError("finite-difference function must return finite values")
    jacobian = np.zeros((base.size, point.size), dtype=float)
    relative_step = np.cbrt(np.finfo(float).eps)
    for column, value in enumerate(point):
        step = relative_step * max(1.0, abs(float(value)))
        plus_room = float(hi[column] - value)
        minus_room = float(value - lo[column])
        if plus_room >= step and minus_room >= step:
            plus = point.copy()
            minus = point.copy()
            plus[column] += step
            minus[column] -= step
            jacobian[:, column] = (
                np.asarray(function(plus), dtype=float).reshape(-1)
                - np.asarray(function(minus), dtype=float).reshape(-1)
            ) / (2.0 * step)
        elif plus_room > 0.0:
            h = min(step, plus_room)
            plus = point.copy()
            plus[column] += h
            jacobian[:, column] = (
                np.asarray(function(plus), dtype=float).reshape(-1) - base
            ) / h
        elif minus_room > 0.0:
            h = min(step, minus_room)
            minus = point.copy()
            minus[column] -= h
            jacobian[:, column] = (
                base - np.asarray(function(minus), dtype=float).reshape(-1)
            ) / h
    if not np.all(np.isfinite(jacobian)):
        raise ValueError("finite-difference Jacobian contains non-finite values")
    return jacobian


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
    background_model="linear", background_degree=2,
    background_interior_knots=3, background_templates=None,
    physical_model="maxwellian", kappa_index=None, fit_kappa_index=False,
    incident_polarization=None, radial_order=24, angular_order=12,
    maximum_recoil_correction_fraction=0.01,
    include_compton_recoil=False,
):
    """Jointly infer Te(r), projected flow(r), and spectral amplitude(r).

    Data layout is channel x wavelength; covariance is flattened in
    channel-major order. The supplied mixing matrix maps spatial nodes to
    detector-space backgrounds are fitted jointly as nuisance basis
    coefficients and are not mixed as plasma emission. Select constant,
    linear, polynomial, cubic-spline, or researcher-supplied template bases.
    """
    if physical_model not in {"maxwellian", "kappa", "relativistic_maxwell_juttner"}:
        raise ValueError("physical_model must be maxwellian, kappa, or relativistic_maxwell_juttner")
    if not isinstance(include_compton_recoil, (bool, np.bool_)):
        raise ValueError("include_compton_recoil must be boolean")
    if include_compton_recoil and physical_model != "relativistic_maxwell_juttner":
        raise ValueError("Compton recoil is available only for relativistic_maxwell_juttner")
    if not isinstance(fit_kappa_index, (bool, np.bool_)):
        raise ValueError("fit_kappa_index must be boolean")
    if physical_model == "kappa":
        if (kappa_index is None or isinstance(kappa_index, (bool, np.bool_))
                or not isinstance(kappa_index, (int, float, np.integer, np.floating))
                or not np.isfinite(kappa_index) or float(kappa_index) <= 1.5):
            raise ValueError("kappa_index must be a finite real value > 1.5 and acts as the optimizer seed or fixed model value")
        kappa_index = float(kappa_index)
    elif kappa_index is not None or fit_kappa_index:
        raise ValueError("kappa_index and fit_kappa_index are used only when physical_model='kappa'")
    if fit_kappa_index and physical_model != "kappa":
        raise ValueError("fit_kappa_index=True requires physical_model='kappa'")
    if physical_model == "relativistic_maxwell_juttner":
        if incident_polarization is None:
            raise ValueError("relativistic Maxwell-Juttner joint inference requires incident_polarization")
        polarization = _real_array(incident_polarization, "incident_polarization")
        if polarization.shape != (3,) or np.linalg.norm(polarization) == 0:
            raise ValueError("incident_polarization must be a finite non-zero 3-vector")
        incident_polarization = polarization / np.linalg.norm(polarization)
        if isinstance(radial_order, (bool, np.bool_)) or not isinstance(radial_order, (int, np.integer)) or radial_order < 12:
            raise ValueError("radial_order must be an integer >= 12")
        if isinstance(angular_order, (bool, np.bool_)) or not isinstance(angular_order, (int, np.integer)) or angular_order < 8:
            raise ValueError("angular_order must be an integer >= 8")
    elif incident_polarization is not None:
        raise ValueError("incident_polarization is used only for relativistic_maxwell_juttner")
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

    from .thomson_background import _background_design
    background_basis = _background_design(
        x, background_model, degree=background_degree,
        interior_knots=background_interior_knots,
        templates=background_templates,
    )
    n_background = int(background_basis.shape[1])
    widths = _bin_widths(x)
    response = _response_stack(instrument_response, m, nlam)
    q = 2.0 * np.sin(angle / 2.0)
    d2 = _curvature(r)
    span = float(x[-1] - x[0])
    spectral_spacing = float(np.min(np.diff(x)))
    peak_scale = max(float(np.max(y) - np.median(y)), np.finfo(float).tiny)
    initial_sigma = max(span / 3.0, spectral_spacing)
    # Data-informed numerical seed only.  Do not clip this estimate to an
    # assumed scientific temperature interval; the selected Maxwellian model's
    # declared asymptotic-domain screen is applied to the fitted result below.
    initial_te = _MEC2_EV * (initial_sigma / (2.0 * laser * np.sin(angle/2.0)))**2
    initial_te = float(max(initial_te, np.finfo(float).tiny))
    initial_amp = peak_scale / max(float(np.max(h.sum(axis=1))), 1.0)

    # Parameter vector: log amplitude[n], log Te[n], flow[1e6 m/s,n],
    # optional log(kappa-1.5), then detector-space background coefficients.
    background_seed = np.zeros((m, n_background), dtype=float)
    background_seed[:, 0] = np.median(y[:, :max(2, nlam//10)], axis=1)
    kappa_parameter_count = int(physical_model == "kappa" and fit_kappa_index)
    p0 = np.r_[np.full(r.size, np.log(initial_amp)),
               np.full(r.size, np.log(initial_te)),
               np.zeros(r.size),
               np.array([np.log(kappa_index-1.5)]) if kappa_parameter_count else np.empty(0),
               background_seed.ravel()]
    # Only numerical overflow guards are imposed on logarithmic positive
    # parameters.  Temperature and flow have no invented physical fit bounds;
    # the Maxwellian model-domain screen below classifies out-of-regime fits.
    # Background nuisance parameters and projected flow are unbounded.
    lower = np.r_[np.full(r.size, -700.0), np.full(r.size, -700.0),
                  np.full(r.size, -np.inf),
                  np.array([-10.0]) if kappa_parameter_count else np.empty(0),
                  np.full(m*n_background, -np.inf)]
    upper = np.r_[np.full(r.size, 700.0), np.full(r.size, 700.0),
                  np.full(r.size, np.inf),
                  np.array([8.0]) if kappa_parameter_count else np.empty(0),
                  np.full(m*n_background, np.inf)]
    if physical_model == "relativistic_maxwell_juttner":
        # Subluminal flow is a physical domain constraint of the relativistic kernel.
        lower[2*r.size:3*r.size] = -0.99*_C_M_S/1e6
        upper[2*r.size:3*r.size] = 0.99*_C_M_S/1e6
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
        fitted_kappa = (1.5+float(np.exp(p[3*r.size]))
                        if kappa_parameter_count else kappa_index)
        background_start = 3*r.size+kappa_parameter_count
        background_coefficients = p[background_start:].reshape(m, n_background)
        detector_background = background_coefficients @ background_basis.T
        sigma = 2.0 * laser * np.sqrt(te / _MEC2_EV) * np.sin(angle/2.0)
        center = laser * (1.0 - q * flow / _C_M_S)
        if physical_model == "maxwellian":
            node_spectra = amp[:, None] * np.exp(
                -0.5 * ((x[None, :] - center[:, None]) / sigma[:, None])**2
            )
        elif physical_model == "kappa":
            # The isotropic 3-D finite-variance Kappa marginal along q is a
            # Student-like 1-D profile with exponent kappa. sigma remains its
            # standard deviation, so the scale is converted consistently.
            from scipy.special import gammaln
            scale = sigma * np.sqrt((2.0*fitted_kappa-3.0)/fitted_kappa)
            normalization = np.exp(
                gammaln(fitted_kappa)
                - 0.5*np.log(np.pi*fitted_kappa)
                - np.log(scale)
                - gammaln(fitted_kappa-0.5)
            )
            z = (x[None,:]-center[:,None])/scale[:,None]
            node_spectra = amp[:,None] * normalization[:,None] * (
                1.0 + z*z/fitted_kappa
            )**(-fitted_kappa)
        else:
            from .thomson_kinetic_models import relativistic_maxwell_juttner_spectrum
            node_spectra = np.empty((r.size,x.size),dtype=float)
            for node in range(r.size):
                spectrum_result = relativistic_maxwell_juttner_spectrum(
                    x, laser, angle, float(te[node]),
                    flow_projection_m_s=float(flow[node]),
                    incident_polarization=incident_polarization,
                    radial_order=int(radial_order), angular_order=int(angular_order),
                    maximum_recoil_correction_fraction=maximum_recoil_correction_fraction,
                    include_compton_recoil=include_compton_recoil,
                )
                node_spectra[node] = amp[node] * np.asarray(
                    spectrum_result["spectral_density_per_nm"],dtype=float
                )
        mixed = h @ node_spectra
        predicted = np.empty_like(y)
        for channel in range(m):
            # Response acts on spectral density through bin-integrated power.
            convolved = (response[channel] @ (mixed[channel] * widths)) / widths
            predicted[channel] = convolved + detector_background[channel]
        return predicted, te, flow, amp, fitted_kappa

    def residual(p):
        predicted, te, flow, amp, _ = predict(p)
        data = whiten(y - predicted)
        if lam_reg > 0:
            penalties = np.r_[np.sqrt(lam_reg) * (d2 @ np.log(te)),
                              np.sqrt(lam_reg) * (d2 @ flow / 1e6),
                              np.sqrt(lam_reg) * (d2 @ np.log(amp))]
            return np.r_[data, penalties]
        return data

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=int(max_nfev))
    predicted, te, flow, amp, fitted_kappa = predict(fit.x)
    data_residual = whiten(y - predicted)
    # Assess identifiability from measured spectral data alone.  The optimizer
    # Jacobian also contains optional regularization rows; counting those rows
    # as independent measurements would let smoothing manufacture rank.
    def data_residual_only(parameters):
        return whiten(y - predict(parameters)[0])

    parameter_count = int(fit.x.size)
    background_start = 3*n + kappa_parameter_count
    fitted_background_coefficients = fit.x[background_start:].reshape(m, n_background).copy()
    data_jacobian = _bound_aware_jacobian(
        data_residual_only, fit.x, lower, upper
    )
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
    # Apply the same conservative non-relativistic asymptotic screens as the
    # channel-wise Maxwellian inference. These are numerical error-budget
    # screens, not universal physical validity boundaries.
    from .thomson_inference import (
        ELECTRON_REST_ENERGY_EV, NONRELATIVE_THETA_E_SCREEN,
        NONRELATIVE_FLOW_BETA_SCREEN,
    )
    theta_e = te / ELECTRON_REST_ENERGY_EV
    flow_beta = np.abs(flow) / _C_M_S
    if physical_model in {"maxwellian", "kappa"}:
        model_domain_valid = bool(
            np.all(theta_e < NONRELATIVE_THETA_E_SCREEN)
            and np.all(flow_beta < NONRELATIVE_FLOW_BETA_SCREEN)
        )
    else:
        model_domain_valid = bool(np.all(flow_beta < 1.0))
    data_identifiable = bool(fit.success and data_rank == parameter_count and model_domain_valid)
    if fit.success and not model_domain_valid:
        primary_status = "model_domain_rejected"
    elif data_identifiable:
        primary_status = "identified_from_measured_data"
    else:
        primary_status = "not_identifiable_from_measured_data"
    dof = max(1, data_residual.size - parameter_count)
    covariance_p = np.linalg.pinv(optimizer_jacobian.T @ optimizer_jacobian)
    if joint_spectral_covariance is None and signal_uncertainty is None:
        covariance_p *= float(data_residual @ data_residual / dof)
    transform = np.diag(np.r_[amp, te, np.full(r.size, 1e6),
                              np.array([fitted_kappa-1.5]) if kappa_parameter_count else np.empty(0),
                              np.ones(m*n_background)])
    physical_covariance = transform @ covariance_p @ transform.T
    diagonal = np.maximum(np.diag(physical_covariance), 0.0)
    uncertainties = np.sqrt(diagonal)
    n = r.size
    return {
        "analysis_status": ("model_domain_rejected" if fit.success and not model_domain_valid else "completed" if fit.success else "optimizer_not_converged"),
        "primary_result_status": primary_status,
        "primary_outputs_reportable_as_identified": data_identifiable,
        "model": {
            "maxwellian": "isotropic_nonrelativistic_maxwellian_single_particle_thomson",
            "kappa": "isotropic_finite_variance_kappa_projected_single_particle_thomson",
            "relativistic_maxwell_juttner": "isotropic_relativistic_maxwell_juttner_single_particle_thomson",
        }[physical_model],
        "physical_model_selection": {
            "model": physical_model,
            "kappa_index": fitted_kappa if physical_model == "kappa" else None,
            "kappa_index_initial_seed": kappa_index if physical_model == "kappa" else None,
            "kappa_index_role": ("inferred_from_measured_spectra_with_profiled_uncertainty" if kappa_parameter_count else "fixed_external_model_parameter_not_inferred") if physical_model == "kappa" else None,
            "incident_polarization": incident_polarization,
            "include_compton_recoil": bool(include_compton_recoil),
            "maximum_recoil_correction_fraction": float(maximum_recoil_correction_fraction),
            "relativistic_quadrature": {"radial_order": int(radial_order), "angular_order": int(angular_order)} if physical_model == "relativistic_maxwell_juttner" else None,
            "amplitude_semantics": "Gaussian peak amplitude" if physical_model == "maxwellian" else "relative integrated spectral amplitude",
        },
        "validity_screening": {
            "theta_e_profile": theta_e,
            "theta_e_limit": NONRELATIVE_THETA_E_SCREEN,
            "bulk_flow_beta_profile": flow_beta,
            "bulk_flow_beta_limit": NONRELATIVE_FLOW_BETA_SCREEN,
            "model_domain_valid": model_domain_valid,
            "rationale": ("Conservative asymptotic screens apply to the non-relativistic Maxwellian/Kappa kernels; the Maxwell-Juttner kernel uses relativistic kinematics and instead requires subluminal flow. These are declared model-domain checks, not universal validity boundaries."),
        },
        "radius_m": r.copy(),
        "kappa_index_uncertainty": (float(uncertainties[3*n]) if data_identifiable and kappa_parameter_count else None),
        "kappa_index_status": ("primary" if data_identifiable and kappa_parameter_count else "conditional_fixed_external_model_parameter" if physical_model == "kappa" else "not_applicable"),
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
            "kappa_index": fitted_kappa if physical_model == "kappa" else None,
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
                            (["kappa_index"] if kappa_parameter_count else []) +
                            [f"background_channel_{channel}_coefficient_{coefficient}"
                             for channel in range(m)
                             for coefficient in range(n_background)]),
        "detector_background_model": background_model,
        "detector_background_basis": background_basis.copy(),
        "detector_background_coefficients": fitted_background_coefficients,
        "detector_background_offset": (
            fitted_background_coefficients[:, 0].copy()
            if background_model in {"linear", "polynomial"} else None
        ),
        "detector_background_slope_per_nm": (
            fitted_background_coefficients[:, 1].copy()
            * (2.0 / float(np.ptp(x))) if background_model == "linear" else None
        ),
        "spatial_mixing_operator": "researcher_supplied_fixed_forward_operator",
        "instrument_response": "researcher_supplied_fixed_forward_operator",
        "regularization": {"kind": "nonuniform_radius_curvature_on_log_Te_flow_and_log_amplitude",
                           "strength": lam_reg, "user_selected": True},
        "noise_model": noise_model,
        "background_contract": {
            "model": background_model,
            "coefficient_count_per_channel": n_background,
            "role": "detector_space_nuisance_not_plasma_emission",
            "basis_source": "researcher_selected_model_or_supplied_templates",
            "fit_is_joint_with_physical_thomson_forward_model": True,
        },
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
            "primary": [f"joint finite-dimensional Te(r), scattering-vector flow projection(r), and relative amplitude(r) under the supplied {physical_model} model and fixed operators"],
            "conditional": ["absolute density only if independent absolute-power and complete optical-geometry inputs are subsequently supplied; this function does not infer it"],
            "non_identifiable": ["flow components perpendicular to the scattering vector",
                                 "spatial structures in the null space of the supplied mixing operator",
                                 "sub-resolution spatial structure not encoded by the measured mixing operator",
                                 "unrestricted anisotropic or non-Maxwellian electron distributions"]},
        "uncertainty_semantics": f"local linearized covariance conditional on the selected {physical_model} model, fixed spatial/spectral response operators, supplied noise model, and user-selected regularization; inspect primary_result_status because regularization is excluded from the data-only identifiability rank",
        "parameter_identifiability": {
            "primary": ["electron_temperature_profile_eV", "electron_velocity_projection_profile_m_s", "relative_spectral_amplitude_profile"] if data_identifiable else [],
            "conditional": ["absolute electron density requires a separate absolute-power and optical-collection inference with its complete auxiliary inputs"],
            "non_identifiable": ["perpendicular flow components", "spatial null-space and sub-resolution structures", "unrestricted anisotropic or non-Maxwellian electron distributions"],
            "status": "identified_from_measured_data" if data_identifiable else "not_identifiable_from_measured_data",
        },
        "experimental_validation": "not_performed; numerical software inference only",
    }

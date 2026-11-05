"""Joint multi-view optical Thomson inference for a gyrotropic bi-Maxwellian.

The model fits calibrated measured spectra directly. Each channel measures the
one-dimensional marginal along its own scattering vector; independent magnetic
pitch angles are required to separate parallel and perpendicular temperatures.
This is a non-relativistic elastic-Thomson small-beta forward model.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.special import gammaln, ndtr, stdtr


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

def _bin_edges(axis):
    """Return finite-volume edges implied by calibrated wavelength centres."""
    wavelength = _real_array(axis, "wavelength_nm", 1)
    if wavelength.size < 2 or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must contain at least two strictly increasing centres")
    edges = np.empty(wavelength.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    return edges


def _gaussian_bin_average(wavelength_nm, centre_nm, sigma_nm):
    """Exact Gaussian probability per finite wavelength bin, divided by bin width."""
    edges = _bin_edges(wavelength_nm)
    sigma = float(sigma_nm)
    centre = float(centre_nm)
    if not np.isfinite(sigma) or sigma <= 0.0 or not np.isfinite(centre):
        raise ValueError("Gaussian centre must be finite and sigma must be positive")
    standardized_edges = (edges - centre) / sigma
    lower_cdf = ndtr(standardized_edges[:-1])
    upper_cdf = ndtr(standardized_edges[1:])
    probability = upper_cdf - lower_cdf
    # Avoid cancellation when both bin edges lie deep in the positive tail.
    positive_tail = standardized_edges[:-1] > 0.0
    probability[positive_tail] = (
        ndtr(-standardized_edges[:-1][positive_tail])
        - ndtr(-standardized_edges[1:][positive_tail])
    )
    widths = np.diff(edges)
    return probability / widths, probability


def _kappa_bin_average(wavelength_nm, laser_wavelength_nm, scattering_q_factor,
                       flow_m_s, theta_m_s, kappa):
    """Exact finite-bin probability for the projected 1-D Kappa/Student-t marginal.

    The projected Kappa marginal is Student-t with nu=2*kappa-1 degrees of
    freedom and scale s²=kappa*theta²/(2*kappa-1). The wavelength-to-velocity
    map is the declared non-relativistic elastic-Thomson map.
    """
    edges = _bin_edges(wavelength_nm)
    laser = float(laser_wavelength_nm)
    q_factor = float(scattering_q_factor)
    flow = float(flow_m_s)
    theta = float(theta_m_s)
    kappa_value = float(kappa)
    if (not np.all(np.isfinite([laser, q_factor, flow, theta, kappa_value]))
            or laser <= 0.0 or q_factor <= 0.0 or theta <= 0.0
            or kappa_value <= 1.5):
        raise ValueError("Kappa bin integration requires positive laser/q/theta and kappa > 3/2")
    velocity_edges = (_C / q_factor) * (1.0 - edges / laser) - flow
    degrees_of_freedom = 2.0 * kappa_value - 1.0
    student_scale = theta * np.sqrt(kappa_value / degrees_of_freedom)
    # Wavelength edges map in descending velocity order.
    upper_standardized = velocity_edges[:-1] / student_scale
    lower_standardized = velocity_edges[1:] / student_scale
    probability = (
        stdtr(degrees_of_freedom, upper_standardized)
        - stdtr(degrees_of_freedom, lower_standardized)
    )
    # Use survival probabilities for positive-tail intervals to avoid 1-1
    # cancellation in the Student-t CDF.
    positive_tail = lower_standardized > 0.0
    probability[positive_tail] = (
        stdtr(degrees_of_freedom, -lower_standardized[positive_tail])
        - stdtr(degrees_of_freedom, -upper_standardized[positive_tail])
    )
    widths = np.diff(edges)
    probability = np.maximum(probability, 0.0)
    return probability / widths, probability


def _apply_response_in_bin_integrated_space(response_matrix, spectral_density, wavelength_nm):
    """Apply a measured bin-probability response while conserving spectral power.

    Response columns map true wavelength-bin probabilities to measured bins.
    Therefore the operator acts on bin-integrated spectral power, not directly
    on spectral-density samples. This distinction is essential on nonuniform
    wavelength grids.
    """
    wavelength = _real_array(wavelength_nm, "wavelength_nm", 1)
    density = _real_array(spectral_density, "spectral_density", 1)
    response = _real_array(response_matrix, "response_matrix", 2)
    if wavelength.size < 2 or density.shape != wavelength.shape:
        raise ValueError("wavelength_nm and spectral_density must be paired 1-D arrays")
    if np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must be strictly increasing")
    if response.shape != (wavelength.size, wavelength.size) or np.any(response < 0.0):
        raise ValueError("response_matrix must be a non-negative square matrix on the wavelength grid")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("response_matrix columns must sum to one")
    widths = _bin_widths(wavelength)
    return (response @ (density * widths)) / widths


def fit_gyrotropic_bi_maxwellian_thomson(
    wavelength_nm, measured_spectra, *, laser_wavelength_nm,
    scattering_angle_rad, magnetic_field_unit, scattering_vector_unit,
    instrument_response_matrix, measurement_covariance=None,
    signal_uncertainty=None, model_discrepancy_covariance=None,
    initial_temperature_parallel_eV=1000.0,
    initial_temperature_perpendicular_eV=1000.0,
    temperature_bounds_eV=(1.0e-3, 1.0e7),
    flow_bounds_m_s=(-0.05 * _C, 0.05 * _C),
    maximum_projected_rms_beta=0.1,
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
    measurement_covariance_supplied = (
        measurement_covariance is not None or signal_uncertainty is not None
    )
    if measurement_covariance is not None:
        measurement_covariance_only = _real_array(
            measurement_covariance, "measurement_covariance", 2
        )
        if (measurement_covariance_only.shape != (size, size)
                or not np.allclose(
                    measurement_covariance_only, measurement_covariance_only.T,
                    rtol=1e-10, atol=1e-12 * max(
                        float(np.max(np.abs(measurement_covariance_only))), 1e-300
                    )
                )):
            raise ValueError("measurement_covariance must be symmetric and channel-major")
        measurement_covariance_only = 0.5 * (
            measurement_covariance_only + measurement_covariance_only.T
        )
    elif signal_uncertainty is not None:
        sigma = _real_array(signal_uncertainty, "signal_uncertainty", 2)
        if sigma.shape != spectra.shape or np.any(sigma <= 0.0):
            raise ValueError("signal_uncertainty must be positive and match measured_spectra")
        measurement_covariance_only = np.diag(sigma.ravel() ** 2)
    else:
        measurement_covariance_only = None

    discrepancy = None
    if model_discrepancy_covariance is not None:
        discrepancy = _real_array(
            model_discrepancy_covariance, "model_discrepancy_covariance", 2
        )
        if (discrepancy.shape != (size, size)
                or not np.allclose(
                    discrepancy, discrepancy.T, rtol=1e-10,
                    atol=1e-12 * max(float(np.max(np.abs(discrepancy))), 1e-300)
                )):
            raise ValueError("model_discrepancy_covariance must be symmetric and channel-major")
        discrepancy = 0.5 * (discrepancy + discrepancy.T)
        eig = np.linalg.eigvalsh(discrepancy)
        if eig[0] < -1e-10 * max(float(np.max(np.abs(eig))), 1e-300):
            raise ValueError("model_discrepancy_covariance must be positive semidefinite")

    covariance_components = [
        component for component in (measurement_covariance_only, discrepancy)
        if component is not None
    ]
    if covariance_components:
        cov = np.sum(covariance_components, axis=0)
        covariance_model = (
            "measurement_plus_model_discrepancy"
            if measurement_covariance_only is not None and discrepancy is not None
            else "measurement_only" if measurement_covariance_only is not None
            else "model_discrepancy_only"
        )
    else:
        # Unit weighting is an optimizer convention, not a measured covariance.
        cov = np.eye(size)
        covariance_model = "unweighted_identity_metric_residual_scaled"
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
    beta_limit = float(maximum_projected_rms_beta)
    if not np.isfinite(beta_limit) or not 0.0 < beta_limit < 1.0:
        raise ValueError("maximum_projected_rms_beta must lie strictly between zero and one")
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
        captured_probability = np.empty(n_view, dtype=float)
        if np.any(directional <= 0.0):
            raise ValueError("directional bi-Maxwellian temperature became non-positive")
        for i in range(n_view):
            flow, log_amp, offset, slope = parameters[2 + 4*i: 6 + 4*i]
            sigma_v = np.sqrt(_E * directional[i] / _ME)
            center = laser * (1.0 - qmag_factor * flow / _C)
            sigma_lambda = laser * qmag_factor * sigma_v / _C
            if sigma_lambda <= np.finfo(float).tiny:
                raise ValueError("predicted Thomson spectral width is numerically zero")
            intrinsic, bin_probability = _gaussian_bin_average(
                wavelength, center, sigma_lambda
            )
            captured_probability[i] = float(np.sum(bin_probability))
            detector = _apply_response_in_bin_integrated_space(response[i], intrinsic, wavelength)
            predictions[i] = np.exp(log_amp) * detector + offset + slope * (wavelength - laser)
        return predictions, directional, captured_probability

    def residual(parameters):
        return whiten(predict(parameters)[0] - spectra)

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"gyrotropic bi-Maxwellian Thomson fit failed: {fit.message}")
    fitted, directional, captured_probability = predict(fit.x)
    projected_rms_beta = np.sqrt(_E * directional / _ME) / _C
    fitted_flow_beta = np.abs(fit.x[2::4]) / _C
    if np.max(projected_rms_beta) >= beta_limit or np.max(fitted_flow_beta) >= 0.1:
        raise ValueError(
            "Fitted thermal or bulk-flow speed exceeds the declared non-relativistic "
            "validity screen; use a relativistic forward model."
        )
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    tol = np.finfo(float).eps * max(fit.jac.shape) * (singular[0] if singular.size else 0.0)
    full_rank = int(np.count_nonzero(singular > tol)) == fit.x.size
    parameter_covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-12) if (full_rank and geometry_rank == 2) else None
    covariance_scale_method = "supplied_absolute_covariance"
    if parameter_covariance is not None and not measurement_covariance_supplied and discrepancy is None:
        dof = max(1, size - fit.x.size)
        parameter_covariance *= (2.0 * float(fit.cost) / dof)
        covariance_scale_method = "unweighted_residual_scaled"
    elif parameter_covariance is not None and discrepancy is not None and not measurement_covariance_supplied:
        covariance_scale_method = "absolute_model_discrepancy_covariance"
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
            "finite_bin_treatment": "exact Gaussian CDF integration over midpoint-defined wavelength-bin edges before measured response convolution",
            "validity_note": "Use a relativistic forward model when thermal or bulk electron speeds make the small-beta approximation inadequate.",
        },
        "electron_temperature_parallel_eV": tpar,
        "electron_temperature_perpendicular_eV": tperp,
        "directional_temperature_eV": directional,
        "captured_spectral_probability": captured_probability,
        "spectral_window_coverage_warning": (
            "The measured wavelength window truncates more than one percent of the fitted Maxwellian feature in at least one view."
            if np.any(captured_probability < 0.99) else None
        ),
        "validity_screening": {
            "maximum_projected_rms_beta": beta_limit,
            "maximum_fitted_projected_rms_beta": float(np.max(projected_rms_beta)),
            "maximum_fitted_flow_beta": float(np.max(fitted_flow_beta)),
            "status": "within_declared_nonrelativistic_screen",
            "interpretation": "Conservative software screening threshold, not a universal physical boundary."
        },
        "spectral_bin_integration": "exact_Gaussian_CDF_probability_per_finite_wavelength_bin",
        "magnetic_pitch_cosine": mu,
        "magnetic_pitch_cosine_squared": mu2,
        "geometry_design_rank": geometry_rank,
        "anisotropy_identifiability": "identified" if reportable_anisotropy else "non_identifiable",
        "reportable_as_inferred_parameter": bool(reportable_anisotropy),
        "non_identifiability_reason": None if reportable_anisotropy else "Distinct squared magnetic pitch angles and a full-rank spectral fit are required to identify both T_parallel and T_perpendicular.",
        "electron_velocity_projection_m_s": fit.x[2::4].copy(),
        "fitted_spectra": fitted,
        "residual_spectra": spectra - fitted,
        "measurement_covariance": (
            None if measurement_covariance_only is None
            else measurement_covariance_only.copy()
        ),
        "model_discrepancy_covariance": (
            None if discrepancy is None else discrepancy.copy()
        ),
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
                "supplied_pointwise_diagonal" if signal_uncertainty is not None else "not_supplied"
            ),
            "effective_covariance_model": covariance_model,
            "parameter_covariance_scale": covariance_scale_method,
            "model_discrepancy_supplied": discrepancy is not None,
            "combination": "C_effective is the sum of only supplied covariance components; an identity metric is used only when neither component is supplied",
        },
        "experimental_validation": "not_performed",
    }


def fit_gyrotropic_bi_kappa_thomson(
    wavelength_nm, measured_spectra, *, laser_wavelength_nm,
    scattering_angle_rad, magnetic_field_unit, scattering_vector_unit,
    instrument_response_matrix, measurement_covariance=None,
    signal_uncertainty=None, model_discrepancy_covariance=None,
    initial_temperature_parallel_eV=1000.0,
    initial_temperature_perpendicular_eV=1000.0,
    initial_kappa=4.0, kappa_bounds=(1.5001, 200.0),
    temperature_bounds_eV=(1.0e-3, 1.0e7),
    flow_bounds_m_s=(-0.05 * _C, 0.05 * _C),
    maximum_projected_rms_beta=0.1,
    max_nfev=6000,
):
    r"""Fit a gyrotropic anisotropic 3-D Kappa distribution to multi-view spectra.

    The assumed normalized 3-D distribution is proportional to
      [1 + v_parallel^2/(kappa theta_parallel^2)
         + v_perp^2/(kappa theta_perp^2)]^(-(kappa+1)).
    Its exact one-dimensional marginal along a view q is
      p(v_q) = Gamma(kappa)/(sqrt(pi*kappa)*Gamma(kappa-1/2)*theta_q)
               [1 + v_q^2/(kappa*theta_q^2)]^(-kappa),
    where theta_q^2 = (2*kappa-3)/kappa * e*T_q/m_e (T_q in eV).
    This parameterization makes T_q the actual second-moment temperature
    (m_e <(v_q-u_q)^2>/e, with T_q in eV), not the Kappa core/scale temperature.
    The finite-variance domain kappa>3/2 is enforced.

    Only the non-relativistic elastic-Thomson wavelength mapping is used.
    T_parallel and T_perpendicular are reported only when the independent
    pitch-angle design and complete whitened spectral Jacobian both identify
    them. A single view cannot identify anisotropy.
    """
    wavelength = _real_array(wavelength_nm, "wavelength_nm", 1)
    spectra = _real_array(measured_spectra, "measured_spectra", 2)
    if wavelength.size < 16 or np.any(wavelength <= 0.0) or np.any(np.diff(wavelength) <= 0.0):
        raise ValueError("wavelength_nm must be positive, increasing, and contain >=16 bins")
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
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured response columns must sum to one; response is never silently renormalized")
    if measurement_covariance is not None and signal_uncertainty is not None:
        raise ValueError("supply measurement_covariance or signal_uncertainty, not both")
    size = n_view * n_bin
    measurement_covariance_supplied = (
        measurement_covariance is not None or signal_uncertainty is not None
    )
    if measurement_covariance is not None:
        measurement_covariance_only = _real_array(
            measurement_covariance, "measurement_covariance", 2
        )
        if (measurement_covariance_only.shape != (size, size)
                or not np.allclose(
                    measurement_covariance_only, measurement_covariance_only.T,
                    rtol=1e-10, atol=1e-12 * max(
                        float(np.max(np.abs(measurement_covariance_only))), 1e-300
                    )
                )):
            raise ValueError("measurement_covariance must be symmetric and channel-major")
        measurement_covariance_only = 0.5 * (
            measurement_covariance_only + measurement_covariance_only.T
        )
    elif signal_uncertainty is not None:
        sigma = _real_array(signal_uncertainty, "signal_uncertainty", 2)
        if sigma.shape != spectra.shape or np.any(sigma <= 0.0):
            raise ValueError("signal_uncertainty must be positive and match measured_spectra")
        measurement_covariance_only = np.diag(sigma.ravel() ** 2)
    else:
        measurement_covariance_only = None

    discrepancy = None
    if model_discrepancy_covariance is not None:
        discrepancy = _real_array(
            model_discrepancy_covariance, "model_discrepancy_covariance", 2
        )
        if (discrepancy.shape != (size, size)
                or not np.allclose(
                    discrepancy, discrepancy.T, rtol=1e-10,
                    atol=1e-12 * max(float(np.max(np.abs(discrepancy))), 1e-300)
                )):
            raise ValueError("model_discrepancy_covariance must be symmetric and channel-major")
        discrepancy = 0.5 * (discrepancy + discrepancy.T)
        eig = np.linalg.eigvalsh(discrepancy)
        if eig[0] < -1e-10 * max(float(np.max(np.abs(eig))), 1e-300):
            raise ValueError("model_discrepancy_covariance must be positive semidefinite")

    covariance_components = [
        component for component in (measurement_covariance_only, discrepancy)
        if component is not None
    ]
    if covariance_components:
        cov = np.sum(covariance_components, axis=0)
        covariance_model = (
            "measurement_plus_model_discrepancy"
            if measurement_covariance_only is not None and discrepancy is not None
            else "measurement_only" if measurement_covariance_only is not None
            else "model_discrepancy_only"
        )
    else:
        cov = np.eye(size)
        covariance_model = "unweighted_identity_metric_residual_scaled"
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("effective spectral covariance must be positive definite") from exc
    def whiten(value):
        return np.linalg.solve(chol, np.asarray(value, dtype=float).ravel())

    t0p, t0t = float(initial_temperature_parallel_eV), float(initial_temperature_perpendicular_eV)
    tlo, thi = map(float, temperature_bounds_eV)
    klo, khi = map(float, kappa_bounds)
    k0 = float(initial_kappa)
    if not (0.0 < tlo < thi and tlo <= t0p <= thi and tlo <= t0t <= thi):
        raise ValueError("initial temperatures must lie inside positive temperature_bounds_eV")
    if not (1.5 < klo < khi and klo <= k0 <= khi):
        raise ValueError("kappa_bounds and initial_kappa must satisfy 1.5 < lower <= initial <= upper")
    beta_limit = float(maximum_projected_rms_beta)
    if not np.isfinite(beta_limit) or not 0.0 < beta_limit < 1.0:
        raise ValueError("maximum_projected_rms_beta must lie strictly between zero and one")
    flow_lo, flow_hi = map(float, flow_bounds_m_s)
    if not np.isfinite(flow_lo + flow_hi) or flow_lo >= flow_hi or flow_lo <= -_C or flow_hi >= _C:
        raise ValueError("flow_bounds_m_s must be ordered and strictly subluminal")
    if isinstance(max_nfev, (bool, np.bool_)) or int(max_nfev) < 1:
        raise ValueError("max_nfev must be a positive integer")

    qmag = 2.0 * np.sin(angle / 2.0)
    widths = _bin_widths(wavelength)
    edge = max(2, n_bin // 10)
    p0 = [np.log(t0p), np.log(t0t), np.log(k0 - 1.5)]
    lower = [np.log(tlo), np.log(tlo), np.log(klo - 1.5)]
    upper = [np.log(thi), np.log(thi), np.log(khi - 1.5)]
    for i in range(n_view):
        y = spectra[i]
        span = max(float(np.ptp(y)), np.finfo(float).tiny)
        baseline = float(np.median(np.r_[y[:edge], y[-edge:]]))
        p0.extend([0.0, np.log(max(float(np.max(y) - baseline), span * 1e-6)), baseline, 0.0])
        lower.extend([flow_lo, -50.0, -np.inf, -np.inf])
        upper.extend([flow_hi, 50.0, np.inf, np.inf])
    p0, lower, upper = np.asarray(p0), np.asarray(lower), np.asarray(upper)

    def predict(parameters):
        tpar, tperp = np.exp(parameters[:2])
        kappa = 1.5 + np.exp(parameters[2])
        directional = tperp + (tpar - tperp) * mu2
        if np.any(directional <= 0.0):
            raise ValueError("directional bi-Kappa temperature became non-positive")
        predictions = np.empty_like(spectra)
        captured_probability = np.empty(n_view, dtype=float)
        projected_velocity_grid = np.empty_like(spectra)
        projected_velocity_pdf = np.empty_like(spectra)
        for i in range(n_view):
            flow, log_amp, offset, slope = parameters[3 + 4*i:7 + 4*i]
            theta2 = ((2.0*kappa - 3.0)/kappa) * (_E*directional[i]/_ME)
            theta = np.sqrt(theta2)
            velocity = (_C/qmag) * (1.0 - wavelength/laser) - flow
            log_norm = (gammaln(kappa) - 0.5*np.log(np.pi*kappa)
                        - gammaln(kappa - 0.5) - np.log(theta))
            log_pdf = log_norm - kappa*np.log1p(velocity**2/(kappa*theta2))
            pdf_velocity = np.exp(log_pdf)
            velocity_order = np.argsort(velocity)
            projected_velocity_grid[i] = velocity[velocity_order]
            projected_velocity_pdf[i] = pdf_velocity[velocity_order]
            intrinsic, bin_probability = _kappa_bin_average(
                wavelength, laser, qmag, flow, theta, kappa
            )
            captured_probability[i] = float(np.sum(bin_probability))
            detector = _apply_response_in_bin_integrated_space(response[i], intrinsic, wavelength)
            predictions[i] = np.exp(log_amp)*detector + offset + slope*(wavelength-laser)
        return predictions, directional, kappa, captured_probability, projected_velocity_grid, projected_velocity_pdf

    def residual(parameters):
        return whiten(predict(parameters)[0] - spectra)

    fit = least_squares(residual, p0, bounds=(lower, upper), method="trf",
                        x_scale="jac", max_nfev=int(max_nfev))
    if not fit.success or not np.all(np.isfinite(fit.x)):
        raise RuntimeError(f"gyrotropic bi-Kappa Thomson fit failed: {fit.message}")
    fitted, directional, kappa, captured_probability, projected_velocity_grid, projected_velocity_pdf = predict(fit.x)
    projected_rms_beta = np.sqrt(_E*directional/_ME)/_C
    fitted_flow_beta = np.abs(fit.x[3::4])/_C
    if np.max(projected_rms_beta) >= beta_limit or np.max(fitted_flow_beta) >= 0.1:
        raise ValueError(
            "Fitted thermal or bulk-flow speed exceeds the declared non-relativistic "
            "validity screen; use a relativistic forward model."
        )
    singular = np.linalg.svd(fit.jac, compute_uv=False)
    tol = np.finfo(float).eps * max(fit.jac.shape) * (singular[0] if singular.size else 0.0)
    full_rank = int(np.count_nonzero(singular > tol)) == fit.x.size
    parameter_covariance = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-12) if (full_rank and geometry_rank == 2) else None
    covariance_scale_method = "supplied_absolute_covariance"
    if parameter_covariance is not None and not measurement_covariance_supplied and discrepancy is None:
        dof = max(1, size - fit.x.size)
        parameter_covariance *= (2.0 * float(fit.cost) / dof)
        covariance_scale_method = "unweighted_residual_scaled"
    elif parameter_covariance is not None and discrepancy is not None and not measurement_covariance_supplied:
        covariance_scale_method = "absolute_model_discrepancy_covariance"
    if parameter_covariance is not None:
        transform = np.eye(fit.x.size)
        transform[0, 0] = np.exp(fit.x[0])
        transform[1, 1] = np.exp(fit.x[1])
        transform[2, 2] = np.exp(fit.x[2])
        parameter_covariance = transform @ parameter_covariance @ transform.T
    reportable = geometry_rank == 2 and full_rank
    return {
        "model": "gyrotropic_bi_kappa_nonrelativistic_optical_thomson",
        "distribution_definition": "f3(v) proportional to [1 + v_parallel^2/(kappa*theta_parallel^2) + v_perp^2/(kappa*theta_perp^2)]^(-(kappa+1))",
        "projected_marginal": "p(v_q)=Gamma(kappa)/(sqrt(pi*kappa)*Gamma(kappa-1/2)*theta_q)*(1+v_q^2/(kappa*theta_q^2))^(-kappa)",
        "temperature_definition": "T_q[eV]=m_e*Var(v_q)/e; theta_q^2=((2*kappa-3)/kappa)*(e*T_q/m_e)",
        "electron_temperature_parallel_eV": float(np.exp(fit.x[0])) if reportable else None,
        "electron_temperature_perpendicular_eV": float(np.exp(fit.x[1])) if reportable else None,
        "kappa_index": float(kappa) if full_rank else None,
        "kappa_finite_variance_domain": bool(kappa > 1.5),
        "directional_temperature_eV": directional,
        "captured_spectral_probability": captured_probability,
        "spectral_bin_integration": "exact_Student_t_CDF_probability_per_finite_wavelength_bin",
        "projected_velocity_grid_m_s": projected_velocity_grid,
        "projected_velocity_distribution_m_s_inverse": projected_velocity_pdf,
        "spectral_window_coverage_warning": (
            "The measured wavelength window truncates more than one percent of the fitted Kappa feature in at least one view."
            if np.any(captured_probability < 0.99) else None
        ),
        "validity_screening": {
            "maximum_projected_rms_beta": beta_limit,
            "maximum_fitted_projected_rms_beta": float(np.max(projected_rms_beta)),
            "maximum_fitted_flow_beta": float(np.max(fitted_flow_beta)),
            "status": "within_declared_nonrelativistic_screen",
            "interpretation": "Conservative software screening threshold, not a universal physical boundary."
        },
        "magnetic_pitch_cosine": mu,
        "magnetic_pitch_cosine_squared": mu2,
        "geometry_design_rank": geometry_rank,
        "anisotropy_identifiability": "identified" if reportable else "non_identifiable",
        "kappa_identifiability": "identified" if full_rank else "non_identifiable",
        "reportable_as_inferred_parameter": bool(reportable and full_rank),
        "non_identifiability_reason": None if reportable else "Distinct squared magnetic pitch angles and a full-rank spectral Jacobian are required to identify both temperatures; kappa also requires a full-rank spectral Jacobian.",
        "electron_velocity_projection_m_s": fit.x[3::4].copy(),
        "fitted_spectra": fitted,
        "residual_spectra": spectra - fitted,
        "effective_covariance": cov,
        "parameter_covariance": parameter_covariance,
        "parameter_order": ["T_parallel_eV", "T_perpendicular_eV", "kappa_minus_1p5"] + [
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
            "effective_covariance_model": covariance_model,
            "parameter_covariance_scale": covariance_scale_method,
            "model_discrepancy_supplied": discrepancy is not None,
            "combination": "C_effective is the sum of only supplied covariance components; an identity metric is used only when neither component is supplied",
        },
        "model_validity": {
            "kinematics": "non-relativistic elastic Thomson mapping",
            "distribution": "gyrotropic bi-Kappa with zero heat flux and finite second moments",
            "kappa_domain": "kappa > 3/2",
            "relativistic_corrections": False,
            "limitations": ["No relativistic Doppler/cross-section correction", "No unrestricted 3-D EVDF reconstruction", "Kappa is a fitted parametric closure, not a general measured distribution"],
        },
        "experimental_validation": "not_performed",
    }

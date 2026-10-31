"""End-to-end, spatially resolved Thomson spectral inference.

Input spectra are measured, wavelength-calibrated and intensity-calibrated data.
This module does not calibrate detectors or repair/prepare measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln, stdtr

from .thomson_scattering import density_from_thomson_scattered_power, _covariance_is_positive_semidefinite
from .formula_library import (
    thomson_electron_temperature_from_spectral_width,
    thomson_velocity_projection,
)
from .thomson_kinetic_models import (
    fit_relativistic_maxwell_juttner,
    fit_collective_kappa,
)

ELECTRON_REST_ENERGY_EV = 510_998.95069
C_M_S = 299_792_458.0
# Conservative asymptotic screening policy: theta_e bounds leading thermal
# relativistic corrections (~theta_e), while beta_flow bounds leading bulk
# kinematic corrections (~beta_flow^2). These are software screening limits,
# not universal physical validity boundaries.
NONRELATIVE_THETA_E_SCREEN = 0.01
NONRELATIVE_FLOW_BETA_SCREEN = 0.1
R_E_M = 2.8179403262e-15


def _finite_array(value, name, ndim=None):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values, not booleans, strings or complex values")
    if ndim is not None and raw.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    a = raw.astype(float, copy=False)
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite numeric values")
    return a


def _finite_scalar(value, name, *, minimum=None, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be greater than or equal to {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be less than or equal to {maximum}")
    return result


def _response_for_channel(response, channel, nchan, nlam, wavelength=None):
    if callable(response):
        if wavelength is None:
            raise ValueError("wavelength_nm is required when instrument_response is callable")
        response = response(np.asarray(wavelength,dtype=float))
    r = _finite_array(response, "instrument_response")
    if r.ndim == 1:
        if r.size < 1 or r.size % 2 == 0 or np.any(r < 0) or np.sum(r) <= 0:
            raise ValueError("a 1-D instrument-response kernel must be non-negative, odd-length and have positive area")
        if wavelength is not None:
            wavelength_steps = np.diff(np.asarray(wavelength, dtype=float))
            if not np.allclose(wavelength_steps, wavelength_steps[0], rtol=1e-6, atol=1e-12):
                raise ValueError("a 1-D index-space response kernel requires a uniform wavelength grid; use a measured wavelength-to-wavelength response matrix for nonuniform grids")
        kernel = r/np.sum(r)
        out = np.zeros((nlam,nlam),dtype=float)
        center = kernel.size//2
        for col in range(nlam):
            for k,value in enumerate(kernel):
                row=col+(k-center)
                if 0 <= row < nlam:
                    out[row,col]+=value
    elif r.ndim == 2:
        out = r
    elif r.ndim == 3 and r.shape[0] == nchan:
        out = r[channel]
    else:
        raise ValueError("instrument_response must be a callable, a 1-D kernel, a 2-D response matrix, or a channel-specific 3-D response matrix")
    if out.shape != (nlam,nlam) or np.any(out < 0):
        raise ValueError("instrument response must be a non-negative square matrix on the measured wavelength grid")
    sums = out.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError("each instrument-response column must have positive normalization")
    return out / sums[None, :]


def _apply_response(response, spectrum, wavelength):
    """Apply a bin-probability response to spectral-density samples.

    The response columns describe probabilities from true wavelength bins to
    measured bins. Convert spectral density to bin-integrated power before the
    matrix operation, then divide by measured-bin widths. This reduces to the
    usual matrix product for a uniform wavelength grid.
    """
    x = np.asarray(wavelength, dtype=float)
    values = np.asarray(spectrum, dtype=float)
    edges = np.empty(x.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (x[:-1] + x[1:])
    edges[0] = x[0] - 0.5 * (x[1] - x[0])
    edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])
    widths = np.diff(edges)
    if np.any(widths <= 0) or values.shape != x.shape:
        raise ValueError("Wavelength-bin widths and spectrum shape must be valid")
    return (response @ (values * widths)) / widths


def _gaussian(x, amp, center, sigma, offset, slope):
    return amp * np.exp(-0.5 * ((x-center)/sigma)**2) + offset + slope*(x-center)



def _fit_identifiability(
    covariance, residual, uncertainty, parameter_count, *,
    optimizer_success=True, jacobian_rank=None, jacobian_condition_number=None,
):
    """Reject failed optimizers and non-identifiable fits, not merely malformed arrays."""
    cov = np.asarray(covariance, dtype=float)
    residual = np.asarray(residual, dtype=float)
    reasons = []
    if not optimizer_success:
        reasons.append("nonlinear optimizer did not report successful convergence")
    if cov.shape != (parameter_count, parameter_count) or not np.all(np.isfinite(cov)):
        reasons.append("covariance is missing, non-finite or has an invalid shape")
        condition = None
    else:
        diagonal = np.diag(cov)
        if np.any(diagonal <= 0):
            reasons.append("one or more fitted parameter variances are non-positive")
        condition = None
        if np.all(diagonal > 0):
            scale = np.sqrt(diagonal)
            correlation = cov / np.outer(scale, scale)
            condition = float(np.linalg.cond(correlation))
            if np.linalg.matrix_rank(correlation) < parameter_count:
                reasons.append("parameter covariance correlation matrix is rank deficient")
            if not np.isfinite(condition) or condition > 1e14:
                reasons.append("parameter covariance correlation matrix is ill-conditioned")
    if not np.all(np.isfinite(residual)):
        reasons.append("residual contains non-finite values")
    if jacobian_rank is not None and int(jacobian_rank) < parameter_count:
        reasons.append("fit Jacobian is rank deficient; fitted parameters are not jointly identifiable")
    if jacobian_condition_number is not None and (
        not np.isfinite(jacobian_condition_number) or float(jacobian_condition_number) > 1e12
    ):
        reasons.append("fit Jacobian is ill-conditioned")
    dof = max(1, residual.size - parameter_count)
    chi2 = None
    if uncertainty is not None and np.all(np.isfinite(residual)):
        u = np.asarray(uncertainty, dtype=float)
        if u.shape != residual.shape or np.any(~np.isfinite(u)) or np.any(u <= 0):
            reasons.append("fit uncertainty is invalid or does not match the residual")
        else:
            chi2 = float(np.sum((residual / u) ** 2) / dof)
            if not np.isfinite(chi2):
                reasons.append("reduced chi-square is non-finite")
            # Reduced chi-square is retained as a model-adequacy diagnostic and
            # contributes to the separately reported discrepancy covariance.
            # It is not used as a hard acceptance threshold without a user- or
            # reference-defined noise/acceptance policy.
    return {
        "fit_accepted": not reasons,
        "fit_quality_status": "accepted_with_identifiability_diagnostics" if not reasons else "rejected_unidentifiable_or_invalid",
        "fit_quality_reasons": reasons,
        "covariance_correlation_condition_number": condition,
        "fit_residual_rms": float(np.sqrt(np.mean(residual ** 2))) if residual.size else None,
        "fit_reduced_chi_square": chi2,
        "fit_reduced_chi_square_policy": "diagnostic_only_no_hard_acceptance_cutoff_without_an_explicit_noise_and_significance_policy",
        "fit_degrees_of_freedom": int(dof),
    }


def _channel_covariance(value, channel, n_channels, name):
    if value is None:
        return None
    raw = np.asarray(value, dtype=float)
    if raw.shape == (6, 6):
        return raw
    if raw.shape == (n_channels, 6, 6):
        return raw[channel]
    raise ValueError(f"{name} must be a shared 6x6 covariance matrix or one 6x6 matrix per channel")


def _channel_value(value, channel, n_channels, name, *, vector_length=None):
    """Accept a scalar shared by channels or a channel-specific value."""
    if value is None:
        return None
    raw = np.asarray(value)
    if raw.ndim == 0:
        return value
    if vector_length is not None:
        if raw.shape == (vector_length,):
            return value
        if raw.shape == (n_channels, vector_length):
            return raw[channel]
    if raw.shape == (n_channels,):
        return raw[channel]
    raise ValueError(f"{name} must be a scalar shared across channels or have one value per channel")



def _fit_noncollective(wavelength, signal, laser_nm, angle, response, uncertainty):
    signal_scale = float(np.max(np.abs(signal)))
    if not np.isfinite(signal_scale) or signal_scale <= 0:
        raise ValueError("signal must have a positive finite scale")
    signal_scaled = signal/signal_scale
    uncertainty_scaled = None if uncertainty is None else uncertainty/signal_scale
    spacing = float(np.median(np.diff(wavelength)))
    span = float(wavelength[-1] - wavelength[0])
    edge = max(2, wavelength.size // 10)
    baseline = float(np.median(np.r_[signal_scaled[:edge], signal_scaled[-edge:]]))
    peak = int(np.argmax(signal_scaled))
    amp0 = max(float(signal_scaled[peak] - baseline), np.finfo(float).eps)
    p0 = [amp0, float(wavelength[peak]), max(spacing, span/10), baseline, 0.0]
    lower = [0.0, wavelength[0], max(spacing*0.2, 1e-12), -np.inf, -np.inf]
    upper = [np.inf, wavelength[-1], span, np.inf, np.inf]

    def observed(x, amp, center, sigma, offset, slope):
        intrinsic = _gaussian(x, amp, center, sigma, offset, slope)
        # The LSF acts on the signal component; a slowly varying background is
        # represented in detector space and is therefore not convolved.
        signal_only = amp*np.exp(-0.5*((x-center)/sigma)**2)
        return _apply_response(response, signal_only, x) + offset + slope*(x-center)

    p, cov = curve_fit(observed, wavelength, signal_scaled, p0=p0, bounds=(lower, upper),
                       sigma=uncertainty_scaled, absolute_sigma=uncertainty_scaled is not None,
                       maxfev=50000)
    parameter_scales=np.array([signal_scale,1.0,1.0,signal_scale,signal_scale])
    p=np.asarray(p,dtype=float)*parameter_scales
    cov=np.asarray(cov,dtype=float)*np.outer(parameter_scales,parameter_scales)
    amp, center, sigma, offset, slope = map(float, p)
    if sigma <= 0:
        raise ValueError("Fitted intrinsic spectral width is not positive")
    if center-3*sigma < wavelength[0] or center+3*sigma > wavelength[-1]:
        raise ValueError("Wavelength coverage must include at least ±3 fitted intrinsic widths")
    # Non-collective, non-relativistic, isotropic Maxwellian electron model.
    thermal_sigma = sigma
    te = thomson_electron_temperature_from_spectral_width(
        thermal_sigma, laser_nm, angle
    )["electron_temperature_eV"]
    # The canonical formula-library helper defines the Doppler sign convention.
    velocity = thomson_velocity_projection(center, laser_nm, angle)["electron_velocity_projection_m_s"]
    q_projection = 2.0*np.sin(angle/2.0)
    if te / ELECTRON_REST_ENERGY_EV >= NONRELATIVE_THETA_E_SCREEN:
        raise ValueError("Non-relativistic Maxwellian model rejected: fitted electron temperature is not safely small relative to the electron rest energy; the selected non-relativistic model is outside its declared validity range.")
    if abs(velocity) >= NONRELATIVE_FLOW_BETA_SCREEN * C_M_S:
        raise ValueError("Non-relativistic Maxwellian model rejected: fitted flow projection is outside the declared |v|/c < 0.1 regime.")
    dof = max(1, wavelength.size-len(p))
    residual = signal-observed(wavelength, *p)
    fit_cov = np.asarray(cov, dtype=float)
    te_unc = float(te*2*np.sqrt(max(fit_cov[2,2], 0.0))/sigma)
    # A normalized instrument-response operator redistributes power without
    # changing the full-domain integral of the fitted intrinsic Gaussian.
    area = float(amp * sigma * np.sqrt(2.0*np.pi))
    area_gradient = np.array([sigma*np.sqrt(2.0*np.pi), 0.0,
                              amp*np.sqrt(2.0*np.pi), 0.0, 0.0])
    area_unc = float(np.sqrt(max(float(area_gradient @ fit_cov @ area_gradient), 0.0)))
    return {
        "electron_temperature_eV": float(te),
        "electron_temperature_uncertainty_eV": te_unc,
        "electron_velocity_projection_m_s": float(velocity),
        "electron_velocity_projection_uncertainty_m_s": float(C_M_S*np.sqrt(max(fit_cov[1,1],0.0))/(laser_nm*q_projection)),
        "fitted_center_nm": center, "intrinsic_sigma_nm": sigma,
        "amplitude": amp, "background_intercept": offset, "background_slope_per_nm": slope,
        "fit_covariance": fit_cov, "residual": residual,
        "integrated_scattered_power_W": area, "integrated_scattered_power_uncertainty_W": area_unc,
        "reduced_chi_square": float(np.sum((residual/uncertainty)**2)/dof) if uncertainty is not None else None,
        "fitted_spectrum": observed(wavelength, *p),
        "fit_parameters": np.asarray(p, dtype=float),
        "fit_parameter_names": ["amplitude", "center_nm", "intrinsic_sigma_nm", "background_intercept", "background_slope_per_nm"],
        "model": "non-collective, non-relativistic Maxwellian electron spectrum",
        "validity_screening": {
            "theta_e": float(te / ELECTRON_REST_ENERGY_EV),
            "theta_e_limit": NONRELATIVE_THETA_E_SCREEN,
            "bulk_flow_beta": float(abs(velocity) / C_M_S),
            "bulk_flow_beta_limit": NONRELATIVE_FLOW_BETA_SCREEN,
            "rationale": "Conservative asymptotic error-budget screens: thermal relativistic corrections scale with theta_e and leading bulk-flow corrections scale with beta_flow squared; not universal validity boundaries.",
        },
        "assumptions": ["single-scattering non-collective regime", "isotropic Maxwellian electrons",
                        "non-relativistic electron velocities", "known scattering geometry"],
    }



def _fit_kappa(wavelength, signal, laser_nm, angle, response, uncertainty):
    """Fit the 1-D projection of an isotropic kappa VDF (Student-t form)."""
    signal_scale=float(np.max(np.abs(signal)))
    if not np.isfinite(signal_scale) or signal_scale<=0:
        raise ValueError("signal must have a positive finite scale")
    signal_scaled=signal/signal_scale
    uncertainty_scaled=None if uncertainty is None else uncertainty/signal_scale
    spacing = float(np.median(np.diff(wavelength)))
    span = float(wavelength[-1] - wavelength[0])
    edge = max(2, wavelength.size // 10)
    baseline = float(np.median(np.r_[signal_scaled[:edge], signal_scaled[-edge:]]))
    peak = int(np.argmax(signal_scaled))
    amp0 = max(float(signal_scaled[peak] - baseline), np.finfo(float).eps)
    p0 = [amp0, float(wavelength[peak]), max(spacing, span/10), 3.0, baseline, 0.0]
    lower = [0.0, wavelength[0], max(spacing*0.2, 1e-12), 1.51, -np.inf, -np.inf]
    upper = [np.inf, wavelength[-1], span, np.inf, np.inf, np.inf]
    def intrinsic(x, amp, center, sigma, kappa, offset, slope):
        df = 2.0*kappa-1.0
        scale = sigma*np.sqrt((df-2.0)/df)
        return amp*(1.0+((x-center)/scale)**2/df)**(-kappa)+offset+slope*(x-center)
    def observed(x, amp, center, sigma, kappa, offset, slope):
        component = intrinsic(x, amp, center, sigma, kappa, 0.0, 0.0)
        return _apply_response(response, component, x) + offset+slope*(x-center)
    p, cov = curve_fit(observed,wavelength,signal_scaled,p0=p0,bounds=(lower,upper),
                       sigma=uncertainty_scaled,absolute_sigma=uncertainty_scaled is not None,maxfev=80000)
    parameter_scales=np.array([signal_scale,1.0,1.0,1.0,signal_scale,signal_scale])
    p=np.asarray(p,dtype=float)*parameter_scales
    cov=np.asarray(cov,dtype=float)*np.outer(parameter_scales,parameter_scales)
    amp,center,sigma,kappa,offset,slope=map(float,p)
    # A kappa spectrum has power-law tails; Gaussian ±3-sigma coverage is not
    # a valid universal truncation criterion. Report the actual Student-t
    # probability captured by the measured wavelength window instead.
    kappa_df = 2.0 * kappa - 1.0
    kappa_scale = sigma * np.sqrt((kappa_df - 2.0) / kappa_df)
    captured_probability = float(
        stdtr(kappa_df, (wavelength[-1] - center) / kappa_scale)
        - stdtr(kappa_df, (wavelength[0] - center) / kappa_scale)
    )
    te=thomson_electron_temperature_from_spectral_width(
        sigma, laser_nm, angle
    )["electron_temperature_eV"]
    velocity=thomson_velocity_projection(
        center, laser_nm, angle
    )["electron_velocity_projection_m_s"]
    q_projection=2.0*np.sin(angle/2.0)
    if te / ELECTRON_REST_ENERGY_EV >= 0.01:
        raise ValueError("Non-relativistic kappa model rejected: fitted electron temperature is not safely small relative to the electron rest energy; the selected non-relativistic model is outside its declared validity range.")
    if abs(velocity) >= 0.1 * C_M_S:
        raise ValueError("Non-relativistic kappa model rejected: fitted flow projection is outside the declared |v|/c < 0.1 regime.")
    residual=signal-observed(wavelength,*p)
    dof=max(1,wavelength.size-len(p))
    def area_from_params(values):
        a, cen, sig, kap, off, sl = values
        df_local=2.0*kap-1.0
        scale_local=sig*np.sqrt((df_local-2.0)/df_local)
        return float(a*scale_local*np.sqrt(df_local*np.pi)
                     * np.exp(gammaln(kap-0.5)-gammaln(kap)))
    area=area_from_params(p)
    area_gradient=np.zeros(len(p))
    for j in range(len(p)):
        step=np.cbrt(np.finfo(float).eps)*max(abs(float(p[j])),1.0)
        plus=np.array(p,dtype=float); minus=np.array(p,dtype=float)
        plus[j]+=step; minus[j]-=step
        area_gradient[j]=(area_from_params(plus)-area_from_params(minus))/(2.0*step)
    area_unc=float(np.sqrt(max(float(area_gradient@cov@area_gradient),0.0)))
    return {"electron_temperature_eV":float(te),
            "electron_temperature_uncertainty_eV":float(te*2*np.sqrt(max(cov[2,2],0.0))/sigma),
            "electron_velocity_projection_m_s":float(velocity),
            "electron_velocity_projection_uncertainty_m_s":float(C_M_S*np.sqrt(max(cov[1,1],0.0))/(laser_nm*q_projection)),
            "fitted_center_nm":center,"intrinsic_sigma_nm":sigma,"kappa_index":kappa,
            "amplitude":amp,"background_intercept":offset,"background_slope_per_nm":slope,
            "fit_covariance":np.asarray(cov,dtype=float),"residual":residual,
            "fit_parameters":np.asarray(p,dtype=float),
            "fit_parameter_names":["amplitude","center_nm","intrinsic_sigma_nm","kappa_index","background_intercept","background_slope_per_nm"],
            "integrated_scattered_power_W":area,"integrated_scattered_power_uncertainty_W":area_unc,
            "captured_spectral_probability":captured_probability,
            "spectral_window_truncation_warning":("The measured wavelength window captures only a fraction of the fitted heavy-tailed kappa spectrum; integrated model area is extrapolated beyond the measured window." if captured_probability < 0.999 else None),
            "reduced_chi_square":float(np.sum((residual/uncertainty)**2)/dof) if uncertainty is not None else None,
            "fitted_spectrum":observed(wavelength,*p),
            "model":"non-collective isotropic kappa electron VDF (Student-t projected spectrum)",
            "validity_screening":{
                "theta_e":float(te/ELECTRON_REST_ENERGY_EV),
                "theta_e_limit":NONRELATIVE_THETA_E_SCREEN,
                "bulk_flow_beta":float(abs(velocity)/C_M_S),
                "bulk_flow_beta_limit":NONRELATIVE_FLOW_BETA_SCREEN,
                "rationale":"Conservative asymptotic error-budget screens: thermal relativistic corrections scale with theta_e and leading bulk-flow corrections scale with beta_flow squared; not universal validity boundaries."
            },
            "assumptions":["non-collective single-particle scattering","isotropic kappa electron velocity distribution",
                           "non-relativistic velocities","finite-variance kappa index > 1.5",
                           "known scattering geometry"]}


def _default_collective_bounds(initial_density_m3, initial_temperature_eV):
    """Return adaptive two-decade bounds when no experiment-specific bounds are supplied."""
    density = _finite_scalar(initial_density_m3, "initial density", minimum=np.finfo(float).tiny)
    temperature = _finite_scalar(initial_temperature_eV, "initial temperature", minimum=np.finfo(float).tiny)
    return (np.array([density / 100.0, density * 100.0]),
            np.array([temperature / 100.0, temperature * 100.0]))

def _collective_external_parameter_indices(channel_index, ion_species_count):
    """Return covariance-vector indices for one channel in channel-major order."""
    channel_index = int(channel_index)
    ion_species_count = int(ion_species_count)
    if channel_index < 0 or ion_species_count < 1:
        raise ValueError("channel_index must be non-negative and ion_species_count must be positive")
    base = channel_index * (3 + ion_species_count)
    return base, base + 1, base + 2, tuple(range(base + 3, base + 3 + ion_species_count))


def analyze_thomson_spectra(
    wavelength_nm, spectra, radius_m, *, laser_wavelength_nm, scattering_angle_rad,
    laser_wavelength_uncertainty_nm=None, scattering_angle_uncertainty_rad=None,
    incident_direction_uncertainty=None, scattered_direction_uncertainty=None,
    instrument_response=None, signal_uncertainty=None,
    laser_power_w=None, interaction_length_m=None, collection_solid_angle_sr=None,
    optical_efficiency=None, incident_polarization=None, incident_direction=None, scattered_direction=None,
    spectral_power_density_calibrated=False, laser_power_uncertainty_W=None,
    interaction_length_uncertainty_m=None, collection_solid_angle_uncertainty_sr=None,
    optical_efficiency_uncertainty=None, polarization_factor_uncertainty=None,
    density_relative_covariance=None, joint_density_relative_covariance=None,
    instrument_response_uncertainty=None, model=None,
    ion_species=None, ion_fractions=None, ion_temperature_eV=None, ion_velocity_m_s=None, electron_velocity_m_s=None,
    collective_initial_density_m3=None, collective_initial_temperature_eV=None,
    collective_density_bounds_m3=None, collective_temperature_bounds_eV=None,
    instrument_response_mc_samples=200, instrument_response_mc_min_success_fraction=0.8,
    joint_spectral_covariance=None,
    ion_temperature_uncertainty_eV=None, ion_velocity_uncertainty_m_s=None,
    electron_velocity_uncertainty_m_s=None, collective_external_parameter_mc_samples=None,
    collective_kappa_initial=None, collective_kappa_bounds=None,
    incident_polarization_uncertainty=None,
):
    """Infer per-channel Thomson parameters from calibrated multi-channel spectra.

    Spectra have shape (channel, wavelength); wavelength may be shared (1-D) or
    channel-specific (2-D). A measured response callable, kernel or matrix maps true
    spectral bins to detector bins; matrix columns are normalized internally. Absolute density is emitted only
    when all optical-power/geometry/polarization factors are supplied.
    """
    y = _finite_array(spectra, "spectra", 2)
    if y.shape[1] < 7:
        raise ValueError("Each Thomson spectrum must contain at least seven wavelength samples")
    radii = _finite_array(radius_m, "radius_m", 1)
    if y.shape[0] != radii.size:
        raise ValueError("radius_m length must equal the number of spectral channels")
    x = _finite_array(wavelength_nm, "wavelength_nm")
    if x.ndim == 1:
        if x.size != y.shape[1]:
            raise ValueError("wavelength_nm length must equal the spectral width")
        xs = np.broadcast_to(x, y.shape)
    elif x.ndim == 2 and x.shape == y.shape:
        xs = x
    else:
        raise ValueError("wavelength_nm must be shared 1-D or channel-shaped 2-D")
    if np.any(np.diff(xs, axis=1) <= 0):
        raise ValueError("Each channel wavelength axis must be strictly increasing")
    if np.any(xs <= 0):
        raise ValueError("Wavelength samples must be positive")
    if instrument_response is None:
        raise ValueError("A measured instrument-response matrix is required; supply an identity matrix only when a delta-like response is explicitly justified.")
    laser_wavelength_nm = _finite_scalar(laser_wavelength_nm, "laser_wavelength_nm", minimum=np.finfo(float).tiny)
    scattering_angle_rad = _finite_scalar(scattering_angle_rad, "scattering_angle_rad", minimum=np.finfo(float).tiny, maximum=np.pi)
    for uncertainty_value, uncertainty_name in (
        (laser_wavelength_uncertainty_nm,"laser_wavelength_uncertainty_nm"),
        (scattering_angle_uncertainty_rad,"scattering_angle_uncertainty_rad"),
    ):
        if uncertainty_value is not None:
            _finite_scalar(uncertainty_value, uncertainty_name, minimum=0.0)
    laser_wavelength_unc = 0.0 if laser_wavelength_uncertainty_nm is None else float(laser_wavelength_uncertainty_nm)
    scattering_angle_unc = 0.0 if scattering_angle_uncertainty_rad is None else float(scattering_angle_uncertainty_rad)
    incident_direction_unc = None if incident_direction_uncertainty is None else _finite_array(
        incident_direction_uncertainty, "incident_direction_uncertainty", 1)
    scattered_direction_unc = None if scattered_direction_uncertainty is None else _finite_array(
        scattered_direction_uncertainty, "scattered_direction_uncertainty", 1)
    for name, value in (("incident_direction_uncertainty", incident_direction_unc),
                        ("scattered_direction_uncertainty", scattered_direction_unc)):
        if value is not None and (value.shape != (3,) or np.any(value < 0)):
            raise ValueError(f"{name} must be a non-negative 3-vector of component-wise 1-sigma uncertainties")
    if (incident_direction_unc is not None or scattered_direction_unc is not None) and scattering_angle_uncertainty_rad is not None:
        raise ValueError("Use scattering_angle_uncertainty_rad or beam-direction component uncertainties, not both; they describe the same geometry uncertainty.")
    if model is None:
        raise ValueError("Select a Thomson physical model explicitly; no default Maxwellian assumption is applied.")
    supported_models = {
        "non_collective_maxwellian",
        "non_collective_kappa",
        "non_collective_relativistic_maxwell_juttner",
    }
    if model not in supported_models:
        raise ValueError(
            "Unsupported optical Thomson model. Analysis 1 supports non_collective_maxwellian, "
            "non_collective_kappa and non_collective_relativistic_maxwell_juttner. "
            "Optical collective dielectric models are outside this shortlist; microwave CTS "
            "has its own registered workflow (analysis 30)."
        )
    if (incident_direction_unc is not None or scattered_direction_unc is not None) and model != "non_collective_relativistic_maxwell_juttner":
        raise ValueError(
            "Beam-direction component uncertainty is supported only by the relativistic "
            "Maxwell-Juttner optical Thomson model."
        )
    polarization_unc = None
    if incident_polarization_uncertainty is not None:
        if model != "non_collective_relativistic_maxwell_juttner":
            raise ValueError("incident_polarization_uncertainty is supported only by the relativistic Maxwell-Juttner model")
        polarization_unc = _finite_array(
            incident_polarization_uncertainty, "incident_polarization_uncertainty", 1
        )
        if polarization_unc.shape != (3,) or np.any(polarization_unc < 0):
            raise ValueError("incident_polarization_uncertainty must be a non-negative 3-vector of component-wise 1-sigma uncertainties")
    collective_only_inputs = (
        ion_species, ion_fractions, ion_temperature_eV, ion_velocity_m_s,
        electron_velocity_m_s, ion_temperature_uncertainty_eV,
        ion_velocity_uncertainty_m_s, electron_velocity_uncertainty_m_s,
        collective_initial_density_m3, collective_initial_temperature_eV,
        collective_density_bounds_m3, collective_temperature_bounds_eV,
        collective_external_parameter_mc_samples, collective_kappa_initial,
        collective_kappa_bounds,
    )
    if any(value is not None for value in collective_only_inputs):
        raise ValueError(
            "Collective-only inputs are not accepted by optical Thomson Analysis 1; "
            "use the separate registered microwave CTS workflow for collective Thomson data."
        )
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")
    if isinstance(instrument_response_mc_samples, (bool, np.bool_)) or not isinstance(instrument_response_mc_samples, (int, np.integer)) or instrument_response_mc_samples < 20:
        raise ValueError("instrument_response_mc_samples must be an integer >= 20")
    instrument_response_mc_min_success_fraction = _finite_scalar(instrument_response_mc_min_success_fraction, "instrument_response_mc_min_success_fraction", minimum=0.5, maximum=1.0)
    collective_models = set()
    density_bounds = temperature_bounds = None
    if model in collective_models:
        if collective_initial_density_m3 is None or collective_initial_temperature_eV is None:
            raise ValueError("Collective Thomson models require explicit initial electron density and temperature estimates.")
        collective_initial_density_m3 = _finite_scalar(
            collective_initial_density_m3, "collective_initial_density_m3",
            minimum=np.finfo(float).tiny,
        )
        collective_initial_temperature_eV = _finite_scalar(
            collective_initial_temperature_eV, "collective_initial_temperature_eV",
            minimum=np.finfo(float).tiny,
        )
        if collective_density_bounds_m3 is not None:
            density_bounds = _finite_array(collective_density_bounds_m3, "collective_density_bounds_m3", 1)
            if density_bounds.shape != (2,) or np.any(density_bounds <= 0) or density_bounds[0] >= density_bounds[1]:
                raise ValueError("collective_density_bounds_m3 must be a positive increasing pair")
            if not density_bounds[0] <= collective_initial_density_m3 <= density_bounds[1]:
                raise ValueError("collective_initial_density_m3 must lie within collective_density_bounds_m3")
        if collective_temperature_bounds_eV is not None:
            temperature_bounds = _finite_array(collective_temperature_bounds_eV, "collective_temperature_bounds_eV", 1)
            if temperature_bounds.shape != (2,) or np.any(temperature_bounds <= 0) or temperature_bounds[0] >= temperature_bounds[1]:
                raise ValueError("collective_temperature_bounds_eV must be a positive increasing pair")
            if not temperature_bounds[0] <= collective_initial_temperature_eV <= temperature_bounds[1]:
                raise ValueError("collective_initial_temperature_eV must lie within collective_temperature_bounds_eV")
        if model == "collective_maxwellian" and (density_bounds is None or temperature_bounds is None):
            raise ValueError("Collective Maxwellian inference requires explicit electron-density and electron-temperature bounds; no scientific range is assumed.")
    if model == "collective_kappa":
        if collective_kappa_initial is None:
            raise ValueError("Collective kappa inference requires an explicit initial kappa index >1.5.")
        collective_kappa_initial = _finite_scalar(
            collective_kappa_initial, "collective_kappa_initial", minimum=1.5000001,
        )
        if collective_kappa_bounds is not None:
            collective_kappa_bounds = _finite_array(collective_kappa_bounds, "collective_kappa_bounds", 1)
            if (collective_kappa_bounds.shape != (2,) or collective_kappa_bounds[0] <= 1.5
                    or collective_kappa_bounds[0] >= collective_kappa_bounds[1]):
                raise ValueError("collective_kappa_bounds must be an increasing pair with lower bound >1.5")
            if not collective_kappa_bounds[0] <= collective_kappa_initial <= collective_kappa_bounds[1]:
                raise ValueError("collective_kappa_initial must lie within collective_kappa_bounds")
    probe_vec = scatter_vec = None
    if model in collective_models or model == "non_collective_relativistic_maxwell_juttner":
        if incident_direction is None or scattered_direction is None:
            raise ValueError("The selected Thomson model requires explicit incident_direction and scattered_direction 3-vectors.")
        if model == "non_collective_relativistic_maxwell_juttner" and incident_polarization is None:
            raise ValueError("The relativistic Maxwell-Juttner Thomson model requires an explicit incident_polarization vector.")
        probe_vec = _finite_array(incident_direction, "incident_direction", 1)
        scatter_vec = _finite_array(scattered_direction, "scattered_direction", 1)
        if probe_vec.shape != (3,) or scatter_vec.shape != (3,):
            raise ValueError("incident_direction and scattered_direction must each be 3-vectors")
        probe_norm = float(np.linalg.norm(probe_vec))
        scatter_norm = float(np.linalg.norm(scatter_vec))
        if probe_norm <= 0 or scatter_norm <= 0:
            raise ValueError("incident_direction and scattered_direction must be non-zero")
        probe_vec = probe_vec/probe_norm
        scatter_vec = scatter_vec/scatter_norm
        geometry_angle = float(np.arccos(np.clip(np.dot(probe_vec, scatter_vec), -1.0, 1.0)))
        if not np.isclose(geometry_angle, float(scattering_angle_rad), rtol=0.0, atol=1e-5):
            raise ValueError("scattering_angle_rad is inconsistent with incident_direction and scattered_direction")

    if model in collective_models:
        if model in collective_models and (
            isinstance(collective_external_parameter_mc_samples,bool)
            or not isinstance(collective_external_parameter_mc_samples,(int,np.integer))
            or collective_external_parameter_mc_samples<20
        ):
            raise ValueError("collective_external_parameter_mc_samples must be an integer >= 20")
        if not ion_species or isinstance(ion_species, (str, bytes)):
            raise ValueError("Collective Thomson inference requires an explicit non-empty ion species list")
        if ion_velocity_m_s is None or electron_velocity_m_s is None:
            raise ValueError("Collective inference requires explicit ion/electron flow vectors; use zero vectors only to declare stationary flow.")
        if ion_temperature_eV is None:
            raise ValueError("Collective inference requires independently supplied ion temperature(s); no ion temperature is assumed")
        ion_t_check = _finite_array(np.atleast_1d(ion_temperature_eV), "ion_temperature_eV", 1)
        if ion_t_check.size != len(ion_species) or np.any(ion_t_check <= 0):
            raise ValueError("ion_temperature_eV must contain one positive temperature per ion species")
        if ion_fractions is None:
            if len(ion_species) > 1:
                raise ValueError("Collective multi-species inference requires explicit ion_fractions summing to one")
            ion_fractions = [1.0]
        ion_fract_check = _finite_array(ion_fractions, "ion_fractions", 1)
        if ion_fract_check.size != len(ion_species) or np.any(ion_fract_check <= 0) or not np.isclose(np.sum(ion_fract_check),1.0,rtol=1e-6,atol=1e-8):
            raise ValueError("ion_fractions must contain one positive fraction per species and sum to one")
        ion_v_check = _finite_array(ion_velocity_m_s, "ion_velocity_m_s")
        electron_v_check = _finite_array(electron_velocity_m_s, "electron_velocity_m_s")
        if ion_v_check.shape == (3,) and len(ion_species) == 1:
            ion_v_check = ion_v_check.reshape(1,3)
        if electron_v_check.shape == (3,):
            electron_v_check = electron_v_check.reshape(1,3)
        if electron_v_check.shape != (1, 3) or ion_v_check.shape != (len(ion_species), 3):
            raise ValueError("electron_velocity_m_s must be a 3-vector and ion_velocity_m_s one 3-vector per species")
        if np.any(np.linalg.norm(ion_v_check,axis=1) >= C_M_S) or np.linalg.norm(electron_v_check[0]) >= C_M_S:
            raise ValueError("Collective ion/electron flow speeds must be subluminal")
        ion_t_unc = None
        if ion_temperature_uncertainty_eV is not None:
            ion_t_unc = _finite_array(np.atleast_1d(ion_temperature_uncertainty_eV),"ion_temperature_uncertainty_eV",1)
            if ion_t_unc.shape != ion_t_check.shape or np.any(ion_t_unc < 0):
                raise ValueError("ion_temperature_uncertainty_eV must be non-negative and match ion species")
        ion_v_unc = None
        if ion_velocity_uncertainty_m_s is not None:
            ion_v_unc = _finite_array(ion_velocity_uncertainty_m_s,"ion_velocity_uncertainty_m_s")
            if ion_v_unc.shape == (3,) and len(ion_species) == 1:
                ion_v_unc = ion_v_unc.reshape(1,3)
            if ion_v_unc.shape != ion_v_check.shape or np.any(ion_v_unc < 0):
                raise ValueError("ion_velocity_uncertainty_m_s must be non-negative and match ion velocity vectors")
        electron_v_unc = None
        if electron_velocity_uncertainty_m_s is not None:
            electron_v_unc = _finite_array(electron_velocity_uncertainty_m_s,"electron_velocity_uncertainty_m_s",1)
            if electron_v_unc.shape != (3,) or np.any(electron_v_unc < 0):
                raise ValueError("electron_velocity_uncertainty_m_s must be a non-negative 3-vector")
        ion_t = ion_t_check
        ion_fract = ion_fract_check
        ion_v = ion_v_check
        electron_v = electron_v_check
        if model == "collective_maxwellian":
            try:
                from astropy import units as u
                from plasmapy.diagnostics import thomson
            except ImportError as exc:
                raise ImportError("Collective Maxwellian Thomson inference requires the optional PlasmaPy and Astropy packages") from exc
        else:
            thomson = None
    else:
        thomson = None
        ion_t = ion_fract = ion_v = electron_v = None
        ion_t_unc = ion_v_unc = electron_v_unc = None


    unc = None if signal_uncertainty is None else _finite_array(signal_uncertainty, "signal_uncertainty", 2)
    if unc is not None and (unc.shape != y.shape or np.any(unc <= 0)):
        raise ValueError("signal_uncertainty must match spectra and be strictly positive")
    lsf_unc = None if instrument_response_uncertainty is None else _finite_array(instrument_response_uncertainty, "instrument_response_uncertainty")
    if lsf_unc is not None and (np.any(lsf_unc < 0) or instrument_response is None or lsf_unc.ndim not in (2,3) or lsf_unc.shape[-2:] != (y.shape[1],y.shape[1]) or (lsf_unc.ndim == 3 and lsf_unc.shape[0] != y.shape[0])):
        raise ValueError("instrument_response_uncertainty must be a non-negative 2-D/3-D matrix uncertainty on the measured wavelength grid")

    def require_zero_scattered_power_covariance(covariance_value, name):
        matrix = _finite_array(covariance_value, name)
        if matrix.shape == (6, 6):
            power_rows = [0]
            row_check = matrix[0, :]
            column_check = matrix[:, 0]
        elif matrix.shape == (y.shape[0], 6, 6):
            power_rows = None
            row_check = matrix[:, 0, :]
            column_check = matrix[:, :, 0]
        elif matrix.shape == (6 * y.shape[0], 6 * y.shape[0]):
            power_rows = np.arange(0, 6 * y.shape[0], 6, dtype=int)
            row_check = matrix[power_rows, :]
            column_check = matrix[:, power_rows]
        else:
            raise ValueError(
                f"{name} must be 6x6, n_channels x 6 x 6, or (6*n_channels) x (6*n_channels)."
            )
        if not np.allclose(row_check, 0.0, rtol=0.0, atol=1e-14) or not np.allclose(
            column_check, 0.0, rtol=0.0, atol=1e-14
        ):
            raise ValueError(
                f"{name} must have zero scattered-power rows and columns when joint spectral "
                "covariance or measured-response uncertainty is supplied; those independent "
                "power uncertainties are propagated from the spectrum/response Monte Carlo."
            )
        return matrix

    if density_relative_covariance is not None and (
        joint_spectral_covariance is not None or lsf_unc is not None
    ):
        require_zero_scattered_power_covariance(
            density_relative_covariance, "density_relative_covariance"
        )
    if joint_density_relative_covariance is not None:
        if model in {"collective_maxwellian", "collective_kappa"}:
            raise ValueError("joint_density_relative_covariance is for non-collective absolute-density inversion; collective models infer density from the dielectric spectral shape.")
        if density_relative_covariance is not None:
            raise ValueError("Supply density_relative_covariance or joint_density_relative_covariance, not both.")
        if not spectral_power_density_calibrated or any(value is None for value in (
            laser_power_w, interaction_length_m, collection_solid_angle_sr, optical_efficiency
        )):
            raise ValueError("joint_density_relative_covariance requires calibrated W/nm spectra and complete absolute-density optical geometry for every channel.")
        scalar_density_uncertainties = (
            laser_power_uncertainty_W, interaction_length_uncertainty_m,
            collection_solid_angle_uncertainty_sr, optical_efficiency_uncertainty,
            polarization_factor_uncertainty,
        )
        if any(value is not None for value in scalar_density_uncertainties):
            raise ValueError("joint_density_relative_covariance is the cross-channel optical-factor uncertainty model; do not also supply scalar optical-factor uncertainties.")
        joint_density_covariance = _finite_array(
            joint_density_relative_covariance, "joint_density_relative_covariance", 2
        )
        expected_density_covariance_shape = (6 * y.shape[0], 6 * y.shape[0])
        if joint_density_covariance.shape != expected_density_covariance_shape:
            raise ValueError("joint_density_relative_covariance must have shape (6*n_channels, 6*n_channels), ordered as six density-inversion inputs per channel.")
        if not np.allclose(joint_density_covariance, joint_density_covariance.T, rtol=1e-9, atol=0.0):
            raise ValueError("joint_density_relative_covariance must be symmetric")
        if not _covariance_is_positive_semidefinite(joint_density_covariance):
            raise ValueError("joint_density_relative_covariance must be positive semidefinite")
        if joint_spectral_covariance is not None or lsf_unc is not None:
            require_zero_scattered_power_covariance(
                joint_density_covariance, "joint_density_relative_covariance"
            )
    else:
        joint_density_covariance = None

    if joint_spectral_covariance is not None and signal_uncertainty is not None:
        raise ValueError(
            "Supply either signal_uncertainty (independent pointwise 1-sigma errors) "
            "or joint_spectral_covariance (the complete channel-major covariance), not both. "
            "The full covariance is authoritative and must not be silently combined with pointwise errors."
        )
    joint_covariance = None
    if joint_spectral_covariance is not None:
        if model not in {"non_collective_maxwellian","non_collective_kappa"}:
            raise ValueError("joint_spectral_covariance currently supports non_collective_maxwellian and non_collective_kappa; other physical models require model-specific joint forward models")
        joint_covariance = _finite_array(joint_spectral_covariance, "joint_spectral_covariance", 2)
        expected_joint_shape = (y.size, y.size)
        if joint_covariance.shape != expected_joint_shape or not np.allclose(joint_covariance, joint_covariance.T, rtol=1e-9, atol=0.0):
            raise ValueError("joint_spectral_covariance must be a symmetric (n_channels*n_wavelengths) square matrix in channel-major order")
        try:
            joint_cholesky = np.linalg.cholesky(joint_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("joint_spectral_covariance must be positive definite") from exc
    results = []
    channel_responses = []
    for i in range(y.shape[0]):
        xi = xs[i]
        response = _response_for_channel(instrument_response, i, y.shape[0], y.shape[1], xi)
        channel_responses.append(response)
        ui = None if unc is None else unc[i]
        if model == "non_collective_maxwellian":
            fit = _fit_noncollective(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), response, ui)
        elif model == "non_collective_kappa":
            fit = _fit_kappa(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), response, ui)
        elif model == "non_collective_relativistic_maxwell_juttner":
            fit = fit_relativistic_maxwell_juttner(
                xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                response, ui, incident_direction=probe_vec, scattered_direction=scatter_vec,
                incident_polarization=incident_polarization,
            )
        elif model == "collective_kappa":
            fit = fit_collective_kappa(
                xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                response, ui, incident_direction=probe_vec, scattered_direction=scatter_vec,
                ion_species=ion_species, ion_fractions=ion_fract,
                ion_temperature_eV=ion_t, electron_velocity_m_s=electron_v[0],
                ion_velocity_m_s=ion_v,
                initial_density_m3=collective_initial_density_m3,
                initial_temperature_eV=collective_initial_temperature_eV,
                density_bounds_m3=density_bounds, temperature_bounds_eV=temperature_bounds,
                initial_kappa=collective_kappa_initial, kappa_bounds=collective_kappa_bounds,
            )
        elif model == "collective_maxwellian":
            # PlasmaPy's kinetic Maxwellian spectral-density model evaluates
            # collective and non-collective regimes from the same dielectric
            # response; ion species/temperature are never silently defaulted.
            ion_t = _finite_array(np.atleast_1d(ion_temperature_eV), "ion_temperature_eV", 1)
            if ion_t.size != len(ion_species) or np.any(ion_t <= 0):
                raise ValueError("ion_temperature_eV must contain one positive temperature per ion species")
            ion_fract = _finite_array(ion_fractions, "ion_fractions", 1)
            if ion_fract.size != len(ion_species) or np.any(ion_fract <= 0) or not np.isclose(np.sum(ion_fract),1.0,rtol=1e-6,atol=1e-8):
                raise ValueError("ion_fractions must contain one positive fraction per species and sum to one")
            ion_v = _finite_array(ion_velocity_m_s, "ion_velocity_m_s")
            electron_v = _finite_array(electron_velocity_m_s, "electron_velocity_m_s")
            if ion_v.shape == (3,) and len(ion_species) == 1:
                ion_v = ion_v.reshape(1,3)
            if electron_v.shape == (3,):
                electron_v = electron_v.reshape(1,3)
            if electron_v.shape != (1, 3):
                raise ValueError("electron_velocity_m_s must be a 3-vector")
            if ion_v.shape != (len(ion_species), 3):
                raise ValueError("ion_velocity_m_s must have shape (number_of_ion_species, 3)")
            if np.any(np.linalg.norm(ion_v,axis=1) >= C_M_S) or np.linalg.norm(electron_v[0]) >= C_M_S:
                raise ValueError("Collective ion/electron flow speeds must be subluminal")
            if ion_temperature_uncertainty_eV is not None:
                ion_t_unc=_finite_array(np.atleast_1d(ion_temperature_uncertainty_eV),"ion_temperature_uncertainty_eV",1)
                if ion_t_unc.shape!=ion_t.shape or np.any(ion_t_unc<0):
                    raise ValueError("ion_temperature_uncertainty_eV must be non-negative and match ion species")
            else: ion_t_unc=None
            if ion_velocity_uncertainty_m_s is not None:
                ion_v_unc=_finite_array(ion_velocity_uncertainty_m_s,"ion_velocity_uncertainty_m_s")
                if ion_v_unc.shape==(3,) and len(ion_species)==1: ion_v_unc=ion_v_unc.reshape(1,3)
                if ion_v_unc.shape!=ion_v.shape or np.any(ion_v_unc<0):
                    raise ValueError("ion_velocity_uncertainty_m_s must be non-negative and match ion velocity vectors")
            else: ion_v_unc=None
            if electron_velocity_uncertainty_m_s is not None:
                electron_v_unc=_finite_array(electron_velocity_uncertainty_m_s,"electron_velocity_uncertainty_m_s",1)
                if electron_v_unc.shape!=(3,) or np.any(electron_v_unc<0):
                    raise ValueError("electron_velocity_uncertainty_m_s must be a non-negative 3-vector")
            else: electron_v_unc=None
            # Fit ne and Te against the full PlasmaPy spectrum. Absolute signal
            # scaling is a nuisance amplitude; density is identifiable from
            # the collective spectral shape when the spectrum resolves it.
            from scipy.optimize import least_squares
            def collective_shape(params):
                ne, te = params
                _, spec = thomson.spectral_density(
                    xi*u.nm, float(laser_wavelength_nm)*u.nm, ne/u.m**3,
                    T_e=te*u.eV, T_i=ion_t*u.eV, ions=ion_species, ifract=ion_fract,
                    ion_vel=ion_v*u.m/u.s, electron_vel=electron_v*u.m/u.s,
                    probe_vec=probe_vec, scatter_vec=scatter_vec)
                z = np.asarray(spec.value, dtype=float)
                return _apply_response(response, z, xi)
            residual_scale = ui if ui is not None else np.full_like(y[i],max(float(np.max(np.abs(y[i]))),np.finfo(float).tiny))
            def fit_nuisance(shape):
                shape_scale=max(float(np.max(np.abs(shape))),np.finfo(float).tiny)
                design=np.column_stack((shape/shape_scale,np.ones_like(shape),xi-float(laser_wavelength_nm)))
                weighted_design=design/residual_scale[:,None]
                weighted_signal=y[i]/residual_scale
                coefficients=np.linalg.lstsq(weighted_design,weighted_signal,rcond=None)[0]
                if coefficients[0] < 0:
                    background_design=design[:,1:]/residual_scale[:,None]
                    background=np.linalg.lstsq(background_design,weighted_signal,rcond=None)[0]
                    coefficients=np.array([0.0,background[0],background[1]])
                coefficients[0] /= shape_scale
                return coefficients
            def objective(params):
                shape = collective_shape(params)
                coefficients=fit_nuisance(shape)
                model_signal=coefficients[0]*shape+coefficients[1]+coefficients[2]*(xi-float(laser_wavelength_nm))
                return (model_signal-y[i])/residual_scale
            fitopt = least_squares(
                lambda log_params: objective(10.0**log_params),
                np.log10([collective_initial_density_m3, collective_initial_temperature_eV]),
                bounds=(np.log10([density_bounds[0], temperature_bounds[0]]),
                        np.log10([density_bounds[1], temperature_bounds[1]])),
                x_scale="jac", max_nfev=5000,
            )
            ne_fit, te_fit = map(float, 10.0**fitopt.x)
            # Two nonlinear plasma parameters plus three profiled linear nuisance parameters
            # (amplitude, background intercept, and background slope) are estimated.
            dof = max(1, y[i].size-5)
            jacobian_rank = int(np.linalg.matrix_rank(fitopt.jac))
            jacobian_condition = float(np.linalg.cond(fitopt.jac))
            try:
                covariance_log = np.linalg.pinv(fitopt.jac.T @ fitopt.jac)
                if ui is None:
                    covariance_log *= (2.0*fitopt.cost/dof)
                transform = np.diag(np.log(10.0)*np.array([ne_fit, te_fit]))
                parameter_covariance = transform @ covariance_log @ transform.T
                parameter_uncertainties = np.sqrt(np.maximum(np.diag(parameter_covariance),0.0))
            except np.linalg.LinAlgError:
                parameter_covariance = None
                parameter_uncertainties = np.array([np.nan,np.nan])
            # fitopt.x is in log10 space; the forward model requires physical ne and Te.
            model_spec = collective_shape(10.0**fitopt.x)
            alpha_fit, _ = thomson.spectral_density(
                xi*u.nm, float(laser_wavelength_nm)*u.nm, ne_fit/u.m**3,
                T_e=te_fit*u.eV, T_i=ion_t*u.eV, ions=ion_species, ifract=ion_fract,
                ion_vel=ion_v*u.m/u.s, electron_vel=electron_v*u.m/u.s,
                probe_vec=probe_vec, scatter_vec=scatter_vec)
            alpha_value=float(np.asarray(getattr(alpha_fit,"value",alpha_fit)))
            # PlasmaPy's scattering parameter follows alpha = 1/(|k_s-k_i| lambda_D),
            # hence alpha scales as sqrt(n_e/T_e) * lambda_L / sin(theta/2).
            # Propagate the complete fitted (n_e,T_e) covariance, including its
            # off-diagonal term, plus independently supplied laser/angle errors.
            alpha_gradient_ne_te = np.array([
                alpha_value / (2.0 * ne_fit),
                -alpha_value / (2.0 * te_fit),
            ], dtype=float)
            alpha_variance = None
            if parameter_covariance is not None:
                alpha_variance = float(
                    alpha_gradient_ne_te @ parameter_covariance @ alpha_gradient_ne_te
                )
                alpha_variance += (alpha_value * laser_wavelength_unc / float(laser_wavelength_nm)) ** 2
                if scattering_angle_uncertainty_rad is not None:
                    alpha_angle_sensitivity = -0.5 * alpha_value / np.tan(
                        float(scattering_angle_rad) / 2.0
                    )
                    alpha_variance += (
                        alpha_angle_sensitivity * scattering_angle_unc
                    ) ** 2
                alpha_variance = max(alpha_variance, 0.0)
            alpha_uncertainty = (
                float(np.sqrt(alpha_variance)) if alpha_variance is not None else None
            )
            amp_fit, background_fit, background_slope_fit = map(float,fit_nuisance(model_spec))
            fitted_collective = amp_fit*model_spec+background_fit+background_slope_fit*(xi-float(laser_wavelength_nm))
            residual_collective = y[i]-fitted_collective
            q_vector = scatter_vec-probe_vec
            q_hat = q_vector/np.linalg.norm(q_vector)
            electron_flow_projection = float(np.dot(electron_v[0],q_hat))
            ion_flow_projections = np.asarray(ion_v @ q_hat,dtype=float)
            fit = {"electron_density_m3": ne_fit, "electron_temperature_eV": te_fit,
                   "electron_velocity_projection_m_s": electron_flow_projection,
                   "ion_velocity_projection_m_s": ion_flow_projections.tolist(),
                   "electron_flow_source": "explicitly supplied electron velocity vector",
                   "fitted_spectrum": fitted_collective, "residual": residual_collective,
                   "amplitude": amp_fit, "background_intercept": background_fit,
                   "background_slope_per_nm": background_slope_fit,
                   "model": "PlasmaPy Maxwellian kinetic dielectric spectral-density model",
                   "electron_density_uncertainty_m3": float(parameter_uncertainties[0]),
                   "electron_temperature_uncertainty_eV": float(parameter_uncertainties[1]),
                   "fit_covariance": parameter_covariance,
                   "optimizer_success": bool(fitopt.success), "optimizer_message": str(fitopt.message),
                   "collective_parameter_jacobian_rank": jacobian_rank,
                   "collective_parameter_jacobian_condition_number": jacobian_condition,
                   "reduced_chi_square": float(np.sum((residual_collective/ui)**2)/dof) if ui is not None else None,
                   "ion_species": list(ion_species), "ion_fractions": ion_fract.tolist(),
                   "scattering_parameter_alpha": alpha_value,
                   "scattering_parameter_alpha_uncertainty": alpha_uncertainty,
                   "scattering_parameter_alpha_gradient_ne_te": alpha_gradient_ne_te,
                   "scattering_parameter_alpha_uncertainty_model": (
                       "first-order propagation of the full fitted ne/Te covariance plus supplied independent laser-wavelength and scalar scattering-angle uncertainties"
                       if alpha_uncertainty is not None else "not_available_without_finite_fit_covariance"
                   ),
                   "scattering_regime": "collective" if alpha_value > 1.0 else "non_collective_or_weakly_collective",
                   "ion_temperature_eV": np.atleast_1d(ion_t).tolist(),
                   "ion_velocity_m_s": ion_v.tolist(), "electron_velocity_m_s": electron_v.tolist()}
            if not fitopt.success:
                fit["electron_density_m3"] = None
                fit["electron_temperature_eV"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["electron_temperature_uncertainty_eV"] = None
                fit["density_status"] = "collective_spectral_optimizer_failed"
            elif jacobian_rank < 2:
                fit["electron_density_m3"] = None
                fit["electron_temperature_eV"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["electron_temperature_uncertainty_eV"] = None
                fit["density_status"] = "collective_density_temperature_not_jointly_identifiable"
                fit["temperature_status"] = "collective_density_temperature_not_jointly_identifiable"
            elif alpha_value <= 1.0:
                fit["electron_density_m3"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["density_status"] = "spectral_shape_does_not_resolve_collective_density"
        expected_parameter_count = (
            2 if model == "collective_maxwellian" else
            3 if model == "collective_kappa" else
            6 if model == "non_collective_kappa" else 5
        )
        quality = _fit_identifiability(
            fit.get("fit_covariance"), fit.get("residual"), ui, expected_parameter_count,
            optimizer_success=fit.get("optimizer_success", True),
            jacobian_rank=(fit.get("collective_parameter_jacobian_rank")
                           if model in {"collective_maxwellian", "collective_kappa"} else None),
            jacobian_condition_number=(fit.get("collective_parameter_jacobian_condition_number")
                                       if model in {"collective_maxwellian", "collective_kappa"} else None),
        )
        fit.update(quality)
        fit_covariance_for_discrepancy=fit.get("fit_covariance")
        if fit.get("fit_reduced_chi_square") is not None and fit_covariance_for_discrepancy is not None:
            discrepancy_scale=max(float(fit["fit_reduced_chi_square"])-1.0,0.0)
            discrepancy_covariance=discrepancy_scale*np.asarray(fit_covariance_for_discrepancy,dtype=float)
            fit["model_discrepancy_scale"]=discrepancy_scale
            fit["model_discrepancy_covariance"]=discrepancy_covariance
            fit["total_fit_covariance_including_discrepancy"]=np.asarray(fit_covariance_for_discrepancy,dtype=float)+discrepancy_covariance
            fit["model_discrepancy_diagnostics"]={
                "residual_rms":fit.get("fit_residual_rms"),
                "reduced_chi_square":fit.get("fit_reduced_chi_square"),
                "interpretation":"Excess reduced chi-square above unity is reported as a separate model-discrepancy covariance component conditional on the supplied noise model; it is not experimental validation.",
            }
            total_sigma=np.sqrt(np.maximum(np.diag(fit["total_fit_covariance_including_discrepancy"]),0.0))
            if model=="collective_maxwellian" and total_sigma.size>=2:
                if fit.get("electron_density_m3") is not None:
                    fit["electron_density_uncertainty_m3"]=float(total_sigma[0])
                fit["electron_temperature_uncertainty_eV"]=float(total_sigma[1])
            elif model=="collective_kappa" and total_sigma.size>=3:
                fit["electron_density_uncertainty_m3"]=float(total_sigma[0])
                fit["electron_temperature_uncertainty_eV"]=float(total_sigma[1])
                fit["kappa_index_uncertainty"]=float(total_sigma[2])
            elif model=="non_collective_kappa":
                fit["electron_temperature_uncertainty_eV"]=float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"],
                    2.0*fit["electron_temperature_eV"]*total_sigma[2]/max(float(fit["intrinsic_sigma_nm"]),np.finfo(float).tiny)))
                qproj=2.0*np.sin(float(scattering_angle_rad)/2.0)
                velocity_discrepancy=C_M_S*total_sigma[1]/(float(laser_wavelength_nm)*qproj)
                fit["electron_velocity_projection_uncertainty_m_s"]=float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"],velocity_discrepancy))
            elif model=="non_collective_maxwellian":
                fit["electron_temperature_uncertainty_eV"]=float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"],
                    2.0*fit["electron_temperature_eV"]*total_sigma[2]/max(float(fit["intrinsic_sigma_nm"]),np.finfo(float).tiny)))
                qproj=2.0*np.sin(float(scattering_angle_rad)/2.0)
                velocity_discrepancy=C_M_S*total_sigma[1]/(float(laser_wavelength_nm)*qproj)
                fit["electron_velocity_projection_uncertainty_m_s"]=float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"],velocity_discrepancy))
        if not fit["fit_accepted"]:
            raise ValueError(f"Thomson fit rejected for channel {i}: {'; '.join(fit['fit_quality_reasons'])}")
        if model in {"collective_maxwellian", "collective_kappa"}:
            fit["parameter_provenance"] = {
                "electron_density": "inferred from the measured collective spectrum",
                "electron_temperature": "inferred from the measured collective spectrum",
                "electron_distribution": ("kappa index inferred from the measured collective spectrum" if model == "collective_kappa" else "Maxwellian"),
                "ion_temperature": "externally supplied fixed input; not inferred",
                "ion_velocity": "externally supplied fixed input; not inferred",
                "electron_velocity": "externally supplied fixed input; not inferred",
                "ion_species_and_fractions": "externally supplied fixed composition",
            }
        else:
            fit["parameter_provenance"] = {
                "electron_temperature": "inferred from measured spectral shape under the selected non-collective model",
                "electron_velocity_projection": "inferred from measured spectral centroid under the selected Doppler model",
                "electron_density": "not identifiable from normalized spectral shape; requires absolute calibrated scattered power and optical geometry",
                "ion_parameters": "not inferred by this non-collective electron feature model",
            }
        fit["model_discrepancy_uncertainty_available"] = bool(
            fit.get("model_discrepancy_covariance") is not None
            and float(fit.get("model_discrepancy_scale", 0.0)) > 0.0
        )
        if fit.get("model_discrepancy_covariance") is None:
            fit["model_discrepancy_diagnostics"] = {
                "residual_rms": fit.get("fit_residual_rms"),
                "reduced_chi_square": fit.get("fit_reduced_chi_square"),
                "covariance_included": False,
                "interpretation": "No per-channel model-discrepancy covariance is available because a supported reduced chi-square/noise covariance estimate was not produced.",
            }
        else:
            fit["model_discrepancy_diagnostics"]["covariance_included"] = fit[
                "model_discrepancy_uncertainty_available"
            ]
        density_inputs = None
        if model in {"non_collective_maxwellian", "non_collective_kappa", "non_collective_relativistic_maxwell_juttner"} and all(v is not None for v in (
            laser_power_w, interaction_length_m, collection_solid_angle_sr, optical_efficiency,
            incident_polarization, incident_direction, scattered_direction
        )):
            if not spectral_power_density_calibrated:
                raise ValueError("Absolute density requires spectral_power_density_calibrated=True and signal units W/nm")
            area = float(fit["integrated_scattered_power_W"])
            area_sigma = fit.get("integrated_scattered_power_uncertainty_W")
            relative_input_covariance = (
                joint_density_covariance[6*i:6*(i+1), 6*i:6*(i+1)].copy()
                if joint_density_covariance is not None
                else _channel_covariance(density_relative_covariance, i, y.shape[0], "density_relative_covariance")
            )
            power_uncertainty_in_covariance = (
                relative_input_covariance is not None
                and not np.allclose(relative_input_covariance[0, :], 0.0, rtol=0.0, atol=1e-14)
            )
            density_inputs = {
                "scattered_power_W": area, "incident_power_W": _channel_value(laser_power_w, i, y.shape[0], "laser_power_w"),
                "interaction_length_m": _channel_value(interaction_length_m, i, y.shape[0], "interaction_length_m"),
                "scattering_angle_rad": scattering_angle_rad,
                "collection_solid_angle_sr": _channel_value(collection_solid_angle_sr, i, y.shape[0], "collection_solid_angle_sr"),
                "optical_efficiency": _channel_value(optical_efficiency, i, y.shape[0], "optical_efficiency"),
                # A supplied relative covariance is the complete uncertainty model
                # for the six multiplicative density inputs. Do not also pass the
                # fitted area sigma or independent scalar terms: the density helper
                # correctly rejects mixed covariance/scalar uncertainty inputs.
                "scattered_power_uncertainty_W": (None if power_uncertainty_in_covariance else area_sigma),
                "incident_power_uncertainty_W": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(laser_power_uncertainty_W, i, y.shape[0], "laser_power_uncertainty_W")),
                "interaction_length_uncertainty_m": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(interaction_length_uncertainty_m, i, y.shape[0], "interaction_length_uncertainty_m")),
                "scattering_angle_uncertainty_rad": (None if scattering_angle_uncertainty_rad is None else scattering_angle_unc),
                "collection_solid_angle_uncertainty_sr": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(collection_solid_angle_uncertainty_sr, i, y.shape[0], "collection_solid_angle_uncertainty_sr")),
                "optical_efficiency_uncertainty": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(optical_efficiency_uncertainty, i, y.shape[0], "optical_efficiency_uncertainty")),
                "incident_polarization_vector": _channel_value(incident_polarization, i, y.shape[0], "incident_polarization", vector_length=3),
                "incident_direction_vector": _channel_value(incident_direction, i, y.shape[0], "incident_direction", vector_length=3),
                "scattered_direction_vector": _channel_value(scattered_direction, i, y.shape[0], "scattered_direction", vector_length=3),
                "polarization_factor_uncertainty": (None if density_relative_covariance is not None or joint_density_covariance is not None else _channel_value(polarization_factor_uncertainty, i, y.shape[0], "polarization_factor_uncertainty")),
                "relative_input_covariance": relative_input_covariance,
            }
            density_result = density_from_thomson_scattered_power(**density_inputs)
            fit["electron_density_m3"] = density_result["electron_density_m3"]
            fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
            fit["integrated_scattered_power_W"] = area
            fit["integrated_scattered_power_uncertainty_W"] = area_sigma
            fit["differential_cross_section_m2_sr"] = density_result["differential_cross_section_m2_sr"]
            fit["scattering_angle_relative_sensitivity"] = density_result.get(
                "scattering_angle_relative_sensitivity"
            )
            fit["density_method"] = "absolute integrated Thomson power using shared polarization-aware density inversion"
            fit["density_uncertainty_model"] = density_result["uncertainty_model"]
            if density_inputs.get("relative_input_covariance") is not None:
                fit["integrated_scattered_power_fit_uncertainty_W"] = float(
                    fit.get("integrated_scattered_power_fit_uncertainty_W", area_sigma)
                )
                covariance_area_sigma = abs(float(area)) * float(np.sqrt(max(
                    np.asarray(density_inputs["relative_input_covariance"], dtype=float)[0, 0], 0.0
                )))
                fit["integrated_scattered_power_covariance_uncertainty_W"] = covariance_area_sigma
                fit["integrated_scattered_power_uncertainty_W"] = covariance_area_sigma
            # Retain the exact, already validated physical inputs so a later
            # joint spectral fit can recompute density from its own final area.
            fit["_density_inputs_for_joint_refit"] = dict(density_inputs)
        else:
            fit["electron_density_m3"] = fit.get("electron_density_m3")
            if "density_status" not in fit:
                fit["density_status"] = "not_identifiable_from_spectral_shape_alone"
        fit["channel_index"] = int(i)
        fit["radius_m"] = float(radii[i])
        fit["instrument_response_used"] = instrument_response is not None
        fit["instrument_response_uncertainty_supplied"] = lsf_unc is not None
        missing_uncertainties=[]
        if unc is None: missing_uncertainties.append("signal_uncertainty")
        if lsf_unc is None: missing_uncertainties.append("instrument_response_uncertainty")
        if laser_wavelength_uncertainty_nm is None: missing_uncertainties.append("laser_wavelength_uncertainty_nm")
        if scattering_angle_uncertainty_rad is None: missing_uncertainties.append("scattering_angle_uncertainty_rad")
        if model == "non_collective_relativistic_maxwell_juttner":
            if laser_wavelength_uncertainty_nm is None:
                missing_uncertainties.append("laser_wavelength_uncertainty_nm")
            if scattering_angle_uncertainty_rad is None and incident_direction_unc is None and scattered_direction_unc is None:
                missing_uncertainties.append("scattering_geometry_uncertainty")
            if polarization_unc is None:
                missing_uncertainties.append("incident_polarization_uncertainty")
        if model in {"collective_maxwellian", "collective_kappa"}:
            missing_uncertainties.extend(["ion_temperature_uncertainty_eV","ion_velocity_uncertainty_m_s","electron_velocity_uncertainty_m_s"])
            if lsf_unc is not None:
                # Collective response uncertainty is propagated in the external-input
                # Monte Carlo below; do not mark it missing before that calculation.
                fit["instrument_response_uncertainty_propagated"] = False
        if all(v is not None for v in (laser_power_w,interaction_length_m,collection_solid_angle_sr,optical_efficiency)) and density_relative_covariance is None:
            for key,value in (("laser_power_uncertainty_W",laser_power_uncertainty_W),
                              ("interaction_length_uncertainty_m",interaction_length_uncertainty_m),
                              ("collection_solid_angle_uncertainty_sr",collection_solid_angle_uncertainty_sr),
                              ("optical_efficiency_uncertainty",optical_efficiency_uncertainty),
                              ("polarization_factor_uncertainty",polarization_factor_uncertainty)):
                if value is None: missing_uncertainties.append(key)
        fit["uncertainty_inputs_missing"] = missing_uncertainties
        fit["uncertainty_completeness"] = "partial" if missing_uncertainties else "all_supported_inputs_supplied"
        if lsf_unc is not None and model in {"non_collective_maxwellian", "non_collective_kappa", "non_collective_relativistic_maxwell_juttner"}:
            lsf_sigma = lsf_unc if lsf_unc.ndim == 2 else lsf_unc[i]
            # A shared 2-D response matrix represents one common instrument
            # realization for every channel. Reuse the same random stream and
            # retain failed draw slots so cross-channel samples stay aligned.
            rng = np.random.default_rng(7319 if lsf_unc.ndim == 2 else 7319 + i)
            te_samples, velocity_samples, area_samples = [], [], []
            for _ in range(int(instrument_response_mc_samples)):
                sampled = np.maximum(response + rng.normal(0.0, lsf_sigma), 0.0)
                colsum = sampled.sum(axis=0)
                if np.any(colsum <= 0):
                    te_samples.append(np.nan)
                    velocity_samples.append(np.nan)
                    area_samples.append(np.nan)
                    continue
                sampled /= colsum[None, :]
                try:
                    if model == "non_collective_kappa":
                        draw = _fit_kappa(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), sampled, ui)
                    elif model == "non_collective_relativistic_maxwell_juttner":
                        draw = fit_relativistic_maxwell_juttner(
                            xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad),
                            sampled, ui, incident_direction=probe_vec, scattered_direction=scatter_vec,
                            incident_polarization=incident_polarization,
                        )
                    else:
                        draw = _fit_noncollective(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), sampled, ui)
                    te_samples.append(draw["electron_temperature_eV"])
                    velocity_samples.append(draw["electron_velocity_projection_m_s"])
                    area_samples.append(float(draw["integrated_scattered_power_W"])
                                        if "integrated_scattered_power_W" in draw
                                        and np.isfinite(draw["integrated_scattered_power_W"]) else np.nan)
                except (ValueError, RuntimeError):
                    te_samples.append(np.nan)
                    velocity_samples.append(np.nan)
                    area_samples.append(np.nan)
            required_successes = max(8, int(np.ceil(instrument_response_mc_min_success_fraction * instrument_response_mc_samples)))
            te_valid = np.asarray(te_samples, dtype=float)
            velocity_valid = np.asarray(velocity_samples, dtype=float)
            area_valid = np.asarray(area_samples, dtype=float)
            te_valid = te_valid[np.isfinite(te_valid)]
            velocity_valid = velocity_valid[np.isfinite(velocity_valid)]
            area_valid = area_valid[np.isfinite(area_valid)]
            if te_valid.size >= required_successes:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], np.std(te_valid, ddof=1)))
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], np.std(velocity_valid, ddof=1)))
                fit["instrument_response_uncertainty_propagated"] = True
                fit["instrument_response_mc_successful_samples"] = int(te_valid.size)
                fit["instrument_response_temperature_uncertainty_eV"] = float(np.std(te_valid, ddof=1))
                fit["instrument_response_velocity_uncertainty_m_s"] = float(np.std(velocity_valid, ddof=1))
                if lsf_unc.ndim == 2:
                    fit["_shared_lsf_temperature_draws"] = np.asarray(te_samples, dtype=float)
                    fit["_shared_lsf_velocity_draws"] = np.asarray(velocity_samples, dtype=float)
                    fit["_shared_lsf_area_draws"] = np.asarray(area_samples, dtype=float)
                fit["integrated_scattered_power_fit_uncertainty_W"] = float(
                    fit.get("integrated_scattered_power_uncertainty_W") or 0.0
                )
                fit["integrated_scattered_power_lsf_uncertainty_W"] = (
                    float(np.std(area_valid, ddof=1)) if area_valid.size >= required_successes else None
                )
                if fit["integrated_scattered_power_lsf_uncertainty_W"] is not None:
                    fit["integrated_scattered_power_uncertainty_W"] = float(np.hypot(
                        fit["integrated_scattered_power_fit_uncertainty_W"],
                        fit["integrated_scattered_power_lsf_uncertainty_W"],
                    ))
            else:
                fit["instrument_response_uncertainty_propagated"] = False
                fit["instrument_response_mc_successful_samples"] = int(te_valid.size)
                fit["integrated_scattered_power_lsf_uncertainty_W"] = None
                fit["instrument_response_uncertainty_warning"] = "Too few valid LSF perturbation fits to estimate its contribution."
        if density_inputs is not None and lsf_unc is not None:
            if area_valid.size >= required_successes:
                fit_area_uncertainty = float(
                    fit.get("integrated_scattered_power_fit_uncertainty_W",
                            fit.get("integrated_scattered_power_uncertainty_W")) or 0.0
                )
                lsf_area_uncertainty = float(np.std(area_valid, ddof=1))
                if density_inputs.get("relative_input_covariance") is None:
                    combined_area_uncertainty = float(np.hypot(fit_area_uncertainty, lsf_area_uncertainty))
                    density_inputs["scattered_power_uncertainty_W"] = combined_area_uncertainty
                else:
                    covariance = np.asarray(density_inputs["relative_input_covariance"], dtype=float).copy()
                    lsf_relative_sigma = 0.0 if area == 0.0 else lsf_area_uncertainty / abs(area)
                    if np.allclose(covariance[0, :], 0.0, rtol=0.0, atol=1e-14) and np.allclose(
                        covariance[:, 0], 0.0, rtol=0.0, atol=1e-14
                    ):
                        # Spectral/response uncertainty is supplied separately;
                        # preserve the optical-factor covariance and combine the
                        # independent area contributions as a scalar marginal.
                        existing_area_sigma = float(
                            density_inputs.get("scattered_power_uncertainty_W") or 0.0
                        )
                        combined_area_uncertainty = float(np.hypot(
                            existing_area_sigma, lsf_area_uncertainty
                        ))
                        density_inputs["scattered_power_uncertainty_W"] = combined_area_uncertainty
                    else:
                        # A complete covariance already contains area uncertainty;
                        # add the independent LSF contribution to its power block.
                        covariance[0, 0] += lsf_relative_sigma**2
                        density_inputs["relative_input_covariance"] = covariance
                        density_inputs["scattered_power_uncertainty_W"] = None
                        combined_area_uncertainty = float(
                            abs(area) * np.sqrt(max(covariance[0, 0], 0.0))
                        )
                density_result = density_from_thomson_scattered_power(**density_inputs)
                fit["electron_density_m3"] = density_result["electron_density_m3"]
                fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
                fit["integrated_scattered_power_fit_uncertainty_W"] = fit_area_uncertainty
                fit["integrated_scattered_power_lsf_uncertainty_W"] = lsf_area_uncertainty
                fit["integrated_scattered_power_uncertainty_W"] = combined_area_uncertainty
                fit["density_uncertainty_model"] = density_result["uncertainty_model"]
                fit["instrument_response_density_uncertainty_propagated"] = True
            else:
                fit["instrument_response_density_uncertainty_propagated"] = False
                fit["uncertainty_inputs_missing"].append("instrument_response_uncertainty_not_propagated_to_absolute_density")
                fit["uncertainty_completeness"] = "partial"
        elif density_inputs is not None:
            fit["instrument_response_density_uncertainty_propagated"] = False

        if model in {"non_collective_maxwellian", "non_collective_kappa"}:
            te = fit["electron_temperature_eV"]
            external_te_sigma = thomson_electron_temperature_from_spectral_width(
                float(fit["intrinsic_sigma_nm"]), float(laser_wavelength_nm),
                float(scattering_angle_rad), sigma_intrinsic_sigma_nm=0.0,
                sigma_laser_wavelength_nm=laser_wavelength_unc,
                sigma_scattering_angle_rad=scattering_angle_unc,
            )["uncertainty_electron_temperature_eV"]
            fit["electron_temperature_external_uncertainty_eV"] = external_te_sigma
            fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                fit["electron_temperature_uncertainty_eV"], external_te_sigma))
            velocity = fit["electron_velocity_projection_m_s"]
            velocity_external = thomson_velocity_projection(
                float(fit["fitted_center_nm"]),float(laser_wavelength_nm),
                float(scattering_angle_rad),sigma_fitted_center_nm=0.0,
                sigma_laser_wavelength_nm=laser_wavelength_unc,
                sigma_scattering_angle_rad=scattering_angle_unc,
            )
            velocity_laser_angle_sigma = velocity_external["uncertainty_electron_velocity_projection_m_s"]
            velocity_laser_sigma = 0.0
            velocity_angle_sigma = float(velocity_laser_angle_sigma)
            external_velocity_sigma = float(np.hypot(velocity_laser_sigma, velocity_angle_sigma))
            fit["electron_velocity_projection_external_uncertainty_m_s"] = external_velocity_sigma
            fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                fit["electron_velocity_projection_uncertainty_m_s"], external_velocity_sigma))
        if model == "non_collective_relativistic_maxwell_juttner" and any(value is not None for value in (
            laser_wavelength_uncertainty_nm, scattering_angle_uncertainty_rad,
            incident_direction_unc, scattered_direction_unc, polarization_unc,
        )):
            rng_external = np.random.default_rng(84271 + i)
            external_te_samples, external_velocity_samples = [], []
            incident_nominal = np.asarray(probe_vec, dtype=float)
            scattered_nominal = np.asarray(scatter_vec, dtype=float)
            polarization_nominal = np.asarray(
                _channel_value(incident_polarization, i, y.shape[0], "incident_polarization", vector_length=3),
                dtype=float,
            )
            for _ in range(int(instrument_response_mc_samples)):
                sampled_laser = float(laser_wavelength_nm) + (
                    0.0 if laser_wavelength_uncertainty_nm is None
                    else float(rng_external.normal(0.0, laser_wavelength_unc))
                )
                if sampled_laser <= 0.0:
                    continue
                sampled_incident = incident_nominal.copy()
                sampled_scattered = scattered_nominal.copy()
                if incident_direction_unc is not None:
                    sampled_incident += rng_external.normal(0.0, incident_direction_unc)
                if scattered_direction_unc is not None:
                    sampled_scattered += rng_external.normal(0.0, scattered_direction_unc)
                if np.linalg.norm(sampled_incident) <= 0.0 or np.linalg.norm(sampled_scattered) <= 0.0:
                    continue
                sampled_incident /= np.linalg.norm(sampled_incident)
                sampled_scattered /= np.linalg.norm(sampled_scattered)
                if scattering_angle_uncertainty_rad is not None:
                    sampled_angle = float(scattering_angle_rad) + float(
                        rng_external.normal(0.0, scattering_angle_unc)
                    )
                    if not 0.0 < sampled_angle < np.pi:
                        continue
                    tangent = sampled_scattered - np.dot(sampled_scattered, sampled_incident) * sampled_incident
                    tangent_norm = float(np.linalg.norm(tangent))
                    if tangent_norm <= 1e-12:
                        continue
                    tangent /= tangent_norm
                    sampled_scattered = (
                        np.cos(sampled_angle) * sampled_incident
                        + np.sin(sampled_angle) * tangent
                    )
                sampled_angle = float(np.arccos(np.clip(
                    np.dot(sampled_incident, sampled_scattered), -1.0, 1.0
                )))
                if not 0.0 < sampled_angle < np.pi:
                    continue
                sampled_polarization = polarization_nominal.copy()
                if polarization_unc is not None:
                    sampled_polarization += rng_external.normal(0.0, polarization_unc)
                sampled_polarization -= (
                    np.dot(sampled_polarization, sampled_incident) * sampled_incident
                )
                polarization_norm = float(np.linalg.norm(sampled_polarization))
                if polarization_norm <= 1e-12:
                    continue
                sampled_polarization /= polarization_norm
                try:
                    draw = fit_relativistic_maxwell_juttner(
                        xi, y[i], sampled_laser, sampled_angle, response, ui,
                        incident_direction=sampled_incident,
                        scattered_direction=sampled_scattered,
                        incident_polarization=sampled_polarization,
                    )
                    external_te_samples.append(float(draw["electron_temperature_eV"]))
                    external_velocity_samples.append(float(draw["electron_velocity_projection_m_s"]))
                except (ValueError, RuntimeError):
                    continue
            required_external = max(
                8, int(np.ceil(instrument_response_mc_min_success_fraction * instrument_response_mc_samples))
            )
            fit["relativistic_external_parameter_mc_requested"] = int(instrument_response_mc_samples)
            fit["relativistic_external_parameter_mc_successful_samples"] = len(external_te_samples)
            if len(external_te_samples) >= required_external:
                te_external_sigma = float(np.std(external_te_samples, ddof=1))
                velocity_external_sigma = float(np.std(external_velocity_samples, ddof=1))
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], te_external_sigma
                ))
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], velocity_external_sigma
                ))
                fit["relativistic_external_parameter_uncertainty_propagated"] = True
                fit["relativistic_external_temperature_uncertainty_eV"] = te_external_sigma
                fit["relativistic_external_velocity_uncertainty_m_s"] = velocity_external_sigma
                fit["uncertainty_inputs_missing"] = [
                    name for name in fit["uncertainty_inputs_missing"]
                    if name not in {
                        "laser_wavelength_uncertainty_nm",
                        "scattering_geometry_uncertainty",
                        "incident_polarization_uncertainty",
                    }
                ]
                fit["uncertainty_completeness"] = (
                    "partial" if fit["uncertainty_inputs_missing"] else "all_supported_inputs_supplied"
                )
            else:
                fit["relativistic_external_parameter_uncertainty_propagated"] = False
                fit["uncertainty_inputs_missing"].append(
                    "relativistic_external_parameter_monte_carlo_propagation_failed"
                )
                fit["uncertainty_completeness"] = "partial"
        if lsf_unc is not None and model not in {"collective_maxwellian", "collective_kappa"} and not fit.get("instrument_response_uncertainty_propagated",False):
            fit["uncertainty_inputs_missing"].append("instrument_response_uncertainty_propagation_failed")
            fit["uncertainty_completeness"] = "partial"
        results.append(fit)
    shared_lsf_covariance = None
    shared_lsf_covariance_order = []
    if lsf_unc is not None and lsf_unc.ndim == 2 and results and all(
        "_shared_lsf_temperature_draws" in item for item in results
    ):
        # Same draw index is the same shared instrument-response realization.
        draw_columns = []
        for channel_index, item in enumerate(results):
            draw_columns.extend([
                item["_shared_lsf_temperature_draws"],
                item["_shared_lsf_velocity_draws"],
                item["_shared_lsf_area_draws"],
            ])
            shared_lsf_covariance_order.extend([
                f"channel_{channel_index}_electron_temperature_eV",
                f"channel_{channel_index}_electron_velocity_projection_m_s",
                f"channel_{channel_index}_integrated_scattered_power_W",
            ])
        aligned_draws = np.column_stack(draw_columns)
        valid_draws = np.all(np.isfinite(aligned_draws), axis=1)
        if np.count_nonzero(valid_draws) >= 2:
            shared_lsf_covariance = np.cov(aligned_draws[valid_draws], rowvar=False, ddof=1)
    shared_response_density_covariance = None
    if shared_lsf_covariance is not None:
        channel_count = len(results)
        area_indices = np.asarray([3 * channel + 2 for channel in range(channel_count)], dtype=int)
        area_covariance = shared_lsf_covariance[np.ix_(area_indices, area_indices)]
        density_values = np.asarray([
            np.nan if item.get("electron_density_m3") is None else item["electron_density_m3"]
            for item in results
        ], dtype=float)
        area_values = np.asarray([
            item.get("integrated_scattered_power_W", np.nan) for item in results
        ], dtype=float)
        shared_response_density_covariance = np.full(
            (channel_count, channel_count), np.nan, dtype=float
        )
        valid_density = np.isfinite(density_values) & np.isfinite(area_values) & (area_values != 0.0)
        if np.any(valid_density):
            scale = density_values[valid_density] / area_values[valid_density]
            shared_response_density_covariance[np.ix_(valid_density, valid_density)] = (
                np.outer(scale, scale)
                * area_covariance[np.ix_(valid_density, valid_density)]
            )
    for item in results:
        item.pop("_shared_lsf_temperature_draws", None)
        item.pop("_shared_lsf_velocity_draws", None)
        item.pop("_shared_lsf_area_draws", None)
    joint_summary = None
    if joint_covariance is not None:
        from scipy.optimize import least_squares
        nchan, nlam = y.shape
        initial = np.concatenate([np.asarray(item["fit_parameters"], dtype=float) for item in results])
        lower_parts, upper_parts = [], []
        joint_parameter_count=5 if model=="non_collective_maxwellian" else 6
        for i in range(nchan):
            dx = float(np.median(np.diff(xs[i])))
            span_i = float(xs[i, -1] - xs[i, 0])
            if model=="non_collective_maxwellian":
                lower_parts.append([0.0, xs[i, 0], max(dx * 0.2, 1e-12), -np.inf, -np.inf])
                upper_parts.append([np.inf, xs[i, -1], span_i, np.inf, np.inf])
            else:
                lower_parts.append([0.0, xs[i, 0], max(dx * 0.2, 1e-12), 1.51, -np.inf, -np.inf])
                upper_parts.append([np.inf, xs[i, -1], span_i, np.inf, np.inf, np.inf])
        lower = np.asarray(lower_parts).ravel()
        upper = np.asarray(upper_parts).ravel()
        def joint_forward(parameters):
            predicted = []
            for channel in range(nchan):
                p=parameters[joint_parameter_count*channel:joint_parameter_count*(channel+1)]
                if model=="non_collective_maxwellian":
                    amp,center,sigma,offset,slope=p
                    signal_component=amp*np.exp(-0.5*((xs[channel]-center)/sigma)**2)
                else:
                    amp,center,sigma,kappa,offset,slope=p
                    df=2.0*kappa-1.0
                    scale=sigma*np.sqrt((df-2.0)/df)
                    signal_component=amp*(1.0+((xs[channel]-center)/scale)**2/df)**(-kappa)
                predicted.append(_apply_response(channel_responses[channel], signal_component, xs[channel])
                                 + offset + slope*(xs[channel]-center))
            return np.asarray(predicted)
        measured_flat = y.ravel()
        def joint_residual(parameters):
            return np.linalg.solve(joint_cholesky, (joint_forward(parameters)-y).ravel())
        joint_fit = least_squares(joint_residual, initial, bounds=(lower, upper), x_scale="jac", max_nfev=30000)
        joint_column_norms = np.linalg.norm(joint_fit.jac, axis=0)
        joint_scaled_jacobian = joint_fit.jac / np.maximum(joint_column_norms, np.finfo(float).tiny)[None, :]
        joint_rank = int(np.linalg.matrix_rank(joint_scaled_jacobian))
        joint_condition = float(np.linalg.cond(joint_scaled_jacobian))
        if not joint_fit.success or joint_rank < initial.size or not np.isfinite(joint_condition) or joint_condition > 1e12:
            raise ValueError("Joint Thomson GLS fit rejected: optimizer failure or non-identifiable channel parameters")
        parameter_covariance = np.linalg.pinv(joint_fit.jac.T @ joint_fit.jac)
        dof_joint = max(1, measured_flat.size-initial.size)
        standardized = joint_residual(joint_fit.x)
        reduced_chi2_joint = float(standardized @ standardized / dof_joint)
        if not np.isfinite(reduced_chi2_joint) or reduced_chi2_joint>10.0:
            raise ValueError("Joint Thomson GLS fit rejected: reduced chi-square exceeds the default acceptance ceiling of 10")
        # Transform the full parameter covariance to all channel Te and flow values.
        def physical_vector(parameters):
            values=[]
            for channel in range(nchan):
                p=parameters[joint_parameter_count*channel:joint_parameter_count*(channel+1)]
                center=float(p[1]); sigma=float(p[2])
                te=thomson_electron_temperature_from_spectral_width(
                    sigma,float(laser_wavelength_nm),float(scattering_angle_rad)
                )["electron_temperature_eV"]
                velocity=thomson_velocity_projection(
                    center,float(laser_wavelength_nm),float(scattering_angle_rad)
                )["electron_velocity_projection_m_s"]
                values.extend([te,velocity])
            return np.asarray(values,dtype=float)
        base_physical=physical_vector(joint_fit.x)
        for channel in range(nchan):
            te_value=base_physical[2*channel]
            velocity_value=base_physical[2*channel+1]
            if te_value/ELECTRON_REST_ENERGY_EV>=0.01:
                raise ValueError("Joint Thomson GLS fit rejected: non-relativistic electron-temperature limit exceeded")
            if abs(velocity_value)>=0.1*C_M_S:
                raise ValueError("Joint Thomson GLS fit rejected: non-relativistic flow-projection limit exceeded")
        transform=np.zeros((2*nchan,joint_parameter_count*nchan))
        for j in range(joint_parameter_count*nchan):
            step=np.cbrt(np.finfo(float).eps)*max(abs(float(joint_fit.x[j])),1.0)
            plus=joint_fit.x.copy(); minus=joint_fit.x.copy()
            plus[j]+=step; minus[j]-=step
            transform[:,j]=(physical_vector(plus)-physical_vector(minus))/(2.0*step)
        physical_covariance=transform@parameter_covariance@transform.T
        # Excess lack-of-fit is reported as a separate model-discrepancy component.
        discrepancy_scale=max(reduced_chi2_joint-1.0,0.0)
        discrepancy_covariance=discrepancy_scale*physical_covariance
        total_physical_covariance=physical_covariance+discrepancy_covariance
        kappa_parameter_covariance = None
        total_kappa_parameter_covariance = None
        if model == "non_collective_kappa":
            kappa_transform = np.zeros((nchan, joint_parameter_count*nchan))
            for channel in range(nchan):
                kappa_transform[channel, joint_parameter_count*channel+3] = 1.0
            kappa_parameter_covariance = (
                kappa_transform @ parameter_covariance @ kappa_transform.T
            )
            total_kappa_parameter_covariance = (
                kappa_parameter_covariance
                + discrepancy_scale*kappa_parameter_covariance
            )
        joint_fitted=joint_forward(joint_fit.x)
        for channel in range(nchan):
            p=joint_fit.x[joint_parameter_count*channel:joint_parameter_count*(channel+1)]
            block=parameter_covariance[joint_parameter_count*channel:joint_parameter_count*(channel+1),
                                        joint_parameter_count*channel:joint_parameter_count*(channel+1)]
            te_index=2*channel; velocity_index=te_index+1
            fit=results[channel]
            if model=="non_collective_maxwellian":
                amp,center,sigma,offset,slope=p
                extra={}
            else:
                amp,center,sigma,kappa,offset,slope=p
                extra={"kappa_index":float(kappa)}
            fit.update({
                "amplitude":float(amp),"fitted_center_nm":float(center),"intrinsic_sigma_nm":float(sigma),
                "background_intercept":float(offset),"background_slope_per_nm":float(slope),**extra,
                "fit_parameters":p.copy(),"fit_covariance":block.copy(),
                "electron_temperature_eV":float(base_physical[te_index]),
                "electron_velocity_projection_m_s":float(base_physical[velocity_index]),
                "electron_temperature_uncertainty_eV":float(np.sqrt(max(total_physical_covariance[te_index,te_index],0.0))),
                "electron_velocity_projection_uncertainty_m_s":float(np.sqrt(max(total_physical_covariance[velocity_index,velocity_index],0.0))),
                "fitted_spectrum":joint_fitted[channel].copy(),
                "residual":(y[channel]-joint_fitted[channel]).copy(),
                "joint_fit_used":True,
                "joint_channel_parameter_covariance":parameter_covariance.copy(),
                "model_discrepancy_uncertainty_available": bool(discrepancy_scale > 0.0),
                "model_discrepancy_covariance_included": bool(discrepancy_scale > 0.0),
            })
            if model == "non_collective_kappa":
                fit["kappa_index_uncertainty"] = float(np.sqrt(max(
                    total_kappa_parameter_covariance[channel, channel], 0.0
                )))
                fit["kappa_parameter_covariance"] = kappa_parameter_covariance.copy()
                fit["total_kappa_parameter_covariance"] = total_kappa_parameter_covariance.copy()
            channel_residual = fit["residual"]
            channel_slice = slice(channel * y.shape[1], (channel + 1) * y.shape[1])
            channel_spectral_covariance = joint_covariance[channel_slice, channel_slice]
            channel_cholesky = np.linalg.cholesky(channel_spectral_covariance)
            channel_whitened_residual = np.linalg.solve(channel_cholesky, channel_residual)
            channel_dof = max(1, channel_residual.size - joint_parameter_count)
            channel_reduced_chi_square = float(
                np.dot(channel_whitened_residual, channel_whitened_residual) / channel_dof
            )
            fit["fit_residual_rms"] = float(np.sqrt(np.mean(channel_residual**2)))
            fit["fit_reduced_chi_square"] = channel_reduced_chi_square
            fit["fit_degrees_of_freedom"] = int(channel_dof)
            fit["joint_fit_reduced_chi_square"] = float(reduced_chi2_joint)
            fit["joint_fit_degrees_of_freedom"] = int(dof_joint)
            fit["joint_fit_accepted"] = True
            fit["fit_acceptance_basis"] = "accepted_joint_generalized_least_squares"
            fit["fit_accepted"] = True
            fit["fit_quality_status"] = "accepted_joint_gls_with_channel_marginal_diagnostics"
            fit["fit_quality_reasons"] = []
            fit["channel_marginal_fit_quality_status"] = (
                "marginal_reduced_chi_square_within_ceiling"
                if channel_reduced_chi_square <= 10.0
                else "marginal_reduced_chi_square_exceeds_ceiling_joint_fit_accepted"
            )
            physical_indices = [te_index, velocity_index]
            fit["physical_parameter_covariance"] = physical_covariance[np.ix_(physical_indices, physical_indices)].copy()
            fit["model_discrepancy_covariance"] = (discrepancy_scale * block).copy()
            fit["model_discrepancy_physical_parameter_covariance"] = discrepancy_covariance[np.ix_(physical_indices, physical_indices)].copy()
            fit["total_fit_covariance_including_discrepancy"] = block + discrepancy_scale * block
            fit["total_physical_parameter_covariance"] = total_physical_covariance[np.ix_(physical_indices, physical_indices)].copy()
            fit["model_discrepancy_scale"] = float(discrepancy_scale)
            fit["model_discrepancy_diagnostics"] = {
                "residual_rms": fit["fit_residual_rms"],
                "channel_marginal_reduced_chi_square": channel_reduced_chi_square,
                "joint_reduced_chi_square": float(reduced_chi2_joint),
                "covariance_included": bool(discrepancy_scale > 0.0),
                "interpretation": "Joint-GLS residual-based model-adequacy covariance; it is not experimental validation.",
            }
            if model=="non_collective_maxwellian":
                area=float(amp*sigma*np.sqrt(2.0*np.pi))
                def area_function(v): return float(v[0]*v[2]*np.sqrt(2*np.pi))
            else:
                df=2.0*kappa-1.0
                scale=sigma*np.sqrt((df-2.0)/df)
                area=float(amp*scale*np.sqrt(df*np.pi)*np.exp(gammaln(kappa-0.5)-gammaln(kappa)))
                def area_function(v):
                    dfv=2.0*v[3]-1.0
                    scalev=v[2]*np.sqrt((dfv-2.0)/dfv)
                    return float(v[0]*scalev*np.sqrt(dfv*np.pi)*np.exp(gammaln(v[3]-0.5)-gammaln(v[3])))
            grad=np.zeros(p.size)
            for j in range(p.size):
                step=1e-5*max(abs(float(p[j])),1.0)
                plus=p.copy(); minus=p.copy(); plus[j]+=step; minus[j]-=step
                grad[j]=(area_function(plus)-area_function(minus))/(2.0*step)
            joint_area_sigma=float(np.sqrt(max(grad@block@grad,0.0)))
            lsf_area_sigma=fit.get("integrated_scattered_power_lsf_uncertainty_W")
            if lsf_area_sigma is not None:
                joint_area_sigma=float(np.hypot(joint_area_sigma,float(lsf_area_sigma)))
            fit["integrated_scattered_power_W"]=area
            fit["integrated_scattered_power_fit_uncertainty_W"]=float(np.sqrt(max(grad@block@grad,0.0)))
            fit["integrated_scattered_power_uncertainty_W"]=joint_area_sigma
            te_external = fit.get("electron_temperature_external_uncertainty_eV")
            te_lsf = fit.get("instrument_response_temperature_uncertainty_eV")
            if te_external is not None:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], float(te_external)))
            if te_lsf is not None:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], float(te_lsf)))
            velocity_external = fit.get("electron_velocity_projection_external_uncertainty_m_s")
            velocity_lsf = fit.get("instrument_response_velocity_uncertainty_m_s")
            if velocity_external is not None:
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], float(velocity_external)))
            if velocity_lsf is not None:
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], float(velocity_lsf)))
            # The absolute density belongs to the final joint-GLS spectrum, not
            # to the superseded independent-channel fit. Re-run the same physical
            # inversion with the final joint area and its uncertainty.
            if fit.get("density_method") and fit.get("density_status") != "not_identifiable_from_spectral_shape_alone":
                density_inputs = fit.get("_density_inputs_for_joint_refit")
                if density_inputs is not None:
                    density_inputs = dict(density_inputs)
                    density_inputs["scattered_power_W"] = area
                    if density_inputs.get("relative_input_covariance") is None:
                        density_inputs["scattered_power_uncertainty_W"] = joint_area_sigma
                    else:
                        covariance = np.asarray(density_inputs["relative_input_covariance"], dtype=float).copy()
                        old_sigma = float(np.sqrt(max(covariance[0, 0], 0.0))) * float(
                            results[channel]["integrated_scattered_power_W"]
                        )
                        new_relative_sigma = 0.0 if area == 0.0 else joint_area_sigma / abs(area)
                        old_relative_sigma = 0.0 if results[channel]["integrated_scattered_power_W"] == 0.0 else old_sigma / abs(float(results[channel]["integrated_scattered_power_W"]))
                        if old_relative_sigma > 0.0:
                            scale_cov = new_relative_sigma / old_relative_sigma
                            covariance[0, :] *= scale_cov
                            covariance[:, 0] *= scale_cov
                        covariance[0, 0] = new_relative_sigma**2
                        density_inputs["relative_input_covariance"] = covariance
                    density_result = density_from_thomson_scattered_power(**density_inputs)
                    fit["electron_density_m3"] = density_result["electron_density_m3"]
                    fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
                    fit["density_uncertainty_model"] = density_result["uncertainty_model"]
                    if density_inputs.get("relative_input_covariance") is not None:
                        fit["integrated_scattered_power_covariance_uncertainty_W"] = joint_area_sigma
                        fit["integrated_scattered_power_uncertainty_W"] = joint_area_sigma
        joint_summary={
            "method":"joint generalized least squares across all channels",
            "channel_major_covariance_order":True,
            "joint_parameter_covariance":parameter_covariance,
            "physical_parameter_order":[name for i in range(nchan) for name in (f"channel_{i}_electron_temperature_eV",f"channel_{i}_electron_velocity_projection_m_s")],
            "joint_parameter_count_per_channel":joint_parameter_count,
            "physical_parameter_covariance":physical_covariance,
            "kappa_parameter_covariance":kappa_parameter_covariance,
            "total_kappa_parameter_covariance":total_kappa_parameter_covariance,
            "kappa_parameter_order": (
                [f"channel_{i}_kappa_index" for i in range(nchan)]
                if model == "non_collective_kappa" else None
            ),
            "model_discrepancy_covariance":discrepancy_covariance,
            "model_discrepancy_uncertainty_available": bool(discrepancy_scale > 0.0),
            "total_physical_parameter_covariance":total_physical_covariance,
            "reduced_chi_square":reduced_chi2_joint,
            "jacobian_rank":joint_rank,"jacobian_condition_number":joint_condition,
            "degrees_of_freedom":int(dof_joint),
            "model_discrepancy_scale":discrepancy_scale,
            "model_discrepancy_interpretation":"Excess whitened residual variance above unity is reported as a separate covariance component; it is a model-adequacy diagnostic, not experimental validation.",
        }
    for channel_result in results:
        channel_result.pop("_density_inputs_for_joint_refit", None)
    density_joint_covariance_output = None
    if joint_density_covariance is not None:
        channel_count = len(results)
        density_joint_covariance_output = np.full((channel_count, channel_count), np.nan, dtype=float)
        relative_density_gradient = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
        for row_index, row_result in enumerate(results):
            density_row = row_result.get("electron_density_m3")
            if density_row is None:
                continue
            row_slice = slice(6*row_index, 6*(row_index+1))
            for column_index, column_result in enumerate(results):
                density_column = column_result.get("electron_density_m3")
                if density_column is None:
                    continue
                column_slice = slice(6*column_index, 6*(column_index+1))
                relative_block = joint_density_covariance[row_slice, column_slice]
                density_joint_covariance_output[row_index, column_index] = (
                    float(density_row) * float(density_column)
                    * float(relative_density_gradient @ relative_block @ relative_density_gradient)
                )
        for channel_index, channel_result in enumerate(results):
            if channel_result.get("electron_density_m3") is not None:
                channel_result["electron_density_joint_covariance_diagonal_m6"] = float(
                    density_joint_covariance_output[channel_index, channel_index]
                )
    density_angle_covariance_output = None
    if scattering_angle_uncertainty_rad is not None and any(
        item.get("scattering_angle_relative_sensitivity") is not None
        and item.get("electron_density_m3") is not None for item in results
    ):
        angle_density_sensitivity = np.asarray([
            (
                float(item["electron_density_m3"])
                * float(item["scattering_angle_relative_sensitivity"])
                if item.get("electron_density_m3") is not None
                and item.get("scattering_angle_relative_sensitivity") is not None
                else np.nan
            )
            for item in results
        ], dtype=float)
        density_angle_covariance_output = (
            np.outer(angle_density_sensitivity, angle_density_sensitivity)
            * float(scattering_angle_unc) ** 2
        )
        if density_joint_covariance_output is not None:
            valid_angle = np.isfinite(density_angle_covariance_output)
            density_joint_covariance_output[valid_angle] += density_angle_covariance_output[valid_angle]
    density_cross_channel_covariance_available = bool(
        (
            density_joint_covariance_output is not None
            or density_angle_covariance_output is not None
        )
        and sum(result.get("electron_density_m3") is not None for result in results) >= 2
    )
    output = {
        "model": model, "channel_count": len(results), "channels": results,
        "joint_inference": joint_summary,
        "radius_m": radii.copy(),
        "electron_temperature_profile_eV": np.array([r.get("electron_temperature_eV", np.nan) for r in results]),
        "electron_temperature_uncertainty_profile_eV": np.array([r.get("electron_temperature_uncertainty_eV", np.nan) for r in results]),
        "electron_density_profile_m3": np.array([r.get("electron_density_m3", np.nan) if r.get("electron_density_m3") is not None else np.nan for r in results]),
        "electron_density_uncertainty_profile_m3": np.array([r.get("electron_density_uncertainty_m3", np.nan) if r.get("electron_density_uncertainty_m3") is not None else np.nan for r in results]),
        "electron_density_joint_covariance_m6": density_joint_covariance_output,
        "electron_density_angle_covariance_m6": density_angle_covariance_output,
        "electron_density_shared_response_covariance_m6": shared_response_density_covariance,
        "electron_density_external_parameter_covariance_m6": None,
        "electron_density_cross_channel_covariance_available": density_cross_channel_covariance_available,
        "shared_instrument_response_parameter_covariance": shared_lsf_covariance,
        "shared_instrument_response_parameter_covariance_order": shared_lsf_covariance_order,
        "shared_instrument_response_cross_channel_covariance_available": bool(
            shared_lsf_covariance is not None and len(results) >= 2
        ),
        "electron_velocity_projection_profile_m_s": np.array([r.get("electron_velocity_projection_m_s", np.nan) for r in results]),
        "electron_velocity_projection_uncertainty_profile_m_s": np.array([r.get("electron_velocity_projection_uncertainty_m_s", np.nan) for r in results]),
        "limitations": [
            *(
                ["Collective Maxwellian inference uses PlasmaPy's Maxwellian kinetic dielectric response; ion temperatures, composition and flow vectors are fixed external inputs."]
                if model == "collective_maxwellian" else
                ["Collective kappa inference uses the implemented unmagnetized longitudinal dielectric response with projected isotropic kappa electrons and Maxwellian ions; supplied external-input and response-matrix 1-sigma uncertainties are propagated by Monte Carlo, but cross-correlations among distinct external inputs and full channel-major GLS are not implemented."]
                if model == "collective_kappa" else
                ["Relativistic Maxwell-Juttner inference assumes an isotropic rest-frame electron distribution and bulk flow parallel to the scattering vector; supplied response-matrix uncertainty is propagated by Monte Carlo, but laser/angle/beam-direction and incident-polarization uncertainty propagation is not yet complete."]
                if model == "non_collective_relativistic_maxwell_juttner" else
                []
            ),
            "Absolute non-collective density requires calibrated W/nm spectra and all listed optical geometry/polarization factors; first-order uncertainty propagation includes only uncertainty terms supplied by the researcher.",
            "For non-collective Maxwellian/Kappa/relativistic fits, supplied response-matrix uncertainty is propagated by Monte Carlo; absolute-density uncertainty includes only supplied optical-factor uncertainty terms and excludes unprovided correlations and model discrepancy.",
        ],
    }
    if model in collective_models and any(v is not None for v in (
        ion_t_unc,ion_v_unc,electron_v_unc,lsf_unc,laser_wavelength_uncertainty_nm,
        scattering_angle_uncertainty_rad,incident_direction_unc,scattered_direction_unc
    )):
        if lsf_unc is not None and (callable(instrument_response) or np.asarray(instrument_response).ndim==1):
            raise ValueError("Collective response uncertainty requires a measured 2-D or channel-specific 3-D response matrix")
        rng=np.random.default_rng(91427)
        draws=[]
        for _ in range(int(collective_external_parameter_mc_samples)):
            sampled_laser=float(laser_wavelength_nm)+rng.normal(0.0,laser_wavelength_unc)
            if sampled_laser<=0: continue
            sampled_angle=float(scattering_angle_rad)
            sampled_incident_direction=incident_direction
            sampled_scattered_direction=scattered_direction
            if scattering_angle_uncertainty_rad is not None:
                sampled_angle += float(rng.normal(0.0, scattering_angle_unc))
                if not 0.0 < sampled_angle < np.pi: continue
                # Rotate the scattered beam in the plane defined by the nominal
                # incident/scattered directions, preserving a normalized beam pair.
                incident_unit=np.asarray(incident_direction,dtype=float)
                incident_unit=incident_unit/np.linalg.norm(incident_unit)
                scattered_unit=np.asarray(scattered_direction,dtype=float)
                scattered_unit=scattered_unit/np.linalg.norm(scattered_unit)
                tangent=scattered_unit-np.dot(scattered_unit,incident_unit)*incident_unit
                tangent_norm=np.linalg.norm(tangent)
                if tangent_norm<=1e-12: continue
                tangent=tangent/tangent_norm
                sampled_scattered_direction=np.cos(sampled_angle)*incident_unit+np.sin(sampled_angle)*tangent
            elif incident_direction_unc is not None or scattered_direction_unc is not None:
                incident_draw=np.asarray(incident_direction,dtype=float)
                scattered_draw=np.asarray(scattered_direction,dtype=float)
                if incident_direction_unc is not None:
                    incident_draw=incident_draw+rng.normal(0.0,incident_direction_unc)
                if scattered_direction_unc is not None:
                    scattered_draw=scattered_draw+rng.normal(0.0,scattered_direction_unc)
                if np.linalg.norm(incident_draw)<=0 or np.linalg.norm(scattered_draw)<=0: continue
                incident_draw=incident_draw/np.linalg.norm(incident_draw)
                scattered_draw=scattered_draw/np.linalg.norm(scattered_draw)
                sampled_angle=float(np.arccos(np.clip(np.dot(incident_draw,scattered_draw),-1.0,1.0)))
                if not 0.0 < sampled_angle <= np.pi: continue
                sampled_incident_direction=incident_draw
                sampled_scattered_direction=scattered_draw
            sampled_ion_t=ion_t.copy() if ion_t_unc is None else np.maximum(ion_t+rng.normal(0.0,ion_t_unc),np.finfo(float).tiny)
            sampled_ion_v=ion_v.copy() if ion_v_unc is None else ion_v+rng.normal(0.0,ion_v_unc)
            sampled_electron_v=electron_v.copy() if electron_v_unc is None else electron_v+rng.normal(0.0,electron_v_unc.reshape(1,3))
            sampled_response=instrument_response
            if lsf_unc is not None:
                sampled_response=np.asarray(instrument_response,dtype=float).copy()
                if sampled_response.ndim==2:
                    sampled_response=np.maximum(sampled_response+rng.normal(0.0,lsf_unc),0.0)
                    sums=sampled_response.sum(axis=0)
                    if np.any(sums<=0): continue
                    sampled_response/=sums[None,:]
                else:
                    valid_response=True
                    for channel in range(sampled_response.shape[0]):
                        sigma_response=lsf_unc[channel] if lsf_unc.ndim==3 else lsf_unc
                        sampled_response[channel]=np.maximum(sampled_response[channel]+rng.normal(0.0,sigma_response),0.0)
                        sums=sampled_response[channel].sum(axis=0)
                        if np.any(sums<=0):
                            valid_response=False
                            break
                        sampled_response[channel]/=sums[None,:]
                    if not valid_response: continue
            try:
                draw=analyze_thomson_spectra(
                    wavelength_nm,spectra,radius_m,laser_wavelength_nm=sampled_laser,
                    scattering_angle_rad=sampled_angle,instrument_response=sampled_response,
                    signal_uncertainty=signal_uncertainty,model=model,ion_species=ion_species,
                    ion_fractions=ion_fractions,ion_temperature_eV=sampled_ion_t,
                    ion_velocity_m_s=sampled_ion_v,electron_velocity_m_s=sampled_electron_v,
                    incident_direction=sampled_incident_direction,scattered_direction=sampled_scattered_direction,
                    collective_initial_density_m3=collective_initial_density_m3,
                    collective_initial_temperature_eV=collective_initial_temperature_eV,
                    collective_density_bounds_m3=collective_density_bounds_m3,
                    collective_temperature_bounds_eV=collective_temperature_bounds_eV,
                    collective_kappa_initial=collective_kappa_initial,
                    collective_kappa_bounds=collective_kappa_bounds,
                    collective_external_parameter_mc_samples=20,
                )
                vector=[]
                for channel_result in draw["channels"]:
                    density_value=channel_result.get("electron_density_m3")
                    vector.extend([np.nan if density_value is None else float(density_value),
                                   float(channel_result["electron_temperature_eV"])])
                    if model == "collective_kappa":
                        vector.append(float(channel_result["kappa_index"]))
                    vector.append(float(channel_result["electron_velocity_projection_m_s"]))
                    vector.extend(np.asarray(channel_result["ion_velocity_projection_m_s"],dtype=float).tolist())
                draws.append(vector)
            except (ValueError,RuntimeError,ImportError,np.linalg.LinAlgError):
                continue
        output["collective_external_mc_successes"]=len(draws)
        output["collective_external_mc_requested"]=int(collective_external_parameter_mc_samples)
        required=max(12,int(np.ceil(.7*collective_external_parameter_mc_samples)))
        if len(draws)>=required:
            draw_array=np.asarray(draws,dtype=float)
            valid_columns=np.all(np.isfinite(draw_array),axis=0)
            covariance=np.zeros((draw_array.shape[1],draw_array.shape[1]))
            if np.any(valid_columns):
                covariance[np.ix_(valid_columns,valid_columns)]=np.cov(draw_array[:,valid_columns],rowvar=False,ddof=1)
                for channel in range(len(results)):
                    stride = (4 if model == "collective_kappa" else 3) + len(ion_species)
                    base = channel * stride
                    density_col, temperature_col = base, base + 1
                    if model == "collective_kappa":
                        kappa_col, velocity_col = base + 2, base + 3
                        ion_columns = tuple(range(base + 4, base + 4 + len(ion_species)))
                        results[channel]["kappa_index_uncertainty"] = float(np.hypot(
                            results[channel]["kappa_index_uncertainty"],
                            np.sqrt(max(covariance[kappa_col, kappa_col], 0.0))))
                    else:
                        velocity_col = base + 2
                        ion_columns = tuple(range(base + 3, base + 3 + len(ion_species)))
                    results[channel]["electron_temperature_uncertainty_eV"]=float(np.hypot(
                        results[channel]["electron_temperature_uncertainty_eV"],
                        np.sqrt(max(covariance[temperature_col,temperature_col],0.0))))
                    velocity_mc_sigma=float(np.sqrt(max(covariance[velocity_col,velocity_col],0.0)))
                    base_velocity_sigma=results[channel].get("electron_velocity_projection_uncertainty_m_s")
                    if base_velocity_sigma is None:
                        results[channel]["electron_velocity_projection_uncertainty_m_s"]=(
                            velocity_mc_sigma if electron_v_unc is not None or scattering_angle_uncertainty_rad is not None
                            or incident_direction_unc is not None or scattered_direction_unc is not None else None
                        )
                    else:
                        results[channel]["electron_velocity_projection_uncertainty_m_s"]=float(np.hypot(
                            base_velocity_sigma,velocity_mc_sigma))
                    if results[channel].get("electron_density_m3") is not None:
                        results[channel]["electron_density_uncertainty_m3"]=float(np.hypot(
                            results[channel]["electron_density_uncertainty_m3"],
                            np.sqrt(max(covariance[density_col,density_col],0.0))))
                    ion_uncertainties=[]
                    for ion_col in ion_columns:
                        ion_sigma=float(np.sqrt(max(covariance[ion_col,ion_col],0.0)))
                        ion_uncertainties.append(
                            ion_sigma if ion_v_unc is not None or scattering_angle_uncertainty_rad is not None
                            or incident_direction_unc is not None or scattered_direction_unc is not None else None
                        )
                    results[channel]["ion_velocity_projection_uncertainty_m_s"]=ion_uncertainties
            covariance_order=[]
            for i in range(len(results)):
                covariance_order.extend([
                    f"channel_{i}_electron_density_m3",
                    f"channel_{i}_electron_temperature_eV",
                ])
                if model == "collective_kappa":
                    covariance_order.append(f"channel_{i}_kappa_index")
                covariance_order.append(f"channel_{i}_electron_velocity_projection_m_s")
                covariance_order.extend(
                    f"channel_{i}_ion_{species_index}_velocity_projection_m_s"
                    for species_index in range(len(ion_species))
                )
            output["collective_external_parameter_covariance_order"]=covariance_order
            output["collective_external_parameter_covariance"]=covariance
            external_stride = (4 if model == "collective_kappa" else 3) + len(ion_species)
            density_columns = [channel * external_stride for channel in range(len(results))]
            external_density_covariance = covariance[np.ix_(density_columns, density_columns)].copy()
            for channel_index, channel_result in enumerate(results):
                if channel_result.get("electron_density_m3") is None:
                    external_density_covariance[channel_index, :] = np.nan
                    external_density_covariance[:, channel_index] = np.nan
            output["electron_density_external_parameter_covariance_m6"] = external_density_covariance
            output["electron_density_cross_channel_covariance_available"] = bool(
                sum(result.get("electron_density_m3") is not None for result in results) >= 2
            )
            output["collective_external_uncertainty_propagated"]=True
            output["electron_temperature_uncertainty_profile_eV"]=np.array([r.get("electron_temperature_uncertainty_eV",np.nan) for r in results])
            output["electron_density_uncertainty_profile_m3"]=np.array([r.get("electron_density_uncertainty_m3",np.nan) if r.get("electron_density_m3") is not None else np.nan for r in results])
            output["electron_velocity_projection_uncertainty_profile_m_s"]=np.array([
                np.nan if r.get("electron_velocity_projection_uncertainty_m_s") is None
                else r["electron_velocity_projection_uncertainty_m_s"] for r in results])
            for channel_result in results:
                channel_result["collective_external_parameter_uncertainty_propagated"]=True
                if lsf_unc is not None:
                    channel_result["instrument_response_uncertainty_propagated"]=True
                    channel_result["instrument_response_uncertainty_mc_successful_samples"]=len(draws)
                channel_result["collective_external_parameter_covariance_order"]=covariance_order
        else:
            output["collective_external_uncertainty_propagated"]=False
            output["collective_external_uncertainty_warning"]="Too few successful external-parameter/response perturbation refits."
            for channel_result in results:
                channel_result["collective_external_parameter_uncertainty_propagated"]=False
                channel_result.setdefault("uncertainty_inputs_missing", []).append(
                    "collective_external_parameter_monte_carlo_propagation_failed"
                )
                channel_result["uncertainty_completeness"] = "partial"
                if lsf_unc is not None:
                    channel_result["instrument_response_uncertainty_propagated"]=False
                    channel_result["instrument_response_uncertainty_warning"]=output["collective_external_uncertainty_warning"]
                    channel_result.setdefault("uncertainty_inputs_missing", []).append(
                        "instrument_response_uncertainty_propagation_failed"
                    )
                    channel_result["uncertainty_completeness"] = "partial"

    # Attach model-independent observed spectral-distribution diagnostics to every
    # channel, using the background nuisance term already fitted by the selected
    # physical forward model. This avoids a second, unrelated edge-background fit.
    from .thomson_distribution import analyze_thomson_spectral_distribution
    spectral_distribution_results = []
    for channel_index, channel_result in enumerate(results):
        channel_uncertainty = None if unc is None else unc[channel_index]
        channel_covariance = None
        if joint_covariance is not None:
            channel_slice = slice(channel_index * y.shape[1], (channel_index + 1) * y.shape[1])
            channel_covariance = joint_covariance[channel_slice, channel_slice]
            channel_uncertainty = None
        background_reference_nm = float(
            channel_result.get("fitted_center_nm", laser_wavelength_nm)
        )
        fitted_background = (
            float(channel_result["background_intercept"])
            + float(channel_result["background_slope_per_nm"])
            * (xs[channel_index] - background_reference_nm)
        )
        try:
            distribution = analyze_thomson_spectral_distribution(
                xs[channel_index],
                y[channel_index],
                instrument_response_matrix=channel_responses[channel_index],
                signal_uncertainty=channel_uncertainty,
                signal_covariance=channel_covariance,
                background_signal=fitted_background,
            )
            distribution["analysis_status"] = "completed"
            distribution["background_source"] = (
                "background nuisance component from the selected Thomson physical forward fit"
            )
            distribution["interpretation"] = (
                "Observed wavelength-space moments and a response-convolved Gaussian shape diagnostic. "
                "Physical Te and flow remain those inferred by the selected optical Thomson forward model; "
                "these descriptive moments are not a unique arbitrary electron VDF inversion."
            )
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as exc:
            distribution = {
                "analysis_status": "unavailable_for_this_channel",
                "reason": str(exc),
                "interpretation": (
                    "The primary physical Thomson inference remains available; spectral-distribution "
                    "diagnostics require sufficient wavelength coverage and a positive signed area/variance."
                ),
            }
        channel_result["spectral_distribution"] = distribution
        spectral_distribution_results.append(distribution)
    output["spectral_distribution"] = spectral_distribution_results

    # A fitted spectral integral is not a watt-valued scattered power unless the
    # measured spectral density is explicitly calibrated in W/nm. Preserve the
    # numerical integral while making its actual unit contract unambiguous.
    output["absolute_scattered_power_available"] = bool(spectral_power_density_calibrated)
    if not spectral_power_density_calibrated:
        for channel_result in results:
            area = channel_result.pop("integrated_scattered_power_W", None)
            area_uncertainty = channel_result.pop("integrated_scattered_power_uncertainty_W", None)
            if area is not None:
                channel_result["integrated_spectral_signal_input_units_nm"] = area
            if area_uncertainty is not None:
                channel_result["integrated_spectral_signal_uncertainty_input_units_nm"] = area_uncertainty
            for old_key, new_key in (
                ("integrated_scattered_power_fit_uncertainty_W", "integrated_spectral_signal_fit_uncertainty_input_units_nm"),
                ("integrated_scattered_power_lsf_uncertainty_W", "integrated_spectral_signal_lsf_uncertainty_input_units_nm"),
                ("integrated_scattered_power_covariance_uncertainty_W", "integrated_spectral_signal_covariance_uncertainty_input_units_nm"),
            ):
                if old_key in channel_result:
                    channel_result[new_key] = channel_result.pop(old_key)
            channel_result["integrated_spectral_signal_unit"] = "input spectral-intensity unit × nm"
            channel_result["absolute_scattered_power_available"] = False
    return output

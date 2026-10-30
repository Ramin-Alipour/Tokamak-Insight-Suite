"""End-to-end, spatially resolved Thomson spectral inference.

Input spectra are measured, wavelength-calibrated and intensity-calibrated data.
This module does not calibrate detectors or repair/prepare measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import gammaln

from .thomson_scattering import density_from_thomson_scattered_power

ELECTRON_REST_ENERGY_EV = 510_998.95069
C_M_S = 299_792_458.0
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
    te = (thermal_sigma/laser_nm)**2 * (ELECTRON_REST_ENERGY_EV/4.0) / np.sin(angle/2.0)**2
    # The fitted spectral shift measures the velocity projection along q.
    q_projection = 2.0*np.sin(angle/2.0)
    velocity = C_M_S * (center-laser_nm)/(laser_nm*q_projection)
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
        "model": "non-collective, non-relativistic Maxwellian electron spectrum",
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
    if center-3*sigma < wavelength[0] or center+3*sigma > wavelength[-1]:
        raise ValueError("Wavelength coverage must include at least ±3 fitted RMS widths")
    te=(sigma/laser_nm)**2*(ELECTRON_REST_ENERGY_EV/4.0)/np.sin(angle/2.0)**2
    q_projection=2.0*np.sin(angle/2.0)
    velocity=C_M_S*(center-laser_nm)/(laser_nm*q_projection)
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
            "integrated_scattered_power_W":area,"integrated_scattered_power_uncertainty_W":area_unc,
            "reduced_chi_square":float(np.sum((residual/uncertainty)**2)/dof) if uncertainty is not None else None,
            "fitted_spectrum":observed(wavelength,*p),
            "model":"non-collective isotropic kappa electron VDF (Student-t projected spectrum)",
            "assumptions":["non-collective single-particle scattering","isotropic kappa electron velocity distribution",
                           "non-relativistic velocities","finite-variance kappa index > 1.5",
                           "known scattering geometry"]}


def _fit_relativistic_maxwell_juttner(wavelength, signal, laser_nm, angle, response, uncertainty):
    """Fit a 1-D projected Maxwell-Juttner VDF with relativistic Doppler mapping.

    This is a projected-velocity approximation: it retains the relativistic
    Maxwell-Juttner marginal and relativistic Doppler kinematics along q, but
    does not integrate the full 3-D incident/scattered photon geometry.
    """
    signal_scale=float(np.max(np.abs(signal)))
    if not np.isfinite(signal_scale) or signal_scale<=0:
        raise ValueError("signal must have a positive finite scale")
    signal_scaled=signal/signal_scale
    uncertainty_scaled=None if uncertainty is None else uncertainty/signal_scale
    spacing=float(np.median(np.diff(wavelength)))
    span=float(wavelength[-1]-wavelength[0])
    edge=max(2,wavelength.size//10)
    baseline=float(np.median(np.r_[signal_scaled[:edge],signal_scaled[-edge:]]))
    peak=int(np.argmax(signal_scaled))
    amp0=max(float(signal_scaled[peak]-baseline),np.finfo(float).eps)
    p0=[amp0,float(wavelength[peak]),100.0,baseline,0.0]
    lower=[0.0,wavelength[0],1e-4,-np.inf,-np.inf]
    upper=[np.inf,wavelength[-1],np.inf,np.inf,np.inf]
    sine=float(np.sin(angle/2.0))
    def component(x, amp, center, temperature, offset=0.0, slope=0.0):
        ratio=x/center
        beta=(ratio-1.0)/(sine*(ratio+1.0))
        valid=np.abs(beta)<1.0
        profile=np.zeros_like(x,dtype=float)
        if np.any(valid):
            b=beta[valid]
            gamma=1.0/np.sqrt(1.0-b*b)
            theta=max(float(temperature)/ELECTRON_REST_ENERGY_EV,1e-12)
            # The common exp(-1/theta) normalization cancels with free amplitude.
            marginal=np.exp(-(gamma-1.0)/theta)*(gamma*gamma+2.0*theta*gamma+2.0*theta*theta)
            jacobian=2.0/(sine*center*(ratio[valid]+1.0)**2)
            profile[valid]=amp*marginal*jacobian
        return profile+offset+slope*(x-center)
    def observed(x,amp,center,temperature,offset,slope):
        physical=component(x,amp,center,temperature,0.0,0.0)
        return _apply_response(response, physical, x)+offset+slope*(x-center)
    p,cov=curve_fit(observed,wavelength,signal_scaled,p0=p0,bounds=(lower,upper),
                    sigma=uncertainty_scaled,absolute_sigma=uncertainty_scaled is not None,maxfev=100000)
    parameter_scales=np.array([signal_scale,1.0,1.0,signal_scale,signal_scale])
    p=np.asarray(p,dtype=float)*parameter_scales
    cov=np.asarray(cov,dtype=float)*np.outer(parameter_scales,parameter_scales)
    amp,center,temperature,offset,slope=map(float,p)
    residual=signal-observed(wavelength,*p)
    dof=max(1,wavelength.size-len(p))
    q_projection=2.0*sine
    velocity=C_M_S*(center-laser_nm)/(laser_nm*q_projection)
    def area_from_params(values):
        a,cen,temp,off,sl=values
        theta_local=max(float(temp)/ELECTRON_REST_ENERGY_EV,1e-12)
        beta_grid=np.linspace(-1.0+1e-9,1.0-1e-9,20001)
        gamma_grid=1.0/np.sqrt(1.0-beta_grid**2)
        marginal_grid=np.exp(-(gamma_grid-1.0)/theta_local)*(
            gamma_grid**2+2.0*theta_local*gamma_grid+2.0*theta_local**2)
        return float(a*np.trapz(marginal_grid,beta_grid))
    area=area_from_params(p)
    grad=np.zeros(len(p))
    for j in range(len(p)):
        step=np.cbrt(np.finfo(float).eps)*max(abs(float(p[j])),1.0)
        plus=np.array(p,dtype=float); minus=np.array(p,dtype=float)
        plus[j]+=step; minus[j]-=step
        grad[j]=(area_from_params(plus)-area_from_params(minus))/(2.0*step)
    area_unc=float(np.sqrt(max(float(grad@cov@grad),0.0)))
    return {"electron_temperature_eV":temperature,
            "electron_temperature_uncertainty_eV":float(np.sqrt(max(cov[2,2],0.0))),
            "electron_velocity_projection_m_s":float(velocity),
            "electron_velocity_projection_uncertainty_m_s":float(C_M_S*np.sqrt(max(cov[1,1],0.0))/(laser_nm*q_projection)),
            "fitted_center_nm":center,"amplitude":amp,"background_intercept":offset,
            "background_slope_per_nm":slope,"fit_covariance":np.asarray(cov,dtype=float),
            "residual":residual,"reduced_chi_square":float(np.sum((residual/uncertainty)**2)/dof) if uncertainty is not None else None,
            "fitted_spectrum":observed(wavelength,*p),
            "integrated_scattered_power_W":area,"integrated_scattered_power_uncertainty_W":area_unc,
            "model":"projected relativistic Maxwell-Juttner electron VDF with relativistic Doppler mapping",
            "assumptions":["non-collective scattering","isotropic Maxwell-Juttner electron distribution",
                           "relativistic 1-D projected-velocity approximation along the scattering wavevector",
                           "full 3-D relativistic photon-geometry integration is not included"]}

def analyze_thomson_spectra(
    wavelength_nm, spectra, radius_m, *, laser_wavelength_nm, scattering_angle_rad,
    laser_wavelength_uncertainty_nm=None, scattering_angle_uncertainty_rad=None,
    instrument_response=None, signal_uncertainty=None,
    laser_power_w=None, interaction_length_m=None, collection_solid_angle_sr=None,
    optical_efficiency=None, incident_polarization=None, incident_direction=None, scattered_direction=None,
    spectral_power_density_calibrated=False, laser_power_uncertainty_W=None,
    interaction_length_uncertainty_m=None, collection_solid_angle_uncertainty_sr=None,
    optical_efficiency_uncertainty=None, polarization_factor_uncertainty=None,
    instrument_response_uncertainty=None, model="non_collective_maxwellian",
    ion_species=None, ion_fractions=None, ion_temperature_eV=None, ion_velocity_m_s=None, electron_velocity_m_s=None,
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
    x = _finite_array(wavelength_nm)
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
    if model not in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner", "collective_maxwellian"}:
        raise ValueError("model must be non_collective_maxwellian, non_collective_kappa, relativistic_maxwell_juttner or collective_maxwellian")
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")
    if model == "collective_maxwellian":
        if not ion_species or isinstance(ion_species, (str, bytes)):
            raise ValueError("Collective Thomson inference requires an explicit non-empty ion-species list")
        if ion_velocity_m_s is None or electron_velocity_m_s is None:
            raise ValueError("Collective inference requires measured ion/electron flow vectors or explicit zero vectors to declare a stationary-flow assumption")
        if ion_temperature_eV is None:
            raise ValueError("Collective Maxwellian inference requires independently supplied ion temperature(s); no ion temperature is assumed")
        if ion_fractions is None:
            if len(ion_species) > 1:
                raise ValueError("Collective multi-species inference requires explicit ion_fractions summing to one")
            ion_fractions = [1.0]
        if incident_direction is None or scattered_direction is None:
            raise ValueError("Collective Thomson inference requires explicit incident_direction and scattered_direction 3-vectors in the same coordinate frame as the supplied flow vectors")
        probe_vec = _finite_array(incident_direction, "incident_direction", 1)
        scatter_vec = _finite_array(scattered_direction, "scattered_direction", 1)
        if probe_vec.shape != (3,) or scatter_vec.shape != (3,):
            raise ValueError("incident_direction and scattered_direction must each be 3-vectors")
        probe_norm = float(np.linalg.norm(probe_vec))
        scatter_norm = float(np.linalg.norm(scatter_vec))
        if probe_norm <= 0 or scatter_norm <= 0:
            raise ValueError("incident_direction and scattered_direction must be non-zero")
        probe_vec = probe_vec / probe_norm
        scatter_vec = scatter_vec / scatter_norm
        geometry_angle = float(np.arccos(np.clip(np.dot(probe_vec, scatter_vec), -1.0, 1.0)))
        if not np.isclose(geometry_angle, float(scattering_angle_rad), rtol=0.0, atol=1e-5):
            raise ValueError("scattering_angle_rad is inconsistent with incident_direction and scattered_direction")
        try:
            from astropy import units as u
            from plasmapy.diagnostics import thomson
        except ImportError as exc:
            raise ImportError("Collective Thomson inference requires the optional PlasmaPy and Astropy packages") from exc
        # PlasmaPy is used below with an explicit, user-supplied species list.
    else:
        thomson = None

    unc = None if signal_uncertainty is None else _finite_array(signal_uncertainty, "signal_uncertainty", 2)
    if unc is not None and (unc.shape != y.shape or np.any(unc <= 0)):
        raise ValueError("signal_uncertainty must match spectra and be strictly positive")
    lsf_unc = None if instrument_response_uncertainty is None else _finite_array(instrument_response_uncertainty, "instrument_response_uncertainty")
    if lsf_unc is not None and (np.any(lsf_unc < 0) or instrument_response is None or lsf_unc.ndim not in (2,3) or lsf_unc.shape[-2:] != (y.shape[1],y.shape[1]) or (lsf_unc.ndim == 3 and lsf_unc.shape[0] != y.shape[0])):
        raise ValueError("instrument_response_uncertainty must be a non-negative 2-D/3-D matrix uncertainty on the measured wavelength grid")

    results = []
    for i in range(y.shape[0]):
        xi = xs[i]
        response = _response_for_channel(instrument_response, i, y.shape[0], y.shape[1], xi)
        ui = None if unc is None else unc[i]
        if model == "non_collective_maxwellian":
            fit = _fit_noncollective(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), response, ui)
        elif model == "non_collective_kappa":
            fit = _fit_kappa(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), response, ui)
        elif model == "relativistic_maxwell_juttner":
            fit = _fit_relativistic_maxwell_juttner(xi, y[i], float(laser_wavelength_nm), float(scattering_angle_rad), response, ui)
        else:
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
            log_min = np.log10(np.finfo(float).tiny)
            log_max = np.log10(np.finfo(float).max) - 1.0
            fitopt = least_squares(
                lambda log_params: objective(10.0**log_params),
                np.log10([1e18, 100.0]),
                bounds=([log_min, log_min], [log_max, log_max]),
                x_scale="jac", max_nfev=5000,
            )
            ne_fit, te_fit = map(float, 10.0**fitopt.x)
            # Two nonlinear plasma parameters plus three profiled linear nuisance parameters
            # (amplitude, background intercept, and background slope) are estimated.
            dof = max(1, y[i].size-5)
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
                   "reduced_chi_square": float(np.sum((residual_collective/residual_scale)**2)/dof),
                   "ion_species": list(ion_species), "ion_fractions": ion_fract.tolist(),
                   "scattering_parameter_alpha": alpha_value,
                   "scattering_regime": "collective" if alpha_value > 1.0 else "non_collective_or_weakly_collective",
                   "ion_temperature_eV": np.atleast_1d(ion_t).tolist(),
                   "ion_velocity_m_s": ion_v.tolist(), "electron_velocity_m_s": electron_v.tolist()}
            if not fitopt.success:
                fit["electron_density_m3"] = None
                fit["electron_temperature_eV"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["electron_temperature_uncertainty_eV"] = None
                fit["density_status"] = "collective_spectral_optimizer_failed"
            elif alpha_value <= 1.0:
                fit["electron_density_m3"] = None
                fit["electron_density_uncertainty_m3"] = None
                fit["density_status"] = "spectral_shape_does_not_resolve_collective_density"
        density_inputs = None
        if model in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner"} and all(v is not None for v in (
            laser_power_w, interaction_length_m, collection_solid_angle_sr, optical_efficiency,
            incident_polarization, incident_direction, scattered_direction
        )):
            if not spectral_power_density_calibrated:
                raise ValueError("Absolute density requires spectral_power_density_calibrated=True and signal units W/nm")
            area = float(fit["integrated_scattered_power_W"])
            area_sigma = fit.get("integrated_scattered_power_uncertainty_W")
            density_inputs = {
                "scattered_power_W": area, "incident_power_W": laser_power_w,
                "interaction_length_m": interaction_length_m,
                "scattering_angle_rad": scattering_angle_rad,
                "collection_solid_angle_sr": collection_solid_angle_sr,
                "optical_efficiency": optical_efficiency,
                "scattered_power_uncertainty_W": area_sigma,
                "incident_power_uncertainty_W": laser_power_uncertainty_W,
                "interaction_length_uncertainty_m": interaction_length_uncertainty_m,
                "scattering_angle_uncertainty_rad": (None if incident_polarization is not None or scattering_angle_uncertainty_rad is None else scattering_angle_unc),
                "collection_solid_angle_uncertainty_sr": collection_solid_angle_uncertainty_sr,
                "optical_efficiency_uncertainty": optical_efficiency_uncertainty,
                "incident_polarization_vector": incident_polarization,
                "incident_direction_vector": incident_direction,
                "scattered_direction_vector": scattered_direction,
                "polarization_factor_uncertainty": polarization_factor_uncertainty,
            }
            density_result = density_from_thomson_scattered_power(**density_inputs)
            fit["electron_density_m3"] = density_result["electron_density_m3"]
            fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
            fit["integrated_scattered_power_W"] = area
            fit["integrated_scattered_power_uncertainty_W"] = area_sigma
            fit["differential_cross_section_m2_sr"] = density_result["differential_cross_section_m2_sr"]
            fit["density_method"] = "absolute integrated Thomson power using shared polarization-aware density inversion"
            fit["density_uncertainty_model"] = density_result["uncertainty_model"]
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
        if model == "collective_maxwellian":
            missing_uncertainties.extend(["ion_temperature_uncertainty_eV","ion_velocity_uncertainty_m_s","electron_velocity_uncertainty_m_s"])
            if lsf_unc is not None:
                missing_uncertainties.append("instrument_response_uncertainty_not_propagated_for_collective_model")
                fit["instrument_response_uncertainty_propagated"] = False
        if all(v is not None for v in (laser_power_w,interaction_length_m,collection_solid_angle_sr,optical_efficiency)):
            for key,value in (("laser_power_uncertainty_W",laser_power_uncertainty_W),
                              ("interaction_length_uncertainty_m",interaction_length_uncertainty_m),
                              ("collection_solid_angle_uncertainty_sr",collection_solid_angle_uncertainty_sr),
                              ("optical_efficiency_uncertainty",optical_efficiency_uncertainty),
                              ("polarization_factor_uncertainty",polarization_factor_uncertainty)):
                if value is None: missing_uncertainties.append(key)
        fit["uncertainty_inputs_missing"] = missing_uncertainties
        fit["uncertainty_completeness"] = "partial" if missing_uncertainties else "all_supported_inputs_supplied"
        if lsf_unc is not None and model in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner"}:
            lsf_sigma = lsf_unc if lsf_unc.ndim == 2 else lsf_unc[i]
            rng = np.random.default_rng(7319 + i)
            te_samples, velocity_samples, area_samples = [], [], []
            for _ in range(24):
                sampled = np.maximum(response + rng.normal(0.0, lsf_sigma), 0.0)
                colsum = sampled.sum(axis=0)
                if np.any(colsum <= 0):
                    continue
                sampled /= colsum[None, :]
                try:
                    fitter = (_fit_kappa if model == "non_collective_kappa" else (_fit_relativistic_maxwell_juttner if model == "relativistic_maxwell_juttner" else _fit_noncollective))
                    draw = fitter(xi, y[i], float(laser_wavelength_nm),
                                  float(scattering_angle_rad), sampled, ui)
                    te_samples.append(draw["electron_temperature_eV"])
                    velocity_samples.append(draw["electron_velocity_projection_m_s"])
                    if "integrated_scattered_power_W" in draw and np.isfinite(draw["integrated_scattered_power_W"]):
                        area_samples.append(float(draw["integrated_scattered_power_W"]))
                except (ValueError, RuntimeError):
                    continue
            if len(te_samples) >= 8:
                fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                    fit["electron_temperature_uncertainty_eV"], np.std(te_samples, ddof=1)))
                fit["electron_velocity_projection_uncertainty_m_s"] = float(np.hypot(
                    fit["electron_velocity_projection_uncertainty_m_s"], np.std(velocity_samples, ddof=1)))
                fit["instrument_response_uncertainty_propagated"] = True
                fit["instrument_response_mc_successful_samples"] = len(te_samples)
            else:
                fit["instrument_response_uncertainty_propagated"] = False
                fit["instrument_response_mc_successful_samples"] = len(te_samples)
                fit["instrument_response_uncertainty_warning"] = "Too few valid LSF perturbation fits to estimate its contribution."
        if density_inputs is not None and lsf_unc is not None:
            if len(area_samples) >= 8:
                fit_area_uncertainty = density_inputs["scattered_power_uncertainty_W"]
                fit_area_uncertainty = 0.0 if fit_area_uncertainty is None else float(fit_area_uncertainty)
                lsf_area_uncertainty = float(np.std(area_samples, ddof=1))
                combined_area_uncertainty = float(np.hypot(fit_area_uncertainty, lsf_area_uncertainty))
                density_inputs["scattered_power_uncertainty_W"] = combined_area_uncertainty
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

        if model in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner"}:
            te = fit["electron_temperature_eV"]
            te_rel_var = (2.0*laser_wavelength_unc/float(laser_wavelength_nm))**2
            te_rel_var += (np.cos(float(scattering_angle_rad)/2.0)/np.sin(float(scattering_angle_rad)/2.0)
                           * scattering_angle_unc)**2
            fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                fit["electron_temperature_uncertainty_eV"], te*np.sqrt(te_rel_var)))
            velocity = fit["electron_velocity_projection_m_s"]
            velocity_laser_sigma = (C_M_S*abs(fit["fitted_center_nm"])*laser_wavelength_unc
                                    /(float(laser_wavelength_nm)**2*2.0*np.sin(float(scattering_angle_rad)/2.0)))
            velocity_angle_sigma = abs(0.5*velocity*np.cos(float(scattering_angle_rad)/2.0)
                                        /np.sin(float(scattering_angle_rad)/2.0)
                                        *scattering_angle_unc)
            fit["electron_velocity_projection_uncertainty_m_s"] = float(np.sqrt(
                fit["electron_velocity_projection_uncertainty_m_s"]**2
                + velocity_laser_sigma**2 + velocity_angle_sigma**2))
        if lsf_unc is not None and model != "collective_maxwellian" and not fit.get("instrument_response_uncertainty_propagated",False):
            fit["uncertainty_inputs_missing"].append("instrument_response_uncertainty_propagation_failed")
            fit["uncertainty_completeness"] = "partial"
        results.append(fit)
    return {
        "model": model, "channel_count": len(results), "channels": results,
        "radius_m": radii.copy(),
        "electron_temperature_profile_eV": np.array([r.get("electron_temperature_eV", np.nan) for r in results]),
        "electron_temperature_uncertainty_profile_eV": np.array([r.get("electron_temperature_uncertainty_eV", np.nan) for r in results]),
        "electron_density_profile_m3": np.array([r.get("electron_density_m3", np.nan) if r.get("electron_density_m3") is not None else np.nan for r in results]),
        "electron_density_uncertainty_profile_m3": np.array([r.get("electron_density_uncertainty_m3", np.nan) if r.get("electron_density_uncertainty_m3") is not None else np.nan for r in results]),
        "electron_velocity_projection_profile_m_s": np.array([r.get("electron_velocity_projection_m_s", np.nan) for r in results]),
        "electron_velocity_projection_uncertainty_profile_m_s": np.array([r.get("electron_velocity_projection_uncertainty_m_s", np.nan) for r in results]),
        "limitations": [
            "The relativistic Maxwell-Juttner option uses a 1-D projected-velocity approximation; full 3-D relativistic photon-geometry integration is not implemented.",
            "Collective model uses PlasmaPy's Maxwellian kinetic dielectric response; non-Maxwellian and relativistic kinetic models are not included.",
            "Absolute density in the non-collective model requires calibrated W/nm spectra and all listed optical geometry/polarization factors; first-order uncertainty propagation includes only uncertainty terms supplied by the user.",
            "For non-collective fits, supplied LSF uncertainty is propagated by Monte Carlo; absolute-density uncertainty includes only supplied independent optical-factor uncertainties, with correlations and model discrepancy excluded.",
        ],
    }

"""End-to-end, spatially resolved Thomson spectral inference.

Input spectra are measured, wavelength-calibrated and intensity-calibrated data.
This module does not calibrate detectors or repair/prepare measurements.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import curve_fit

from .thomson_scattering import density_from_thomson_scattered_power

ELECTRON_REST_ENERGY_EV = 510_998.95069
C_M_S = 299_792_458.0
R_E_M = 2.8179403262e-15


def _finite_array(value, name, ndim=None):
    a = np.asarray(value, dtype=float)
    if ndim is not None and a.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must contain finite numeric values")
    return a


def _response_for_channel(response, channel, nchan, nlam):
    if response is None:
        return np.eye(nlam)
    r = _finite_array(response, "instrument_response")
    if r.ndim == 2:
        out = r
    elif r.ndim == 3 and r.shape[0] == nchan:
        out = r[channel]
    else:
        raise ValueError("instrument_response must have shape (n_wavelength, n_wavelength) or (n_channel, n_wavelength, n_wavelength)")
    if out.shape != (nlam, nlam) or np.any(out < 0):
        raise ValueError("instrument response must be a non-negative square matrix on the measured wavelength grid")
    sums = out.sum(axis=0)
    if np.any(sums <= 0):
        raise ValueError("each instrument-response column must have positive normalization")
    return out / sums[None, :]


def _gaussian(x, amp, center, sigma, offset, slope):
    return amp * np.exp(-0.5 * ((x-center)/sigma)**2) + offset + slope*(x-center)


def _fit_noncollective(wavelength, signal, laser_nm, angle, response, uncertainty):
    spacing = float(np.median(np.diff(wavelength)))
    span = float(wavelength[-1] - wavelength[0])
    edge = max(2, wavelength.size // 10)
    baseline = float(np.median(np.r_[signal[:edge], signal[-edge:]]))
    peak = int(np.argmax(signal))
    amp0 = max(float(signal[peak] - baseline), np.finfo(float).eps)
    p0 = [amp0, float(wavelength[peak]), max(spacing, span/10), baseline, 0.0]
    lower = [0.0, wavelength[0], max(spacing*0.2, 1e-12), -np.inf, -np.inf]
    upper = [np.inf, wavelength[-1], span, np.inf, np.inf]

    def observed(x, amp, center, sigma, offset, slope):
        intrinsic = _gaussian(x, amp, center, sigma, offset, slope)
        # The LSF acts on the signal component; a slowly varying background is
        # represented in detector space and is therefore not convolved.
        signal_only = amp*np.exp(-0.5*((x-center)/sigma)**2)
        return response @ signal_only + offset + slope*(x-center)

    p, cov = curve_fit(observed, wavelength, signal, p0=p0, bounds=(lower, upper),
                       sigma=uncertainty, absolute_sigma=uncertainty is not None,
                       maxfev=50000)
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
    def area_from_params(values):
        a, cen, sig, off, sl = values
        component = response @ (a*np.exp(-0.5*((wavelength-cen)/sig)**2))
        return float(np.trapz(component, wavelength))
    area = area_from_params(p)
    area_gradient = np.zeros(len(p))
    for j in range(len(p)):
        step = np.sqrt(np.finfo(float).eps)*max(abs(float(p[j])),1.0)
        plus=np.array(p,dtype=float); minus=np.array(p,dtype=float)
        plus[j]+=step; minus[j]-=step
        area_gradient[j]=(area_from_params(plus)-area_from_params(minus))/(2.0*step)
    area_unc=float(np.sqrt(max(float(area_gradient@fit_cov@area_gradient),0.0)))
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
    spacing = float(np.median(np.diff(wavelength)))
    span = float(wavelength[-1] - wavelength[0])
    edge = max(2, wavelength.size // 10)
    baseline = float(np.median(np.r_[signal[:edge], signal[-edge:]]))
    peak = int(np.argmax(signal))
    amp0 = max(float(signal[peak] - baseline), np.finfo(float).eps)
    p0 = [amp0, float(wavelength[peak]), max(spacing, span/10), 3.0, baseline, 0.0]
    lower = [0.0, wavelength[0], max(spacing*0.2, 1e-12), 1.51, -np.inf, -np.inf]
    upper = [np.inf, wavelength[-1], span, 100.0, np.inf, np.inf]
    def intrinsic(x, amp, center, sigma, kappa, offset, slope):
        df = 2.0*kappa-1.0
        scale = sigma*np.sqrt((df-2.0)/df)
        return amp*(1.0+((x-center)/scale)**2/df)**(-kappa)+offset+slope*(x-center)
    def observed(x, amp, center, sigma, kappa, offset, slope):
        component = intrinsic(x, amp, center, sigma, kappa, 0.0, 0.0)
        return response @ component + offset+slope*(x-center)
    p, cov = curve_fit(observed,wavelength,signal,p0=p0,bounds=(lower,upper),
                       sigma=uncertainty,absolute_sigma=uncertainty is not None,maxfev=80000)
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
        component=a*(1.0+((wavelength-cen)/scale_local)**2/df_local)**(-kap)
        return float(np.trapz(response @ component,wavelength))
    area=area_from_params(p)
    area_gradient=np.zeros(len(p))
    for j in range(len(p)):
        step=np.sqrt(np.finfo(float).eps)*max(abs(float(p[j])),1.0)
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
    spacing=float(np.median(np.diff(wavelength)))
    span=float(wavelength[-1]-wavelength[0])
    edge=max(2,wavelength.size//10)
    baseline=float(np.median(np.r_[signal[:edge],signal[-edge:]]))
    peak=int(np.argmax(signal))
    amp0=max(float(signal[peak]-baseline),np.finfo(float).eps)
    p0=[amp0,float(wavelength[peak]),100.0,baseline,0.0]
    lower=[0.0,wavelength[0],1e-4,-np.inf,-np.inf]
    upper=[np.inf,wavelength[-1],1e7,np.inf,np.inf]
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
        return response@physical+offset+slope*(x-center)
    p,cov=curve_fit(observed,wavelength,signal,p0=p0,bounds=(lower,upper),
                    sigma=uncertainty,absolute_sigma=uncertainty is not None,maxfev=100000)
    amp,center,temperature,offset,slope=map(float,p)
    residual=signal-observed(wavelength,*p)
    dof=max(1,wavelength.size-len(p))
    q_projection=2.0*sine
    velocity=C_M_S*(center-laser_nm)/(laser_nm*q_projection)
    def area_from_params(values):
        a,cen,temp,off,sl=values
        return float(np.trapz(response@component(wavelength,a,cen,temp,0.0,0.0),wavelength))
    area=area_from_params(p)
    grad=np.zeros(len(p))
    for j in range(len(p)):
        step=np.sqrt(np.finfo(float).eps)*max(abs(float(p[j])),1.0)
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
    laser_wavelength_uncertainty_nm=0.0, scattering_angle_uncertainty_rad=0.0,
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
    channel-specific (2-D). A response matrix maps true spectral bins to detector
    bins; its columns are normalized internally. Absolute density is emitted only
    when all optical-power/geometry/polarization factors are supplied.
    """
    y = _finite_array(spectra, "spectra", 2)
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
    if not np.isfinite(laser_wavelength_nm) or laser_wavelength_nm <= 0:
        raise ValueError("laser_wavelength_nm must be positive")
    if not np.isfinite(scattering_angle_rad) or not 0 < scattering_angle_rad <= np.pi:
        raise ValueError("scattering_angle_rad must lie in (0, pi]")
    if (not np.isfinite(laser_wavelength_uncertainty_nm) or laser_wavelength_uncertainty_nm < 0
            or not np.isfinite(scattering_angle_uncertainty_rad) or scattering_angle_uncertainty_rad < 0):
        raise ValueError("laser-wavelength and scattering-angle uncertainties must be finite and non-negative")
    if model not in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner", "collective_maxwellian"}:
        raise ValueError("model must be non_collective_maxwellian, non_collective_kappa, relativistic_maxwell_juttner or collective_maxwellian")
    if not isinstance(spectral_power_density_calibrated, (bool, np.bool_)):
        raise ValueError("spectral_power_density_calibrated must be boolean")
    if model == "collective_maxwellian":
        if not ion_species:
            raise ValueError("Collective Thomson inference requires at least one explicitly specified ion species")
        if ion_velocity_m_s is None or electron_velocity_m_s is None:
            raise ValueError("Collective inference requires measured ion/electron flow vectors or explicit zero vectors to declare a stationary-flow assumption")
        if ion_temperature_eV is None:
            raise ValueError("Collective Maxwellian inference requires independently supplied ion temperature(s); no ion temperature is assumed")
        if ion_fractions is None:
            if len(ion_species) > 1:
                raise ValueError("Collective multi-species inference requires explicit ion_fractions summing to one")
            ion_fractions = [1.0]
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
    if lsf_unc is not None and (np.any(lsf_unc < 0) or instrument_response is None or lsf_unc.shape != np.asarray(instrument_response).shape):
        raise ValueError("instrument_response_uncertainty must be non-negative and match instrument_response")

    results = []
    for i in range(y.shape[0]):
        xi = xs[i]
        response = _response_for_channel(instrument_response, i, y.shape[0], y.shape[1])
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
            ion_t = np.atleast_1d(ion_temperature_eV)
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
            probe_vec = np.array([1.0, 0.0, 0.0])
            theta = float(scattering_angle_rad)
            scatter_vec = np.array([np.cos(theta), np.sin(theta), 0.0])
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
                return response @ z
            scale = max(float(np.max(np.abs(y[i]))), np.finfo(float).tiny)
            def objective(params):
                shape = collective_shape(params)
                amp = float(np.dot(y[i], shape)/max(np.dot(shape,shape),1e-300))
                return (amp*shape-y[i])/scale
            fitopt = least_squares(objective, [1e18, 100.0], bounds=([1e12, 0.01],[1e25, 1e7]), max_nfev=5000)
            ne_fit, te_fit = map(float, fitopt.x)
            dof = max(1, y[i].size-fitopt.x.size)
            try:
                parameter_covariance = np.linalg.pinv(fitopt.jac.T @ fitopt.jac) * (2.0*fitopt.cost/dof)
                parameter_uncertainties = np.sqrt(np.maximum(np.diag(parameter_covariance),0.0))
            except np.linalg.LinAlgError:
                parameter_covariance = None
                parameter_uncertainties = np.array([np.nan,np.nan])
            model_spec = collective_shape(fitopt.x)
            alpha_fit, _ = thomson.spectral_density(
                xi*u.nm, float(laser_wavelength_nm)*u.nm, ne_fit/u.m**3,
                T_e=te_fit*u.eV, T_i=ion_t*u.eV, ions=ion_species, ifract=ion_fract,
                ion_vel=ion_v*u.m/u.s, electron_vel=electron_v*u.m/u.s,
                probe_vec=probe_vec, scatter_vec=scatter_vec)
            alpha_value=float(np.asarray(getattr(alpha_fit,"value",alpha_fit)))
            amp_fit = float(np.dot(y[i], model_spec)/max(np.dot(model_spec,model_spec),1e-300))
            fit = {"electron_density_m3": ne_fit, "electron_temperature_eV": te_fit,
                   "fitted_spectrum": amp_fit*model_spec, "residual": y[i]-amp_fit*model_spec,
                   "model": "PlasmaPy Maxwellian kinetic dielectric spectral-density model",
                   "electron_density_uncertainty_m3": float(parameter_uncertainties[0]),
                   "electron_temperature_uncertainty_eV": float(parameter_uncertainties[1]),
                   "fit_covariance": parameter_covariance,
                   "optimizer_success": bool(fitopt.success), "optimizer_message": str(fitopt.message),
                   "ion_species": list(ion_species), "ion_fractions": ion_fract.tolist(),
                   "scattering_parameter_alpha": alpha_value,
                   "scattering_regime": "collective" if alpha_value > 1.0 else "non_collective_or_weakly_collective",
                   "ion_temperature_eV": np.atleast_1d(ion_t).tolist(),
                   "ion_velocity_m_s": ion_v.tolist(), "electron_velocity_m_s": electron_v.tolist()}
        if model in {"non_collective_maxwellian", "non_collective_kappa"} and all(v is not None for v in (
            laser_power_w, interaction_length_m, collection_solid_angle_sr, optical_efficiency,
            incident_polarization, incident_direction, scattered_direction
        )):
            if not spectral_power_density_calibrated:
                raise ValueError("Absolute density requires spectral_power_density_calibrated=True and signal units W/nm")
            area = float(fit["integrated_scattered_power_W"])
            area_sigma = fit.get("integrated_scattered_power_uncertainty_W")
            density_result = density_from_thomson_scattered_power(
                scattered_power_W=area, incident_power_W=laser_power_w,
                interaction_length_m=interaction_length_m,
                scattering_angle_rad=scattering_angle_rad,
                collection_solid_angle_sr=collection_solid_angle_sr,
                optical_efficiency=optical_efficiency,
                scattered_power_uncertainty_W=area_sigma,
                incident_power_uncertainty_W=laser_power_uncertainty_W,
                interaction_length_uncertainty_m=interaction_length_uncertainty_m,
                scattering_angle_uncertainty_rad=(None if incident_polarization is not None else scattering_angle_uncertainty_rad),
                collection_solid_angle_uncertainty_sr=collection_solid_angle_uncertainty_sr,
                optical_efficiency_uncertainty=optical_efficiency_uncertainty,
                incident_polarization_vector=incident_polarization,
                incident_direction_vector=incident_direction,
                scattered_direction_vector=scattered_direction,
                polarization_factor_uncertainty=polarization_factor_uncertainty,
            )
            fit["electron_density_m3"] = density_result["electron_density_m3"]
            fit["electron_density_uncertainty_m3"] = density_result["electron_density_uncertainty_m3"]
            fit["integrated_scattered_power_W"] = area
            fit["integrated_scattered_power_uncertainty_W"] = area_sigma
            fit["differential_cross_section_m2_sr"] = density_result["differential_cross_section_m2_sr"]
            fit["density_method"] = "absolute integrated Thomson power using shared polarization-aware density inversion"
            fit["density_uncertainty_model"] = density_result["uncertainty_model"]
        else:
            fit["electron_density_m3"] = fit.get("electron_density_m3")
            fit["density_status"] = "not_identifiable_from_spectral_shape_alone"
        fit["channel_index"] = int(i)
        fit["radius_m"] = float(radii[i])
        fit["instrument_response_used"] = instrument_response is not None
        fit["instrument_response_uncertainty_supplied"] = lsf_unc is not None
        if lsf_unc is not None and model in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner"}:
            lsf_sigma = lsf_unc if lsf_unc.ndim == 2 else lsf_unc[i]
            rng = np.random.default_rng(7319 + i)
            te_samples, velocity_samples = [], []
            for _ in range(48):
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
                except (ValueError, RuntimeError):
                    continue
            if len(te_samples) >= 12:
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
        if model in {"non_collective_maxwellian", "non_collective_kappa", "relativistic_maxwell_juttner"}:
            te = fit["electron_temperature_eV"]
            te_rel_var = (2.0*float(laser_wavelength_uncertainty_nm)/float(laser_wavelength_nm))**2
            te_rel_var += (np.cos(float(scattering_angle_rad)/2.0)/np.sin(float(scattering_angle_rad)/2.0)
                           * float(scattering_angle_uncertainty_rad))**2
            fit["electron_temperature_uncertainty_eV"] = float(np.hypot(
                fit["electron_temperature_uncertainty_eV"], te*np.sqrt(te_rel_var)))
            velocity = fit["electron_velocity_projection_m_s"]
            velocity_laser_sigma = (C_M_S*abs(fit["fitted_center_nm"])*float(laser_wavelength_uncertainty_nm)
                                    /(float(laser_wavelength_nm)**2*2.0*np.sin(float(scattering_angle_rad)/2.0)))
            velocity_angle_sigma = abs(0.5*velocity*np.cos(float(scattering_angle_rad)/2.0)
                                        /np.sin(float(scattering_angle_rad)/2.0)
                                        *float(scattering_angle_uncertainty_rad))
            fit["electron_velocity_projection_uncertainty_m_s"] = float(np.sqrt(
                fit["electron_velocity_projection_uncertainty_m_s"]**2
                + velocity_laser_sigma**2 + velocity_angle_sigma**2))
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
            "For non-collective fits, supplied instrument-response uncertainty is propagated by Monte Carlo perturbation of the LSF matrix; uncertainty in the optical power/geometry inputs must be supplied separately in future extensions.",
        ],
    }

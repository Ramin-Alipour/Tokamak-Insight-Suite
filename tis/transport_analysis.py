"""Fluctuation transport and divertor heat-flux profile calculations."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import optimize, integrate
from scipy.special import erfc, erfcx
from .constants import E_CHARGE
from .spectral_analysis import _r2

def fluctuation_flux(density: Any, radial_velocity: Any, temperature_eV: Any | None = None) -> dict[str, float]:
    """Compute covariance fluxes from aligned physical-unit time series."""
    n=np.asarray(density,float).ravel(); vr=np.asarray(radial_velocity,float).ravel()
    if len(n)!=len(vr) or len(n)<3: raise ValueError('Density and radial velocity must be aligned and have at least 3 samples.')
    te=None
    if temperature_eV is not None:
        te=np.asarray(temperature_eV,float).ravel()
        if len(te)!=len(n): raise ValueError('Temperature series must have same original length as density.')
    good=np.isfinite(n)&np.isfinite(vr)
    if te is not None: good &= np.isfinite(te)
    n,vr=n[good],vr[good]
    if te is not None: te=te[good]
    if len(n)<3: raise ValueError('Not enough finite aligned samples after applying the common validity mask.')
    nf=n-np.mean(n); vf=vr-np.mean(vr)
    out={'particle_flux_covariance_m-2_s-1':float(np.mean(nf*vf)),'mean_density_m-3':float(np.mean(n)),'mean_radial_velocity_m_s':float(np.mean(vr))}
    if te is not None:
        tf=te-np.mean(te)
        mean_n=float(np.mean(n)); mean_te=float(np.mean(te))
        term_n=mean_te*float(np.mean(nf*vf))
        term_te=mean_n*float(np.mean(tf*vf))
        term_triple=float(np.mean(nf*tf*vf))
        out['density_temperature_velocity_covariance_eV_m-2_s-1']=term_triple
        out['electron_heat_flux_density_fluctuation_component_W_m-2']=float(1.5*E_CHARGE*term_n)
        out['electron_heat_flux_temperature_fluctuation_component_W_m-2']=float(1.5*E_CHARGE*term_te)
        out['electron_heat_flux_triple_fluctuation_component_W_m-2']=float(1.5*E_CHARGE*term_triple)
        out['electron_heat_flux_proxy_W_m-2']=float(1.5*E_CHARGE*(term_n+term_te+term_triple))
        out['heat_flux_note']='Decomposition of <n T_e delta(v_r)> into density, temperature and triple-fluctuation terms. Requires synchronized calibrated series; it is not a complete transport power balance.'
    out['note']='Covariance estimates require synchronized measurements, calibrated units and justified radial-velocity reconstruction.'
    return out


def exponential_heat_flux_fit(distance_m: Any, heat_flux_W_m2: Any) -> dict[str, Any]:
    """Fit q(s)=q0 exp(-(s-s0)/lambda_q)+background with covariance diagnostics."""
    x=np.asarray(distance_m,float).ravel(); y=np.asarray(heat_flux_W_m2,float).ravel()
    if len(x) != len(y): raise ValueError('Target coordinates and heat-flux values must have equal lengths.')
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<8 or len(np.unique(x))<8: raise ValueError('At least 8 distinct valid target-coordinate points are required for a stable three-parameter fit.')
    o=np.argsort(x); x,y=x[o],y[o]
    span=float(np.ptp(x))
    if span<=0: raise ValueError("Target coordinate must span a nonzero interval.")
    base0=float(np.percentile(y,10)); q0=max(float(np.max(y)-base0),1e-12)
    def fn(z,q,l,b): return q*np.exp(-(z-x[0])/l)+b
    lower=[0,span/10000,-np.inf]; upper=[np.inf,span*100,np.inf]
    try:
        p,cov=optimize.curve_fit(fn,x,y,p0=[q0,max(span/5,1e-9),base0],bounds=(lower,upper),maxfev=50000)
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        raise ValueError(f"Exponential heat-flux fit failed to converge: {exc}") from exc
    if cov.shape!=(3,3) or not np.all(np.isfinite(cov)):
        raise ValueError("Exponential fit covariance is singular or non-finite; inspect target coverage and signal-to-noise.")
    pred=fn(x,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
    boundary=bool(np.isclose(p[1],lower[1],rtol=1e-3,atol=0) or np.isclose(p[1],upper[1],rtol=1e-3,atol=0))
    return {'q0_W_m-2':float(p[0]),'lambda_q_m':float(p[1]),'background_W_m-2':float(p[2]),
            'standard_error_q0_W_m-2':float(se[0]),'standard_error_lambda_q_m':float(se[1]),
            'standard_error_background_W_m-2':float(se[2]),'r_squared':_r2(y,pred),'n_points':int(len(x)),
            'parameter_at_bound':boundary,
            'note':'Covariance uncertainties assume the fitted model and independent homoscedastic residuals. Inspect residuals, target mapping, parameter identifiability and boundary flags; this is not a heat-transport model.'}


def eich_heat_flux_fit(distance_m: Any, heat_flux_W_m2: Any) -> dict[str, Any]:
    """Fit the standard Eich-like divertor heat-flux profile including Gaussian spreading.

    q(s) = q0/2 * exp(-((s-s0)/S)^2) * erfcx(S/(2*lambda_q) - (s-s0)/S) + q_bg.
    The fit is descriptive and assumes the supplied coordinate is already mapped to target
    distance; it does not infer equilibrium mapping or account for diagnostic response.
    """
    x=np.asarray(distance_m,float).ravel(); y=np.asarray(heat_flux_W_m2,float).ravel()
    if len(x) != len(y): raise ValueError('Target coordinates and heat-flux values must have equal lengths.')
    valid=np.isfinite(x)&np.isfinite(y); x,y=x[valid],y[valid]
    if len(x)<15 or len(np.unique(x))<15:
        raise ValueError("Eich fit requires at least 15 distinct finite target-coordinate points for five free parameters.")
    order=np.argsort(x); x,y=x[order],y[order]
    span=float(np.ptp(x))
    if span<=0: raise ValueError("Target coordinate must span a nonzero interval.")
    baseline=float(np.percentile(y,10)); peak=float(np.max(y))
    if peak<=baseline: raise ValueError("Heat-flux profile must contain a positive signal above its baseline.")
    def model(s,q0,lam,S,s0,bg):
        u=(s-s0)/S
        a=S/(2*lam)
        arg=a-u
        core=np.empty_like(u,dtype=float)
        positive=arg>=0
        # For arg>=0 use erfcx; for arg<0 use the equivalent erfc form whose
        # exponential factor is bounded because u>a in this branch.
        core[positive]=np.exp(-u[positive]*u[positive])*erfcx(arg[positive])
        core[~positive]=np.exp(a*a-2*a*u[~positive])*erfc(arg[~positive])
        return 0.5*q0*core+bg
    p0=[max(peak-baseline,1e-12),max(span/30,span/10000),max(span/50,span/10000),float(x[np.argmax(y)]),baseline]
    lower=[0.0,span/10000,span/10000,float(x[0]-span),-np.inf]
    upper=[np.inf,span*100,span*10,float(x[-1]+span),np.inf]
    try:
        p,cov=optimize.curve_fit(model,x,y,p0=p0,bounds=(lower,upper),maxfev=100000)
    except (RuntimeError,ValueError,FloatingPointError) as exc:
        raise ValueError(f"Eich-like heat-flux fit failed to converge: {exc}") from exc
    if cov.shape!=(5,5) or not np.all(np.isfinite(cov)):
        raise ValueError("Eich fit covariance is singular or non-finite; lambda_q and spreading S may be unidentifiable.")
    pred=model(x,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
    near_bounds=[bool(np.isclose(p[i],lower[i],rtol=1e-3,atol=0) or (np.isfinite(upper[i]) and np.isclose(p[i],upper[i],rtol=1e-3,atol=0))) for i in range(5)]
    return {
        "q0_W_m-2":float(p[0]),"lambda_q_m":float(p[1]),"spreading_S_m":float(p[2]),
        "strike_point_s0_m":float(p[3]),"background_W_m-2":float(p[4]),
        "standard_error_q0_W_m-2":float(se[0]),"standard_error_lambda_q_m":float(se[1]),
        "standard_error_spreading_S_m":float(se[2]),"standard_error_strike_point_s0_m":float(se[3]),
        "standard_error_background_W_m-2":float(se[4]),"r_squared":_r2(y,pred),"n_points":int(len(x)),
        "parameters_at_bounds":{"q0":near_bounds[0],"lambda_q":near_bounds[1],"S":near_bounds[2],"s0":near_bounds[3],"background":near_bounds[4]},
        "lambda_q_S_identifiability_warning":bool(near_bounds[1] or near_bounds[2] or abs(float(cov[1,2]))/max(float(np.sqrt(abs(cov[1,1]*cov[2,2]))),np.finfo(float).tiny)>0.98),
        "model":"Eich-like exponential-convolved-with-Gaussian profile",
        "note":"Parameter identifiability depends on target coverage, noise, spatial mapping and instrument response; inspect covariance and fit residuals before physical interpretation."
    }


def integrate_profile(coordinate: Any, profile: Any) -> dict[str, float]:
    x=np.asarray(coordinate,float).ravel(); y=np.asarray(profile,float).ravel()
    if len(x) != len(y): raise ValueError('Coordinates and profile values must have equal lengths.')
    m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<2: raise ValueError('At least two finite coordinate-profile points are required.')
    o=np.argsort(x); x,y=x[o],y[o]
    if np.any(np.diff(x)<=0): raise ValueError('Coordinate values must be unique after sorting.')
    return {'integral_trapezoid':float(integrate.trapezoid(y,x)),'coordinate_min':float(x[0]),'coordinate_max':float(x[-1]),'profile_max':float(np.max(y)),'profile_min':float(np.min(y))}

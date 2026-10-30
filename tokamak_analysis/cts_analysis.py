"""Scoped frequency-domain microwave CTS inference (unmagnetized electrostatic model)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import wofz

E=1.602176634e-19
EPS0=8.8541878128e-12
KB=1.380649e-23
ME=9.1093837015e-31
AMU=1.66053906660e-27
C=299792458.0


def _array(value,name,ndim=None):
    a=np.asarray(value)
    if a.dtype.kind not in "iuf" or (ndim is not None and a.ndim!=ndim):
        raise ValueError(f"{name} must contain real numeric values")
    a=a.astype(float,copy=False)
    if not a.size or not np.all(np.isfinite(a)): raise ValueError(f"{name} must be finite and non-empty")
    return a


def _positive(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)) or not np.isfinite(value) or value<=0:
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _geometry(g,f0):
    if not isinstance(g,Mapping): raise ValueError("scattering_geometry must be a mapping")
    if "incident_direction" in g and "scattered_direction" in g:
        a=_array(g["incident_direction"],"incident_direction",1); b=_array(g["scattered_direction"],"scattered_direction",1)
        if a.shape!=(3,) or b.shape!=(3,) or np.linalg.norm(a)==0 or np.linalg.norm(b)==0: raise ValueError("beam directions must be nonzero 3-vectors")
        a=a/np.linalg.norm(a); b=b/np.linalg.norm(b); angle=float(np.arccos(np.clip(a@b,-1,1)))
    else:
        angle=_positive(g.get("scattering_angle_rad"),"scattering_angle_rad")
        if angle>np.pi: raise ValueError("scattering_angle_rad must lie in (0, pi]")
        a=np.array([1.,0.,0.]); b=np.array([np.cos(angle),np.sin(angle),0.])
    if "scattering_angle_rad" in g and not np.isclose(float(g["scattering_angle_rad"]),angle,rtol=1e-5,atol=1e-7):
        raise ValueError("scattering angle conflicts with beam direction vectors")
    qv=2*np.pi*f0/C*(b-a); q=float(np.linalg.norm(qv))
    if not np.isfinite(q) or q<=0: raise ValueError("scattering wave number is zero or invalid")
    return q,qv/q,angle


def _species(species,fractions):
    if not isinstance(species,Sequence) or isinstance(species,(str,bytes)) or not species: raise ValueError("ion_species must be a non-empty sequence")
    frac=_array(fractions,"ion_fractions",1)
    if frac.size!=len(species) or np.any(frac<=0) or not np.isclose(frac.sum(),1.,rtol=1e-6,atol=1e-8):
        raise ValueError("ion_fractions must be positive charge-density fractions summing to one")
    out=[]
    for i,s in enumerate(species):
        if not isinstance(s,Mapping): raise ValueError("each ion species must be a mapping")
        out.append({"name":str(s.get("name",f"ion_{i}")),"Z":_positive(s.get("charge_state"),"charge_state"),
                    "mass":_positive(s.get("mass_amu"),"mass_amu")*AMU,"fraction":float(frac[i])})
    return out


def _component(omega,k,n,T,m,Z,u):
    if n<=0: return np.zeros_like(omega,dtype=complex),np.zeros_like(omega)
    vt=np.sqrt(2*T*E/m); z=(omega/k-u)/vt
    lam2=EPS0*T*E/(n*(Z*E)**2)
    plasma_z=1j*np.sqrt(np.pi)*wofz(z)
    chi=(1/(k*k*lam2))*(1+z*plasma_z)
    f=np.exp(-z*z)/(np.sqrt(np.pi)*vt)
    return chi,f


def cts_spectral_density(frequency_shift_Hz,probe_frequency_Hz,scattering_geometry,*,electron_density_m3,
    electron_temperature_eV,ion_temperature_eV,ion_species,ion_fractions,
    ion_velocity_projection_m_s=0.,electron_velocity_projection_m_s=0.,fast_ion=None):
    """Evaluate the longitudinal multi-species S(k,omega); ion fractions are charge-density fractions."""
    f=_array(frequency_shift_Hz,"frequency_shift_Hz",1); f0=_positive(probe_frequency_Hz,"probe_frequency_Hz")
    ne=_positive(electron_density_m3,"electron_density_m3"); te=_positive(electron_temperature_eV,"electron_temperature_eV")
    ti=_positive(ion_temperature_eV,"ion_temperature_eV"); ions=_species(ion_species,ion_fractions)
    ui=float(_array([ion_velocity_projection_m_s],"ion_velocity_projection_m_s")[0]); ue=float(_array([electron_velocity_projection_m_s],"electron_velocity_projection_m_s")[0])
    if abs(ui)>=C or abs(ue)>=C: raise ValueError("flow projections must be subluminal")
    if np.any(f0+f<=0): raise ValueError("scattered frequencies must remain positive")
    k,_,_=_geometry(scattering_geometry,f0); om=2*np.pi*f
    ce,fe=_component(om,k,ne,te,ME,1.,ue); total=ce.copy(); terms=[]
    hot=None
    if fast_ion is not None:
        if not isinstance(fast_ion,Mapping): raise ValueError("fast_ion must be a mapping")
        idx=fast_ion.get("species_index")
        if isinstance(idx,bool) or not isinstance(idx,int) or not 0<=idx<len(ions): raise ValueError("invalid fast_ion.species_index")
        hf=float(fast_ion.get("fraction",0.)); ht=_positive(fast_ion.get("temperature_eV"),"fast_ion.temperature_eV")
        if not 0<=hf<1: raise ValueError("fast-ion fraction must lie in [0,1)")
        hot=(idx,hf,ht)
    for i,s in enumerate(ions):
        ni=s["fraction"]*ne/s["Z"]; hf=hot[1] if hot and hot[0]==i else 0.
        cb,fb=_component(om,k,ni*(1-hf),ti,s["mass"],s["Z"],ui); total+=cb
        if ni*(1-hf)>0: terms.append((s["Z"]**2*ni*(1-hf)/ne,cb,fb))
        if hf:
            ch,fh=_component(om,k,ni*hf,hot[2],s["mass"],s["Z"],ui); total+=ch
            terms.append((s["Z"]**2*ni*hf/ne,ch,fh))
    eps=1+total
    se=np.abs(1-ce/eps)**2*fe; si=np.zeros_like(fe)
    for weight,chi,fi in terms: si+=weight*np.abs(chi/eps)**2*fi
    return (2*np.pi/k)*(se+si)


def _convolve(response,spectrum,axis):
    r=_array(response,"instrument_response",2)
    if r.shape!=(axis.size,axis.size) or np.any(r<0) or np.any(r.sum(axis=0)<=0):
        raise ValueError("instrument_response must be a valid non-negative square matrix on the frequency grid")
    r=r/r.sum(axis=0)[None,:]
    w=np.empty_like(axis); w[1:-1]=.5*(axis[2:]-axis[:-2]); w[0]=.5*(axis[1]-axis[0]); w[-1]=.5*(axis[-1]-axis[-2])
    if np.any(w<=0): raise ValueError("frequency axis must be strictly increasing")
    return (r@(spectrum*w))/w


def analyze_cts_spectrum(frequency_shift_Hz,spectrum,*,probe_frequency_Hz,scattering_geometry,
    ion_species,ion_fractions,instrument_response,signal_uncertainty=None,
    electron_density_initial_m3=1e19,electron_temperature_initial_eV=1000.,ion_temperature_initial_eV=500.,
    ion_velocity_initial_m_s=0.,parameter_bounds=None,electron_velocity_projection_m_s=0.,
    fast_ion=None,global_search=True,random_seed=1729):
    """Infer ne, Te, Ti and ion-flow projection with DE global initialization and bounded least squares."""
    f=_array(frequency_shift_Hz,"frequency_shift_Hz",1); y=_array(spectrum,"spectrum",1)
    if f.size!=y.size or f.size<12 or np.any(np.diff(f)<=0): raise ValueError("frequency and spectrum must match, have >=12 samples and increase strictly")
    if np.any(y<0): raise ValueError("calibrated CTS power spectrum must be non-negative")
    f0=_positive(probe_frequency_Hz,"probe_frequency_Hz"); _geometry(scattering_geometry,f0); ions=_species(ion_species,ion_fractions)
    response=_array(instrument_response,"instrument_response",2)
    if response.shape!=(f.size,f.size) or np.any(response<0) or np.any(response.sum(axis=0)<=0): raise ValueError("invalid instrument_response matrix")
    unc=None if signal_uncertainty is None else _array(signal_uncertainty,"signal_uncertainty",1)
    if unc is not None and (unc.shape!=y.shape or np.any(unc<=0)): raise ValueError("signal_uncertainty must match spectrum and be positive")
    b=parameter_bounds or {}
    def bounds(name,default):
        lo,hi=map(float,b.get(name,default))
        if not np.isfinite(lo+hi) or lo<=0 or lo>=hi: raise ValueError(f"invalid bounds for {name}")
        return lo,hi
    nb=bounds("electron_density_m3",(1e17,1e22)); tb=bounds("electron_temperature_eV",(10.,2e4)); ib=bounds("ion_temperature_eV",(1.,1e4))
    ub=tuple(map(float,b.get("ion_velocity_projection_m_s",(-1e6,1e6))))
    if len(ub)!=2 or not np.isfinite(sum(ub)) or ub[0]>=ub[1] or max(map(abs,ub))>=C: raise ValueError("invalid subluminal ion-flow bounds")
    us=1e5
    p0=np.array([np.log10(_positive(electron_density_initial_m3,"initial density")),np.log10(_positive(electron_temperature_initial_eV,"initial Te")),
        np.log10(_positive(ion_temperature_initial_eV,"initial Ti")),float(ion_velocity_initial_m_s)/us])
    if any(not lo<=10**p0[i]<=hi for i,(lo,hi) in enumerate((nb,tb,ib))) or not ub[0]<=ion_velocity_initial_m_s<=ub[1]:
        raise ValueError("initial values must lie inside parameter_bounds")
    lo=np.array([np.log10(nb[0]),np.log10(tb[0]),np.log10(ib[0]),ub[0]/us]); hi=np.array([np.log10(nb[1]),np.log10(tb[1]),np.log10(ib[1]),ub[1]/us])
    fast=None
    if fast_ion is not None:
        if not isinstance(fast_ion,Mapping): raise ValueError("fast_ion must be a mapping")
        idx=fast_ion.get("species_index")
        if isinstance(idx,bool) or not isinstance(idx,int) or not 0<=idx<len(ions): raise ValueError("invalid fast-ion species_index")
        flo,fhi=map(float,fast_ion.get("fraction_bounds",(.001,.5))); tlo,thi=map(float,fast_ion.get("temperature_bounds_eV",(1e3,1e6)))
        fi=float(fast_ion.get("fraction_initial",.05)); ft=float(fast_ion.get("temperature_initial_eV",max(2e3,3*ion_temperature_initial_eV)))
        if not 0<flo<fi<fhi<1 or not 0<tlo<ft<thi: raise ValueError("invalid fast-ion initial values or bounds")
        fast={"species_index":idx}; p0=np.r_[p0,fi,np.log10(ft)]; lo=np.r_[lo,flo,np.log10(tlo)]; hi=np.r_[hi,fhi,np.log10(thi)]
    noise=unc if unc is not None else np.full_like(y,max(float(np.std(y)),float(np.max(y))*1e-6,np.finfo(float).tiny))
    def decode(p):
        d={"electron_density_m3":10**p[0],"electron_temperature_eV":10**p[1],"ion_temperature_eV":10**p[2],"ion_velocity_projection_m_s":p[3]*us}
        if fast: d["fast_ion"]={"species_index":fast["species_index"],"fraction":p[4],"temperature_eV":10**p[5]}
        return d
    def forward(p):
        d=decode(p); z=cts_spectral_density(f,f0,scattering_geometry,electron_density_m3=d["electron_density_m3"],
          electron_temperature_eV=d["electron_temperature_eV"],ion_temperature_eV=d["ion_temperature_eV"],
          ion_species=ion_species,ion_fractions=ion_fractions,ion_velocity_projection_m_s=d["ion_velocity_projection_m_s"],
          electron_velocity_projection_m_s=electron_velocity_projection_m_s,fast_ion=d.get("fast_ion"))
        return _convolve(response,z,f)
    def residual(p):
        z=forward(p); A=np.column_stack((z,np.ones_like(z)))/noise[:,None]
        amp,off=np.linalg.lstsq(A,y/noise,rcond=None)[0]
        if amp<0: amp,off=0.,float(np.average(y,weights=1/noise**2))
        return (amp*z+off-y)/noise
    score=None
    if global_search:
        de=differential_evolution(lambda p:float(residual(p)@residual(p)),list(zip(lo,hi)),seed=int(random_seed),maxiter=35,popsize=7,polish=False,workers=1,tol=1e-5)
        start=de.x; score=float(de.fun)
    else: start=p0
    fit=least_squares(residual,start,bounds=(lo,hi),x_scale="jac",max_nfev=6000)
    d=decode(fit.x); z=forward(fit.x); A=np.column_stack((z,np.ones_like(z)))/noise[:,None]
    amp,off=np.linalg.lstsq(A,y/noise,rcond=None)[0]
    if amp<0: amp,off=0.,float(np.average(y,weights=1/noise**2))
    fitted=amp*z+off; res=y-fitted; dof=max(1,y.size-len(fit.x)-2)
    rank=int(np.linalg.matrix_rank(fit.jac)); cond=float(np.linalg.cond(fit.jac))
    if not fit.success: raise ValueError(f"CTS refinement failed: {fit.message}")
    if rank<len(fit.x) or not np.isfinite(cond) or cond>1e12: raise ValueError("CTS fit rejected: plasma parameters are not jointly identifiable")
    cov0=np.linalg.pinv(fit.jac.T@fit.jac)
    if unc is None: cov0*=float(np.sum(fit.fun**2)/dof)
    trans=np.eye(len(fit.x)); trans[0,0]=np.log(10)*d["electron_density_m3"]; trans[1,1]=np.log(10)*d["electron_temperature_eV"]; trans[2,2]=np.log(10)*d["ion_temperature_eV"]
    if fast: trans[5,5]=np.log(10)*d["fast_ion"]["temperature_eV"]
    cov=trans@cov0@trans.T; su=np.sqrt(np.maximum(np.diag(cov),0)); q,qhat,angle=_geometry(scattering_geometry,f0)
    alpha=1/(q*np.sqrt(EPS0*d["electron_temperature_eV"]*E/(d["electron_density_m3"]*E**2)))
    names=["electron_density_m3","electron_temperature_eV","ion_temperature_eV","ion_velocity_projection_m_s"]
    out={**d,"electron_density_uncertainty_m3":float(su[0]),"electron_temperature_uncertainty_eV":float(su[1]),
      "ion_temperature_uncertainty_eV":float(su[2]),"ion_velocity_projection_uncertainty_m_s":float(su[3]),
      "parameter_names":names,"fit_covariance":cov,"fitted_spectrum":fitted,"residual":res,
      "reduced_chi_square":float(np.sum((res/unc)**2)/dof) if unc is not None else None,"fit_residual_rms":float(np.sqrt(np.mean(res**2))),
      "optimizer_success":bool(fit.success),"global_search_score":score,"jacobian_rank":rank,"jacobian_condition_number":cond,
      "degrees_of_freedom":int(dof),"scattering_wave_number_m-1":q,"scattering_wavevector_direction":qhat,
      "scattering_angle_rad":angle,"collective_parameter_alpha":float(alpha),"scattering_regime":"collective" if alpha>1 else "weakly_collective_or_noncollective",
      "ion_species":ions,"ion_fractions_charge_density":[float(v) for v in ion_fractions],"probe_frequency_Hz":f0,
      "model":"unmagnetized electrostatic longitudinal microwave CTS",
      "parameter_provenance":{"electron_density":"inferred from measured CTS spectral shape","electron_temperature":"inferred from measured CTS spectral shape",
       "ion_temperature":"inferred from measured ion-feature shape","ion_velocity_projection":"inferred from ion-feature Doppler shift","ion_composition":"externally supplied fixed composition"},
      "uncertainty_model":"Local Jacobian covariance conditional on the selected model, composition, geometry and response; excludes response uncertainty, correlated-bin errors and model discrepancy.",
      "limitations":["Not a full magnetized electromagnetic CTS tensor or ray-tracing model.","Absolute power transfer and polarization coupling are not inverted.",
       "Ion composition is fixed externally.","Optional fast-ion component is a two-Maxwellian approximation, not arbitrary VDF reconstruction.",
       "Model discrepancy, response-matrix uncertainty and correlated frequency-bin errors are not propagated."]}
    if fast:
        names += ["fast_ion_fraction","fast_ion_temperature_eV"]; out["parameter_names"]=names
        out["fast_ion_fraction_uncertainty"]=float(su[4]); out["fast_ion_temperature_uncertainty_eV"]=float(su[5])
    return out

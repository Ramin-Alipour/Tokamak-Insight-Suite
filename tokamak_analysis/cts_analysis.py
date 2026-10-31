"""Scoped frequency-domain microwave CTS inference (unmagnetized electrostatic model)."""
from __future__ import annotations
from collections.abc import Mapping, Sequence
import numpy as np
from scipy.optimize import differential_evolution, least_squares
from scipy.special import wofz
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

E=1.602176634e-19
EPS0=8.8541878128e-12
KB=1.380649e-23
ME=9.1093837015e-31
AMU=1.66053906660e-27
C=299792458.0


def _trapezoid(y, x):
    """NumPy-version-independent trapezoidal integration."""
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:  # NumPy < 2.0
        integrate = np.trapz
    return float(integrate(y, x))


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


def _component_tabulated(omega, k, density, mass, charge, flow, velocity_grid, pdf):
    """Longitudinal susceptibility for a supplied normalized 1-D VDF (Landau contour)."""
    velocity_grid=_array(velocity_grid,"fast-ion velocity grid",1)
    pdf=_array(pdf,"fast-ion probability density",1)
    if velocity_grid.size!=pdf.size or velocity_grid.size<9 or np.any(np.diff(velocity_grid)<=0) or np.any(pdf<0):
        raise ValueError("tabulated fast-ion VDF needs >=9 increasing velocity nodes and non-negative values")
    area=_trapezoid(pdf,velocity_grid)
    if area<=0: raise ValueError("tabulated fast-ion VDF must have positive integral")
    pdf=pdf/area
    interpolator=PchipInterpolator(velocity_grid,pdf,extrapolate=False)
    derivative=interpolator.derivative()
    plasma_frequency_squared=density*(charge*E)**2/(EPS0*mass)
    susceptibility=np.empty(np.asarray(omega).shape,dtype=complex)
    resonant_density=np.empty(np.asarray(omega).shape,dtype=float)
    for index,frequency in np.ndenumerate(np.asarray(omega,dtype=float)):
        resonance=float(frequency/k-flow)
        if resonance<=velocity_grid[0] or resonance>=velocity_grid[-1]:
            if np.isclose(resonance,velocity_grid[0]) or np.isclose(resonance,velocity_grid[-1]):
                raise ValueError("fast-ion velocity grid must extend beyond every resonant velocity")
            pv=quad(lambda v: float(derivative(v))/(v-resonance),
                    float(velocity_grid[0]),float(velocity_grid[-1]),limit=150)[0]
            f_res=0.0
        else:
            pv=quad(lambda v: float(derivative(v)),float(velocity_grid[0]),float(velocity_grid[-1]),
                    weight="cauchy",wvar=resonance,limit=150)[0]
            f_res=max(float(interpolator(resonance)),0.0)
        landau_integral=pv+1j*np.pi*float(derivative(resonance)) if velocity_grid[0]<resonance<velocity_grid[-1] else pv
        susceptibility[index]=-(plasma_frequency_squared/(k*k))*landau_integral
        resonant_density[index]=f_res
    return susceptibility,resonant_density


def cts_spectral_density(frequency_shift_Hz,probe_frequency_Hz,scattering_geometry,*,electron_density_m3,
    electron_temperature_eV,ion_temperature_eV,ion_species,ion_fractions,
    ion_velocity_projection_m_s=0.,electron_velocity_projection_m_s=None,fast_ion=None):
    """Evaluate the longitudinal multi-species S(k,omega); ion fractions are charge-density fractions.

    Electron flow is an explicit external input. Pass 0.0 only when a stationary
    electron-flow assumption is intentionally selected by the caller.
    """
    f=_array(frequency_shift_Hz,"frequency_shift_Hz",1); f0=_positive(probe_frequency_Hz,"probe_frequency_Hz")
    ne=_positive(electron_density_m3,"electron_density_m3"); te=_positive(electron_temperature_eV,"electron_temperature_eV")
    ti=_positive(ion_temperature_eV,"ion_temperature_eV"); ions=_species(ion_species,ion_fractions)
    if electron_velocity_projection_m_s is None:
        raise ValueError("electron_velocity_projection_m_s must be supplied explicitly; use 0.0 only to declare a stationary-electron-flow assumption")
    ui=float(_array([ion_velocity_projection_m_s],"ion_velocity_projection_m_s")[0]); ue=float(_array([electron_velocity_projection_m_s],"electron_velocity_projection_m_s")[0])
    if abs(ui)>=C or abs(ue)>=C: raise ValueError("flow projections must be subluminal")
    if np.any(f0+f<=0): raise ValueError("scattered frequencies must remain positive")
    k,_,_=_geometry(scattering_geometry,f0); om=2*np.pi*f
    ce,fe=_component(om,k,ne,te,ME,1.,ue); total=ce.copy(); terms=[]
    hot=None
    if fast_ion is not None:
        if not isinstance(fast_ion,Mapping): raise ValueError("fast_ion must be a mapping")
        idx=fast_ion.get("species_index")
        if isinstance(idx,bool) or not isinstance(idx,(int,np.integer)) or not 0<=idx<len(ions): raise ValueError("invalid fast_ion.species_index")
        idx=int(idx)
        hf=float(fast_ion.get("fraction",0.))
        if not np.isfinite(hf) or not 0<=hf<1: raise ValueError("fast-ion fraction must lie in [0,1)")
        kind=fast_ion.get("kind","maxwellian")
        if kind not in {"maxwellian","tabulated_1d"}:
            raise ValueError("cts_spectral_density fast_ion.kind must be maxwellian or tabulated_1d")
        if kind=="tabulated_1d":
            vdf=_regularized_tabulated_fast_ion(fast_ion.get("velocity_m_s"),fast_ion.get("probability_density_m_s_inverse"),
                                                smoothness=fast_ion.get("smoothness",0.0))
            hot=(idx,hf,"tabulated",vdf)
        else:
            ht=_positive(fast_ion.get("temperature_eV"),"fast_ion.temperature_eV")
            hot=(idx,hf,"maxwellian",ht)
    for i,s in enumerate(ions):
        ni=s["fraction"]*ne/s["Z"]; hf=hot[1] if hot and hot[0]==i else 0.
        cb,fb=_component(om,k,ni*(1-hf),ti,s["mass"],s["Z"],ui); total+=cb
        if ni*(1-hf)>0: terms.append((s["Z"]**2*ni*(1-hf)/ne,cb,fb))
        if hf:
            if hot[2]=="tabulated":
                ch,fh=_component_tabulated(om,k,ni*hf,s["mass"],s["Z"],ui,
                    hot[3]["velocity_m_s"],hot[3]["probability_density_m_s_inverse"])
            else:
                ch,fh=_component(om,k,ni*hf,hot[3],s["mass"],s["Z"],ui)
            total+=ch
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




def _improved_genetic_search(objective, bounds, *, seed=1729, population_size=32, generations=40):
    """Bounded real-coded GA: tournament selection, blend crossover, elitism and adaptive mutation."""
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    if bounds.ndim != 2 or bounds.shape[1] != 2 or np.any(~np.isfinite(bounds)) or np.any(lower >= upper):
        raise ValueError("genetic-search bounds must be finite increasing pairs")
    dimension = lower.size
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    scores = np.asarray([objective(p) for p in population], dtype=float)
    if np.any(~np.isfinite(scores)):
        scores = np.where(np.isfinite(scores), scores, np.inf)
    best_history = []
    for generation in range(generations):
        order = np.argsort(scores)
        population, scores = population[order], scores[order]
        best_history.append(float(scores[0]))
        elite_count = max(2, population_size // 10)
        next_population = [p.copy() for p in population[:elite_count]]
        progress=generation/max(generations-1,1)
        mutation_scale = 0.18 * (1.0 - progress) + 0.015
        crossover_rate = 0.90 - 0.35*progress
        mutation_rate = min(0.5,(1.0/dimension)*(1.0+2.0*progress))
        def tournament():
            candidates = rng.integers(0, population_size, size=3)
            return population[candidates[np.argmin(scores[candidates])]]
        while len(next_population) < population_size:
            p1, p2 = tournament(), tournament()
            if rng.random()<crossover_rate:
                alpha = rng.uniform(-0.15, 1.15, size=dimension)
                child = alpha * p1 + (1.0 - alpha) * p2
            else:
                child=p1.copy()
            mutate = rng.random(dimension) < mutation_rate
            child += mutate * rng.normal(0.0, mutation_scale, size=dimension) * (upper - lower)
            next_population.append(np.clip(child, lower, upper))
        population = np.asarray(next_population)
        scores = np.asarray([objective(p) for p in population], dtype=float)
        scores = np.where(np.isfinite(scores), scores, np.inf)
    best = int(np.argmin(scores))
    return population[best].copy(), float(scores[best]), np.asarray(best_history, dtype=float)


def _whitener(covariance, n, uncertainty):
    if covariance is not None:
        cov = _array(covariance, "spectral_covariance", 2)
        if cov.shape != (n, n) or not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-12):
            raise ValueError("spectral_covariance must be a symmetric n_frequency x n_frequency matrix")
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError("spectral_covariance must be positive definite") from exc
        return lambda residual: np.linalg.solve(chol, residual), cov
    noise = uncertainty if uncertainty is not None else None
    if noise is None:
        raise ValueError("internal uncertainty scale is required")
    return lambda residual: residual / noise.reshape((-1,) + (1,) * (np.ndim(residual) - 1)), np.diag(noise ** 2)


def _trapezoid_node_weights(axis):
    axis=np.asarray(axis,dtype=float)
    weights=np.empty_like(axis)
    weights[0]=0.5*(axis[1]-axis[0])
    weights[-1]=0.5*(axis[-1]-axis[-2])
    weights[1:-1]=0.5*(axis[2:]-axis[:-2])
    return weights


def _softmax(values):
    z=np.asarray(values,dtype=float)
    z=z-np.max(z)
    e=np.exp(z)
    return e/e.sum()


def _second_derivative_matrix(axis):
    axis=np.asarray(axis,dtype=float)
    if axis.ndim!=1 or axis.size<3 or np.any(np.diff(axis)<=0):
        raise ValueError("second-derivative grid must be strictly increasing with at least three nodes")
    matrix=np.zeros((axis.size-2,axis.size),dtype=float)
    for i in range(1,axis.size-1):
        h0=axis[i]-axis[i-1]
        h1=axis[i+1]-axis[i]
        matrix[i-1,i-1]=2.0/(h0*(h0+h1))
        matrix[i-1,i]=-2.0/(h0*h1)
        matrix[i-1,i+1]=2.0/(h1*(h0+h1))
    return matrix


def _regularized_tabulated_fast_ion(velocity_m_s, pdf_m_s_inverse, *, smoothness=1e-2):
    """Validate and regularize a non-negative tabulated 1-D fast-ion VDF."""
    v = _array(velocity_m_s, "fast-ion velocity grid", 1)
    p = _array(pdf_m_s_inverse, "fast-ion probability density", 1)
    if v.size != p.size or v.size < 7 or np.any(np.diff(v) <= 0) or np.any(p < 0):
        raise ValueError("tabulated fast-ion VDF needs >=7 increasing velocity nodes and non-negative density")
    smoothness = float(smoothness)
    if not np.isfinite(smoothness) or smoothness < 0:
        raise ValueError("fast-ion smoothness must be finite and non-negative")
    area = _trapezoid(p, v)
    if area <= 0:
        raise ValueError("tabulated fast-ion VDF must have positive integral")
    p = p / area
    second_difference = _second_derivative_matrix(v)
    roughness = float(np.linalg.norm(second_difference @ p) ** 2)
    return {"velocity_m_s": v, "probability_density_m_s_inverse": p,
            "normalization_integral": 1.0, "second_difference_roughness": roughness,
            "smoothness_parameter": smoothness,
            "regularization_note": "Input VDF is normalized; roughness is reported, not silently subtracted from the physical distribution."}


def analyze_cts_spectrum(frequency_shift_Hz,spectrum,*,probe_frequency_Hz,scattering_geometry,
    ion_species,ion_fractions,instrument_response,signal_uncertainty=None,
    electron_density_initial_m3=1e19,electron_temperature_initial_eV=1000.,ion_temperature_initial_eV=500.,
    ion_velocity_initial_m_s=0.,parameter_bounds=None,electron_velocity_projection_m_s=None,
    fast_ion=None,global_search=True,random_seed=1729, inversion_method="differential_evolution",
    spectral_covariance=None, instrument_response_uncertainty=None, response_mc_samples=100,
    regularization_strength=0.0, external_parameter_uncertainties=None):
    """Infer CTS plasma parameters with selectable genetic, differential-evolution or regularized inversion."""
    f=_array(frequency_shift_Hz,"frequency_shift_Hz",1); y=_array(spectrum,"spectrum",1)
    if f.size!=y.size or f.size<12 or np.any(np.diff(f)<=0): raise ValueError("frequency and spectrum must match, have >=12 samples and increase strictly")
    if np.any(y<0): raise ValueError("calibrated CTS power spectrum must be non-negative")
    f0=_positive(probe_frequency_Hz,"probe_frequency_Hz"); _geometry(scattering_geometry,f0); ions=_species(ion_species,ion_fractions)
    if electron_velocity_projection_m_s is None:
        raise ValueError("electron_velocity_projection_m_s is required. Enter a measured/provided projection, or explicitly enter 0.0 to declare stationary electron flow.")
    electron_velocity_projection_m_s=float(_array([electron_velocity_projection_m_s],"electron_velocity_projection_m_s")[0])
    if abs(electron_velocity_projection_m_s)>=C:
        raise ValueError("electron_velocity_projection_m_s must be subluminal")
    response=_array(instrument_response,"instrument_response",2)
    if response.shape!=(f.size,f.size) or np.any(response<0) or np.any(response.sum(axis=0)<=0): raise ValueError("invalid instrument_response matrix")
    unc=None if signal_uncertainty is None else _array(signal_uncertainty,"signal_uncertainty",1)
    if unc is not None and (unc.shape!=y.shape or np.any(unc<=0)): raise ValueError("signal_uncertainty must match spectrum and be positive")
    if unc is not None and spectral_covariance is not None:
        raise ValueError("Supply either signal_uncertainty or spectral_covariance, not both")
    if parameter_bounds is not None and not isinstance(parameter_bounds, Mapping):
        raise ValueError("parameter_bounds must be a mapping")
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
        if isinstance(idx,bool) or not isinstance(idx,(int,np.integer)) or not 0<=idx<len(ions): raise ValueError("invalid fast-ion species_index")
        idx=int(idx)
        kind=fast_ion.get("kind","maxwellian")
        if kind not in {"maxwellian","tabulated_1d","infer_tabulated_1d"}:
            raise ValueError("fast_ion.kind must be maxwellian, tabulated_1d or infer_tabulated_1d")
        flo,fhi=map(float,fast_ion.get("fraction_bounds",(.001,.5)))
        fi=float(fast_ion.get("fraction_initial",.05))
        if not 0<flo<fi<fhi<1: raise ValueError("invalid fast-ion fraction initial value or bounds")
        if kind=="tabulated_1d":
            vdf=_regularized_tabulated_fast_ion(fast_ion.get("velocity_m_s"),fast_ion.get("probability_density_m_s_inverse"),
                                                smoothness=fast_ion.get("smoothness",0.0))
            fast={"species_index":idx,"kind":"tabulated_1d","vdf":vdf}
            p0=np.r_[p0,fi]; lo=np.r_[lo,flo]; hi=np.r_[hi,fhi]
        elif kind=="infer_tabulated_1d":
            vdf=_regularized_tabulated_fast_ion(fast_ion.get("velocity_m_s"),fast_ion.get("initial_probability_density_m_s_inverse"),
                                                smoothness=fast_ion.get("smoothness",1.0))
            weights=_trapezoid_node_weights(vdf["velocity_m_s"])
            masses=vdf["probability_density_m_s_inverse"]*weights
            logits=np.clip(np.log(np.maximum(masses[:-1],1e-12)/max(masses[-1],1e-12)),-19.0,19.0)
            if inversion_method!="regularized" or regularization_strength<=0:
                raise ValueError("infer_tabulated_1d fast-ion VDF requires regularized inversion with positive regularization_strength")
            fast={"species_index":idx,"kind":"infer_tabulated_1d","vdf":vdf}
            p0=np.r_[p0,fi,logits]; lo=np.r_[lo,flo,np.full(logits.size,-20.0)]; hi=np.r_[hi,fhi,np.full(logits.size,20.0)]
        else:
            tlo,thi=map(float,fast_ion.get("temperature_bounds_eV",(1e3,1e6)))
            ft=float(fast_ion.get("temperature_initial_eV",max(2e3,3*ion_temperature_initial_eV)))
            if not 0<tlo<ft<thi: raise ValueError("invalid fast-ion temperature initial value or bounds")
            fast={"species_index":idx,"kind":"maxwellian"}; p0=np.r_[p0,fi,np.log10(ft)]; lo=np.r_[lo,flo,np.log10(tlo)]; hi=np.r_[hi,fhi,np.log10(thi)]
    noise=unc if unc is not None else np.full_like(y,max(float(np.std(y)),float(np.max(y))*1e-6,np.finfo(float).tiny))
    if inversion_method not in {"differential_evolution", "improved_genetic", "regularized"}:
        raise ValueError("inversion_method must be differential_evolution, improved_genetic or regularized")
    regularization_strength=float(regularization_strength)
    if not np.isfinite(regularization_strength) or regularization_strength<0:
        raise ValueError("regularization_strength must be finite and non-negative")
    if inversion_method=="regularized" and (not np.isfinite(regularization_strength) or regularization_strength<=0):
        raise ValueError("regularized inversion requires regularization_strength to be finite and > 0")
    if inversion_method=="regularized" and (fast is None or fast["kind"]!="infer_tabulated_1d"):
        raise ValueError("regularized inversion requires fast_ion.kind='infer_tabulated_1d'; initial plasma guesses are not regularization priors")
    if fast and fast["kind"]=="infer_tabulated_1d" and fast["vdf"]["smoothness_parameter"]<=0:
        raise ValueError("infer_tabulated_1d requires a positive VDF smoothness parameter")
    if spectral_covariance is not None:
        cov_y=_array(spectral_covariance,"spectral_covariance",2)
        if cov_y.shape!=(f.size,f.size) or not np.allclose(cov_y,cov_y.T,rtol=1e-9,atol=1e-12):
            raise ValueError("spectral_covariance must be a symmetric n_frequency x n_frequency matrix")
        try: chol=np.linalg.cholesky(cov_y)
        except np.linalg.LinAlgError as exc: raise ValueError("spectral_covariance must be positive definite") from exc
        whiten=lambda z: np.linalg.solve(chol,z)
    else:
        whiten=lambda z: z/noise.reshape((-1,) + (1,) * (np.ndim(z) - 1))
    response_sigma=None if instrument_response_uncertainty is None else _array(instrument_response_uncertainty,"instrument_response_uncertainty",2)
    if response_sigma is not None and (response_sigma.shape!=response.shape or np.any(response_sigma<0)):
        raise ValueError("instrument_response_uncertainty must match the response matrix and be non-negative")
    ext_unc=external_parameter_uncertainties or {}
    if not isinstance(ext_unc,Mapping):
        raise ValueError("external_parameter_uncertainties must be a mapping")
    allowed_external={"probe_frequency_Hz","electron_velocity_projection_m_s","scattering_angle_rad","ion_fractions"}
    if set(ext_unc)-allowed_external:
        raise ValueError(f"unsupported external uncertainty keys: {sorted(set(ext_unc)-allowed_external)}")
    for key,value in ext_unc.items():
        if key=="ion_fractions":
            arr=_array(value,"ion_fractions uncertainty",1)
            if arr.shape!=(len(ions),) or np.any(arr<0):
                raise ValueError("ion_fractions uncertainty must be a non-negative vector matching ion species")
        else:
            _positive(value,key+" uncertainty")
    if "scattering_angle_rad" in ext_unc and ("incident_direction" in scattering_geometry or "scattered_direction" in scattering_geometry):
        raise ValueError("scattering_angle_rad uncertainty requires scalar-angle geometry, not fixed beam-direction vectors")
    if isinstance(response_mc_samples,bool) or not isinstance(response_mc_samples,(int,np.integer)) or response_mc_samples<20:
        raise ValueError("response_mc_samples must be an integer >= 20")
    def decode(p):
        d={"electron_density_m3":10**p[0],"electron_temperature_eV":10**p[1],"ion_temperature_eV":10**p[2],"ion_velocity_projection_m_s":p[3]*us}
        if fast:
            if fast["kind"] in {"tabulated_1d","infer_tabulated_1d"}:
                if fast["kind"]=="infer_tabulated_1d":
                    masses=_softmax(np.r_[p[5:],0.0])
                    weights=_trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
                    pdf=masses/weights
                else:
                    pdf=fast["vdf"]["probability_density_m_s_inverse"]
                d["fast_ion"]={"species_index":fast["species_index"],"fraction":p[4],"kind":"tabulated_1d",
                    "velocity_m_s":fast["vdf"]["velocity_m_s"],
                    "probability_density_m_s_inverse":pdf,
                    "smoothness":fast["vdf"]["smoothness_parameter"]}
            else:
                d["fast_ion"]={"species_index":fast["species_index"],"fraction":p[4],"temperature_eV":10**p[5]}
        return d
    def forward(p, response_matrix=None, probe_frequency=None, electron_flow=None, geometry_override=None, fractions_override=None):
        d=decode(p); z=cts_spectral_density(f,f0 if probe_frequency is None else probe_frequency,
          scattering_geometry if geometry_override is None else geometry_override,electron_density_m3=d["electron_density_m3"],
          electron_temperature_eV=d["electron_temperature_eV"],ion_temperature_eV=d["ion_temperature_eV"],
          ion_species=ion_species,ion_fractions=ion_fractions if fractions_override is None else fractions_override,
          ion_velocity_projection_m_s=d["ion_velocity_projection_m_s"],
          electron_velocity_projection_m_s=electron_velocity_projection_m_s if electron_flow is None else electron_flow,
          fast_ion=d.get("fast_ion"))
        return _convolve(response if response_matrix is None else response_matrix,z,f)
    def profile_amplitude(z):
        A=whiten(np.column_stack((z,np.ones_like(z))))
        amp,off=np.linalg.lstsq(A,whiten(y),rcond=None)[0]
        if amp<0:
            amp=0.0
            whitened_one=whiten(np.ones_like(y))
            whitened_y=whiten(y)
            denominator=float(whitened_one@whitened_one)
            off=float(whitened_one@whitened_y/denominator)
        return amp,off
    def data_residual(p, response_matrix=None, probe_frequency=None, electron_flow=None, geometry_override=None, fractions_override=None):
        z=forward(p,response_matrix,probe_frequency,electron_flow,geometry_override,fractions_override); amp,off=profile_amplitude(z)
        return whiten(amp*z+off-y)
    def regularization_terms(p):
        if inversion_method!="regularized":
            return []
        masses=_softmax(np.r_[p[5:],0.0])
        pdf=masses/_trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
        d2=_second_derivative_matrix(fast["vdf"]["velocity_m_s"])*float(np.ptp(fast["vdf"]["velocity_m_s"]))**2
        scale=max(float(np.max(pdf)),np.finfo(float).tiny)
        penalty=np.sqrt(regularization_strength)*np.sqrt(fast["vdf"]["smoothness_parameter"])*(d2@pdf)/scale
        return [penalty]

    def residual(p):
        data=data_residual(p)
        penalties=regularization_terms(p)
        return np.concatenate([data,*penalties]) if penalties else data
    score=None; genetic_history=None
    if global_search and inversion_method=="improved_genetic":
        start,score,genetic_history=_improved_genetic_search(
            lambda p:float(residual(p)@residual(p)),list(zip(lo,hi)),seed=int(random_seed))
    elif global_search and inversion_method=="differential_evolution":
        de=differential_evolution(lambda p:float(residual(p)@residual(p)),list(zip(lo,hi)),seed=int(random_seed),maxiter=35,popsize=7,polish=False,workers=1,tol=1e-5)
        start=de.x; score=float(de.fun)
    else: start=p0
    fit=least_squares(residual,start,bounds=(lo,hi),x_scale="jac",max_nfev=6000)
    d=decode(fit.x); z=forward(fit.x); amp,off=profile_amplitude(z)
    fitted=amp*z+off; res=y-fitted; data_fit_residual=data_residual(fit.x); dof=max(1,y.size-len(fit.x)-2)
    rank=int(np.linalg.matrix_rank(fit.jac)); cond=float(np.linalg.cond(fit.jac))
    if not fit.success: raise ValueError(f"CTS refinement failed: {fit.message}")
    if rank<len(fit.x) or not np.isfinite(cond) or cond>1e12: raise ValueError("CTS fit rejected: plasma parameters are not jointly identifiable")
    cov0=np.linalg.pinv(fit.jac.T@fit.jac)
    if unc is None and spectral_covariance is None: cov0*=float(data_fit_residual@data_fit_residual/dof)
    trans=np.eye(len(fit.x)); trans[0,0]=np.log(10)*d["electron_density_m3"]; trans[1,1]=np.log(10)*d["electron_temperature_eV"]; trans[2,2]=np.log(10)*d["ion_temperature_eV"]
    if fast and fast["kind"]=="maxwellian": trans[5,5]=np.log(10)*d["fast_ion"]["temperature_eV"]
    cov=trans@cov0@trans.T; su=np.sqrt(np.maximum(np.diag(cov),0)); q,qhat,angle=_geometry(scattering_geometry,f0)
    alpha=1/(q*np.sqrt(EPS0*d["electron_temperature_eV"]*E/(d["electron_density_m3"]*E**2)))
    names=["electron_density_m3","electron_temperature_eV","ion_temperature_eV","ion_velocity_projection_m_s"]
    out={**d,"electron_velocity_projection_m_s":float(electron_velocity_projection_m_s),
      "electron_velocity_projection_is_fitted":False,
      "electron_velocity_projection_input_uncertainty_m_s":(
          float(ext_unc["electron_velocity_projection_m_s"])
          if "electron_velocity_projection_m_s" in ext_unc else None
      ),
      "electron_density_uncertainty_m3":float(su[0]),"electron_temperature_uncertainty_eV":float(su[1]),
      "ion_temperature_uncertainty_eV":float(su[2]),"ion_velocity_projection_uncertainty_m_s":float(su[3]),
      "parameter_names":names,"fit_covariance":cov,"fitted_spectrum":fitted,"residual":res,
      "reduced_chi_square":float(data_fit_residual@data_fit_residual/dof) if (unc is not None or spectral_covariance is not None) else None,"fit_residual_rms":float(np.sqrt(np.mean(res**2))),
      "optimizer_success":bool(fit.success),"global_search_score":score,"genetic_best_score_history":genetic_history,"inversion_method":inversion_method,"regularization_strength":regularization_strength,"regularization_scope":"fast-ion VDF smoothness only; no plasma-parameter prior","jacobian_rank":rank,"jacobian_condition_number":cond,
      "degrees_of_freedom":int(dof),"scattering_wave_number_m-1":q,"scattering_wavevector_direction":qhat,
      "scattering_angle_rad":angle,"collective_parameter_alpha":float(alpha),"scattering_regime":"collective" if alpha>1 else "weakly_collective_or_noncollective",
      "ion_species":ions,"ion_fractions_charge_density":[float(v) for v in ion_fractions],"probe_frequency_Hz":f0,
      "model":"unmagnetized electrostatic longitudinal microwave CTS",
      "parameter_provenance":{"electron_density":"inferred from measured CTS spectral shape","electron_temperature":"inferred from measured CTS spectral shape",
       "ion_temperature":"one common ion temperature inferred from the measured ion feature and shared by all supplied species",
       "ion_velocity_projection":"one common projected ion flow inferred from the measured ion-feature Doppler shift and shared by all supplied species",
       "ion_composition":"externally supplied fixed composition",
       "electron_velocity_projection":"externally supplied fixed scalar projection; no stationary-flow default is imposed"},
      "uncertainty_model":"Local Jacobian covariance with optional full spectral covariance whitening; response-matrix uncertainty is propagated by Monte Carlo when supplied.",
      "limitations":["Not a full magnetized electromagnetic CTS tensor or ray-tracing model.","Absolute power transfer and polarization coupling are not inverted.",
       "Ion composition is fixed externally.",
       "The current Maxwellian background model has one common ion temperature and one common projected ion flow for all species; species-specific Ti and flow inference is not implemented.",
       "Electron flow is a required external scalar projection; enter zero explicitly only when a stationary-electron-flow assumption is intended.",
       "Fast-ion VDF support is one-dimensional; fixed-tabulated mode treats shape as external, while infer_tabulated_1d reconstructs it under grid and regularization constraints.",
       "Model discrepancy remains a separate diagnostic and is not automatically a confidence interval.","Response-matrix uncertainty is propagated by Monte Carlo when supplied."]}
    if out["reduced_chi_square"] is not None:
        discrepancy_scale=max(float(out["reduced_chi_square"])-1.0,0.0)
        discrepancy_covariance=discrepancy_scale*cov
        out["model_discrepancy_scale"]=discrepancy_scale
        out["model_discrepancy_covariance"]=discrepancy_covariance
        out["total_fit_covariance_including_discrepancy"]=cov+discrepancy_covariance
        total_sigma=np.sqrt(np.maximum(np.diag(cov+discrepancy_covariance),0.0))
        out["electron_density_uncertainty_m3"]=float(total_sigma[0])
        out["electron_temperature_uncertainty_eV"]=float(total_sigma[1])
        out["ion_temperature_uncertainty_eV"]=float(total_sigma[2])
        out["ion_velocity_projection_uncertainty_m_s"]=float(total_sigma[3])
        out["model_discrepancy_diagnostics"]={
            "whitened_residual_lag1_correlation":float(np.corrcoef(data_fit_residual[:-1],data_fit_residual[1:])[0,1])
                if data_fit_residual.size>2 and np.std(data_fit_residual[:-1])>0 and np.std(data_fit_residual[1:])>0 else None,
            "interpretation":"Excess reduced chi-square above unity is treated as a model-discrepancy covariance component; it is conditional on supplied noise covariance and is not experimental validation.",
        }
    if fast:
        names += ["fast_ion_fraction"]
        if fast["kind"]=="maxwellian":
            names += ["fast_ion_temperature_eV"]
        elif fast["kind"]=="tabulated_1d":
            out["fast_ion_vdf_provenance"]="externally supplied normalized tabulated 1-D velocity distribution; only its density fraction is inferred"
            out["fast_ion_vdf_regularization"]=fast["vdf"]
        else:
            names += [f"fast_ion_vdf_logit_{i}" for i in range(fast["vdf"]["velocity_m_s"].size-1)]
            out["fast_ion_vdf_provenance"]="non-negative tabulated 1-D fast-ion VDF inferred from the CTS spectrum with a second-difference smoothness penalty"
            out["fast_ion_vdf_velocity_m_s"]=fast["vdf"]["velocity_m_s"]
            out["fast_ion_vdf_probability_density_m_s_inverse"]=d["fast_ion"]["probability_density_m_s_inverse"]
            out["fast_ion_vdf_smoothness_parameter"]=fast["vdf"]["smoothness_parameter"]
        out["parameter_names"]=names
        if fast["kind"]=="maxwellian":
            out["parameter_provenance"]["fast_ion_fraction"]="inferred from CTS spectral shape under a two-Maxwellian population model"
            out["parameter_provenance"]["fast_ion_temperature_eV"]="inferred from CTS spectral shape under a two-Maxwellian population model"
        elif fast["kind"]=="tabulated_1d":
            out["parameter_provenance"]["fast_ion_fraction"]="inferred; tabulated 1-D fast-ion VDF shape supplied externally"
        else:
            out["parameter_provenance"]["fast_ion_fraction"]="inferred jointly with a regularized tabulated 1-D fast-ion VDF shape"
            out["parameter_provenance"]["fast_ion_vdf"]="inferred from CTS spectrum with non-negative softmax representation and second-difference regularization"
        out["fast_ion_fraction_uncertainty"]=float(total_sigma[4] if out.get("model_discrepancy_scale") is not None else su[4])
        if fast["kind"]=="maxwellian":
            out["fast_ion_temperature_uncertainty_eV"]=float(total_sigma[5] if out.get("model_discrepancy_scale") is not None else su[5])
        elif fast["kind"]=="infer_tabulated_1d":
            logits=fit.x[5:]
            def decode_pdf(values):
                return _softmax(np.r_[values,0.0])/_trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
            pdf_jac=np.empty((fast["vdf"]["velocity_m_s"].size,logits.size))
            for j in range(logits.size):
                step=1e-5
                plus=logits.copy(); minus=logits.copy()
                plus[j]+=step; minus[j]-=step
                pdf_jac[:,j]=(decode_pdf(plus)-decode_pdf(minus))/(2.0*step)
            vdf_parameter_covariance=cov[5:,5:]
            if out.get("model_discrepancy_scale") is not None:
                vdf_parameter_covariance=(cov+discrepancy_covariance)[5:,5:]
            vdf_covariance=pdf_jac@vdf_parameter_covariance@pdf_jac.T
            out["fast_ion_vdf_covariance"]=vdf_covariance
            out["fast_ion_vdf_uncertainty_m_s_inverse"]=np.sqrt(np.maximum(np.diag(vdf_covariance),0.0))
    if response_sigma is not None or ext_unc:
        rng=np.random.default_rng(int(random_seed)+991); draws=[]
        for _ in range(int(response_mc_samples)):
            sampled=response
            if response_sigma is not None:
                sampled=np.maximum(response+rng.normal(0.0,response_sigma),0.0)
                sums=sampled.sum(axis=0)
                if np.any(sums<=0): continue
                sampled=sampled/sums[None,:]
            sampled_f0=f0+rng.normal(0.0,float(ext_unc.get("probe_frequency_Hz",0.0)))
            sampled_electron_flow=electron_velocity_projection_m_s+rng.normal(0.0,float(ext_unc.get("electron_velocity_projection_m_s",0.0)))
            sampled_geometry=scattering_geometry
            if "scattering_angle_rad" in ext_unc:
                sampled_geometry=dict(scattering_geometry)
                sampled_geometry["scattering_angle_rad"]=float(scattering_geometry["scattering_angle_rad"])+rng.normal(0.0,float(ext_unc["scattering_angle_rad"]))
                if not 0<sampled_geometry["scattering_angle_rad"]<=np.pi: continue
            sampled_fractions=ion_fractions
            if "ion_fractions" in ext_unc:
                sampled_fractions=np.maximum(np.asarray(ion_fractions,dtype=float)+rng.normal(0.0,np.asarray(ext_unc["ion_fractions"],dtype=float)),1e-12)
                sampled_fractions=sampled_fractions/sampled_fractions.sum()
            if sampled_f0<=0 or abs(sampled_electron_flow)>=C: continue
            try:
                def sampled_residual(p):
                    data=data_residual(p,sampled,sampled_f0,sampled_electron_flow,sampled_geometry,sampled_fractions)
                    penalties=regularization_terms(p)
                    return np.concatenate([data,*penalties]) if penalties else data
                draw=least_squares(sampled_residual,fit.x,bounds=(lo,hi),x_scale="jac",max_nfev=2500)
                if draw.success and np.all(np.isfinite(draw.x)): draws.append(draw.x)
            except (ValueError,RuntimeError,np.linalg.LinAlgError):
                continue
        out["response_external_mc_successes"]=len(draws); out["response_external_mc_requested"]=int(response_mc_samples)
        if len(draws)>=max(12,int(np.ceil(.7*response_mc_samples))):
            response_cov=np.cov(np.asarray(draws),rowvar=False,ddof=1)
            out["response_parameter_covariance"]=trans@response_cov@trans.T
            out["fit_covariance"]=cov+out["response_parameter_covariance"]
            total_covariance=out["fit_covariance"]
            if out.get("model_discrepancy_covariance") is not None:
                total_covariance=total_covariance+out["model_discrepancy_covariance"]
            out["total_fit_covariance_including_discrepancy"]=total_covariance
            total=np.sqrt(np.maximum(np.diag(total_covariance),0.0))
            out["electron_density_uncertainty_m3"]=float(total[0])
            out["electron_temperature_uncertainty_eV"]=float(total[1])
            out["ion_temperature_uncertainty_eV"]=float(total[2])
            out["ion_velocity_projection_uncertainty_m_s"]=float(total[3])
            if fast:
                out["fast_ion_fraction_uncertainty"]=float(total[4])
                if fast["kind"]=="maxwellian":
                    out["fast_ion_temperature_uncertainty_eV"]=float(total[5])
                elif fast["kind"]=="infer_tabulated_1d":
                    response_vdf_covariance=pdf_jac@response_cov[5:,5:]@pdf_jac.T
                    out["fast_ion_vdf_covariance"]=out["fast_ion_vdf_covariance"]+response_vdf_covariance
                    out["fast_ion_vdf_uncertainty_m_s_inverse"]=np.sqrt(np.maximum(np.diag(out["fast_ion_vdf_covariance"]),0.0))
            out["response_uncertainty_propagated"]=response_sigma is not None
            out["external_parameter_uncertainty_propagated"]=bool(ext_unc)
        else:
            out["response_uncertainty_propagated"]=False
            out["external_parameter_uncertainty_propagated"]=False
            out["response_uncertainty_warning"]="Too few successful response/external perturbation refits; no Monte Carlo contribution was added."
    else:
        out["response_uncertainty_propagated"]=False
        out["external_parameter_uncertainty_propagated"]=False
    return out


# Separate magnetized electromagnetic CTS path; the electrostatic API remains unchanged.
from .cts_full_em import full_em_cts_spectral_density, analyze_full_em_cts_spectrum


def analyze_cts_spectrum_model(model=None, **kwargs):
    """Explicit dispatcher for the independent electrostatic and magnetized full-EM CTS inversions."""
    if model is None:
        raise ValueError("Select model='electrostatic' or model='full_electromagnetic' explicitly; no default model is selected.")
    if model == "electrostatic":
        return analyze_cts_spectrum(**kwargs)
    if model == "full_electromagnetic":
        return analyze_full_em_cts_spectrum(**kwargs)
    raise ValueError("model must be 'electrostatic' or 'full_electromagnetic'; no automatic model substitution is performed")

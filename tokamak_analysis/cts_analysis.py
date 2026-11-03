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
    elif isinstance(g.get("frequency_resolved_geometry"), Sequence) and g["frequency_resolved_geometry"]:
        entries=g["frequency_resolved_geometry"]
        entry=entries[len(entries)//2]
        if not isinstance(entry, Mapping) or "incident_direction" not in entry or "scattered_direction" not in entry:
            raise ValueError("frequency_resolved_geometry must contain per-frequency incident_direction and scattered_direction vectors")
        a=_array(entry["incident_direction"],"incident_direction",1); b=_array(entry["scattered_direction"],"scattered_direction",1)
        if a.shape!=(3,) or b.shape!=(3,) or np.linalg.norm(a)==0 or np.linalg.norm(b)==0: raise ValueError("frequency-resolved beam directions must be nonzero 3-vectors")
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



def _maximum_geometry_wave_number(scattering_geometry, probe_frequency_Hz):
    """Return the largest |k| represented by the supplied CTS geometry.

    A frequency-resolved geometry has one scattering vector per spectral bin.
    Use the largest magnitude for conservative Doppler-support bounds; never
    substitute the middle bin's geometry for the complete measured spectrum.
    """
    f0 = _positive(probe_frequency_Hz, "probe_frequency_Hz")
    if isinstance(scattering_geometry, Mapping) and "frequency_resolved_geometry" in scattering_geometry:
        entries = scattering_geometry["frequency_resolved_geometry"]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or not entries:
            raise ValueError("frequency_resolved_geometry must be a non-empty sequence of geometry mappings")
        wave_numbers = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or "frequency_resolved_geometry" in entry:
                raise ValueError(f"frequency_resolved_geometry[{index}] must be a single-frequency geometry mapping")
            wave_numbers.append(_geometry(entry, f0)[0])
        return float(max(wave_numbers))
    return float(_geometry(scattering_geometry, f0)[0])

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
    ion_velocity_projection_m_s=None,electron_velocity_projection_m_s=None,fast_ion=None):
    """Evaluate longitudinal multi-species S(k,omega).

    ion_temperature_eV may be one shared positive scalar or one positive
    temperature per species. ion_fractions are charge-density fractions.
    Electron flow is explicit; pass 0.0 only to select stationary flow.
    """
    f=_array(frequency_shift_Hz,"frequency_shift_Hz",1); f0=_positive(probe_frequency_Hz,"probe_frequency_Hz")
    if isinstance(scattering_geometry, Mapping) and "frequency_resolved_geometry" in scattering_geometry:
        entries=scattering_geometry["frequency_resolved_geometry"]
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)) or len(entries) != f.size:
            raise ValueError("frequency_resolved_geometry must contain exactly one geometry mapping per frequency-shift sample")
        spectra=[]
        for index, entry in enumerate(entries):
            if not isinstance(entry, Mapping) or "frequency_resolved_geometry" in entry:
                raise ValueError(f"frequency_resolved_geometry[{index}] must be a single-frequency geometry mapping")
            spectra.append(float(cts_spectral_density(
                np.asarray([f[index]]), f0, entry,
                electron_density_m3=electron_density_m3,
                electron_temperature_eV=electron_temperature_eV,
                ion_temperature_eV=ion_temperature_eV,
                ion_species=ion_species, ion_fractions=ion_fractions,
                ion_velocity_projection_m_s=ion_velocity_projection_m_s,
                electron_velocity_projection_m_s=electron_velocity_projection_m_s,
                fast_ion=fast_ion,
            )[0]))
        return np.asarray(spectra, dtype=float)
    ne=_positive(electron_density_m3,"electron_density_m3"); te=_positive(electron_temperature_eV,"electron_temperature_eV")
    ions=_species(ion_species,ion_fractions)
    ti_raw=np.asarray(ion_temperature_eV)
    if ti_raw.dtype.kind not in "iuf" or ti_raw.ndim>1:
        raise ValueError("ion_temperature_eV must be a positive scalar or one positive value per ion species")
    if ti_raw.ndim==0:
        ti_values=np.full(len(ions),_positive(float(ti_raw),"ion_temperature_eV"),dtype=float)
    else:
        ti_values=_array(ti_raw,"ion_temperature_eV",1)
        if ti_values.shape!=(len(ions),) or np.any(ti_values<=0):
            raise ValueError("ion_temperature_eV vector must contain one positive value per ion species")
    # Preserve scalar-or-species-vector input; each ion term uses its own Ti.
    if electron_velocity_projection_m_s is None:
        raise ValueError("electron_velocity_projection_m_s must be supplied explicitly; use 0.0 only to declare a stationary-electron-flow assumption")
    if ion_velocity_projection_m_s is None:
        raise ValueError("ion_velocity_projection_m_s must be supplied explicitly; use 0.0 only to declare stationary ion flow")
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
        if "fraction" not in fast_ion:
            raise ValueError("fast_ion.fraction must be supplied explicitly when a fast-ion population is selected")
        hf=float(fast_ion["fraction"])
        if not np.isfinite(hf) or not 0<=hf<1: raise ValueError("fast-ion fraction must lie in [0,1)")
        kind=fast_ion.get("kind")
        if kind not in {"maxwellian","tabulated_1d"}:
            raise ValueError("cts_spectral_density requires explicit fast_ion.kind='maxwellian' or 'tabulated_1d'")
        if kind=="tabulated_1d":
            vdf=_regularized_tabulated_fast_ion(fast_ion.get("velocity_m_s"),fast_ion.get("probability_density_m_s_inverse"),
                                                smoothness=fast_ion.get("smoothness",0.0))
            hot=(idx,hf,"tabulated",vdf)
        else:
            ht=_positive(fast_ion.get("temperature_eV"),"fast_ion.temperature_eV")
            hot=(idx,hf,"maxwellian",ht)
    for i,s in enumerate(ions):
        ni=s["fraction"]*ne/s["Z"]; hf=hot[1] if hot and hot[0]==i else 0.
        cb,fb=_component(om,k,ni*(1-hf),float(ti_values[i]),s["mass"],s["Z"],ui); total+=cb
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
    if not np.allclose(r.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument_response columns must already sum to one; the calibrated response operator is not renormalized")
    edges=np.empty(axis.size+1,dtype=float)
    edges[1:-1]=.5*(axis[:-1]+axis[1:])
    edges[0]=axis[0]-.5*(axis[1]-axis[0])
    edges[-1]=axis[-1]+.5*(axis[-1]-axis[-2])
    w=np.diff(edges)
    if np.any(w<=0): raise ValueError("frequency axis must be strictly increasing")
    return (r@(spectrum*w))/w




def _improved_genetic_search(objective, bounds, *, seed=1729, population_size=32, generations=40):
    """Bounded real-coded GA with elitism and adaptive crossover/mutation.

    Returns the best parameter vector, its objective value, and a monotone
    best-so-far history containing the initial and final populations.
    """
    bounds = np.asarray(bounds, dtype=float)
    if bounds.ndim != 2 or bounds.shape[1] != 2 or bounds.shape[0] < 1:
        raise ValueError("genetic-search bounds must have shape (n_parameters, 2)")
    lower, upper = bounds[:, 0], bounds[:, 1]
    if np.any(~np.isfinite(bounds)) or np.any(lower >= upper):
        raise ValueError("genetic-search bounds must be finite increasing pairs")
    if isinstance(population_size, (bool, np.bool_)) or not isinstance(population_size, (int, np.integer)) or population_size < 4:
        raise ValueError("genetic-search population_size must be an integer >= 4")
    if isinstance(generations, (bool, np.bool_)) or not isinstance(generations, (int, np.integer)) or generations < 1:
        raise ValueError("genetic-search generations must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("genetic-search seed must be an integer")

    rng = np.random.default_rng(int(seed))
    dimension = lower.size
    population = rng.uniform(lower, upper, size=(int(population_size), dimension))

    def evaluate(candidates):
        values = np.asarray([objective(candidate) for candidate in candidates], dtype=float)
        if values.shape != (len(candidates),):
            raise ValueError("genetic-search objective must return one scalar per candidate")
        values = np.where(np.isfinite(values), values, np.inf)
        if not np.any(np.isfinite(values)):
            raise ValueError("genetic-search objective is non-finite for the entire population")
        return values

    scores = evaluate(population)
    best_history = [float(np.min(scores))]
    for generation in range(int(generations)):
        order = np.argsort(scores)
        population, scores = population[order], scores[order]
        elite_count = max(2, int(population_size) // 10)
        next_population = [candidate.copy() for candidate in population[:elite_count]]
        progress = generation / max(int(generations) - 1, 1)
        mutation_scale = 0.18 * (1.0 - progress) + 0.015
        crossover_rate = 0.90 - 0.35 * progress
        mutation_rate = min(0.5, (1.0 / dimension) * (1.0 + 2.0 * progress))

        def tournament():
            candidates = rng.integers(0, int(population_size), size=3)
            return population[candidates[np.argmin(scores[candidates])]]

        while len(next_population) < int(population_size):
            parent_a, parent_b = tournament(), tournament()
            if rng.random() < crossover_rate:
                blend = rng.uniform(-0.15, 1.15, size=dimension)
                child = blend * parent_a + (1.0 - blend) * parent_b
            else:
                child = parent_a.copy()
            mutate = rng.random(dimension) < mutation_rate
            child += mutate * rng.normal(0.0, mutation_scale, size=dimension) * (upper - lower)
            next_population.append(np.clip(child, lower, upper))

        population = np.asarray(next_population, dtype=float)
        scores = evaluate(population)
        best_history.append(float(min(best_history[-1], np.min(scores))))

    best = int(np.argmin(scores))
    return population[best].copy(), float(scores[best]), np.asarray(best_history, dtype=float)


def _whitener(covariance, n, uncertainty):
    if covariance is not None:
        cov = _array(covariance, "spectral_covariance", 2)
        if cov.shape != (n, n):
            raise ValueError("spectral_covariance must be a symmetric n_frequency x n_frequency matrix")
        covariance_scale = float(np.max(np.abs(cov)))
        asymmetry = float(np.max(np.abs(cov - cov.T)))
        if asymmetry > 1e-9 * covariance_scale:
            raise ValueError("spectral_covariance must be symmetric to a scale-relative tolerance")
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



def estimate_cts_optimizer_initial_parameters(
    frequency_shift_Hz, spectrum, *, probe_frequency_Hz, scattering_geometry,
    ion_species, ion_fractions,
):
    """Create numerical CTS optimizer seeds from the calibrated spectral shape.

    These are starting values only. Density uses alpha=1 as a dimensionless
    numerical seed; Te uses the full-spectrum RMS width and Ti uses the
    central 68%-weight core width. No seed is reported as a measurement.
    """
    frequency = _array(frequency_shift_Hz, "frequency_shift_Hz", 1)
    power = _array(spectrum, "spectrum", 1)
    if frequency.size != power.size or frequency.size < 12 or np.any(np.diff(frequency) <= 0):
        raise ValueError("CTS seed estimation requires matching, increasing frequency and spectrum arrays with >=12 samples")
    if np.any(power < 0):
        raise ValueError("calibrated CTS spectrum must be non-negative")
    baseline = float(np.quantile(power, 0.05))
    weights = np.maximum(power - baseline, 0.0)
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= np.finfo(float).tiny:
        raise ValueError("CTS spectral-shape seed estimation failed: spectrum has no positive contrast above its baseline")
    centroid = float(np.sum(weights * frequency) / total)
    offsets = frequency - centroid
    variance = float(np.sum(weights * offsets**2) / total)
    if not np.isfinite(variance) or variance <= 0:
        raise ValueError("CTS spectral-shape seed estimation failed: spectrum has no resolvable frequency width")
    if isinstance(scattering_geometry, Mapping) and "frequency_resolved_geometry" in scattering_geometry:
        raise ValueError(
            "Automatic width-based optimizer seeds require one fixed scattering vector. "
            "For frequency_resolved_geometry, supply electron_density_initial_m3, "
            "electron_temperature_initial_eV and ion_temperature_initial_eV explicitly."
        )
    q, _, _ = _geometry(scattering_geometry, _positive(probe_frequency_Hz, "probe_frequency_Hz"))
    electron_velocity_rms = 2.0 * np.pi * np.sqrt(variance) / q
    te_seed = ME * electron_velocity_rms**2 / E
    # The central 68% of spectral weight supplies a robust ion-feature width.
    order = np.argsort(np.abs(offsets))
    cumulative = np.cumsum(weights[order]) / total
    selected = order[cumulative <= 0.68]
    if selected.size < 3:
        selected = order[:min(max(3, int(np.ceil(0.25 * frequency.size))), frequency.size)]
    core_weights = weights[selected]
    core_total = float(np.sum(core_weights))
    core_variance = float(np.sum(core_weights * offsets[selected]**2) / core_total)
    ions = _species(ion_species, ion_fractions)
    ion_spectral_weights = np.asarray([item["Z"] * item["fraction"] for item in ions], dtype=float)
    effective_ion_mass = float(np.sum(ion_spectral_weights * np.asarray([item["mass"] for item in ions])) / np.sum(ion_spectral_weights))
    ion_velocity_rms = 2.0 * np.pi * np.sqrt(max(core_variance, variance * 1e-6)) / q
    ti_seed = effective_ion_mass * ion_velocity_rms**2 / E
    # alpha=1 is an explicitly labelled dimensionless optimizer seed, not an
    # inferred density. The fit subsequently determines density from the full spectrum.
    density_seed = EPS0 * (te_seed * E) * q**2 / E**2
    floor = np.finfo(float).tiny ** 0.25
    values = {
        "electron_density_m3": max(float(density_seed), floor),
        "electron_temperature_eV": max(float(te_seed), floor),
        "ion_temperature_eV": max(float(ti_seed), floor),
    }
    if not all(np.isfinite(value) and value > 0 for value in values.values()):
        raise ValueError("CTS spectral-shape seed estimation produced non-finite or non-positive optimizer seeds")
    return {
        **values,
        "method": "calibrated_spectrum_RMS_and_central_68_percent_width; alpha=1 density seed",
        "role": "optimizer_initial_estimates_only",
        "density_seed_assumption": "dimensionless collective parameter alpha=1 used only to construct a numerical starting value",
        "spectral_centroid_Hz": centroid,
        "spectral_rms_width_Hz": float(np.sqrt(variance)),
        "spectral_core_rms_width_Hz": float(np.sqrt(core_variance)),
    }


def analyze_cts_spectrum(frequency_shift_Hz,spectrum,*,probe_frequency_Hz,scattering_geometry,
    ion_species,ion_fractions,instrument_response,signal_uncertainty=None,
    electron_density_initial_m3=None,electron_temperature_initial_eV=None,ion_temperature_initial_eV=None,
    ion_velocity_initial_m_s=None,parameter_bounds=None,electron_velocity_projection_m_s=None,
    fast_ion=None,global_search=False,random_seed=1729, inversion_method=None,
    spectral_covariance=None, instrument_response_uncertainty=None, response_mc_samples=100,
    regularization_strength=0.0, external_parameter_uncertainties=None, external_parameter_covariance=None, external_parameter_order=None,
    background_reference_spectra=None, background_components=None,
    background_reference_basis_mc_samples=200, background_reference_basis_mc_seed=1729,
    model_discrepancy_covariance=None):
    """Infer CTS plasma parameters, including electron flow fitted from the spectrum.
    electron_velocity_projection_m_s is an optional optimizer seed only, never fixed.
    Omission uses zero solely as a numerical seed, not a stationary-flow assumption.
    """
    f=_array(frequency_shift_Hz,"frequency_shift_Hz",1); y=_array(spectrum,"spectrum",1)
    if f.size!=y.size or f.size<12 or np.any(np.diff(f)<=0): raise ValueError("frequency and spectrum must match, have >=12 samples and increase strictly")
    if np.any(y<0): raise ValueError("calibrated CTS power spectrum must be non-negative")
    f0=_positive(probe_frequency_Hz,"probe_frequency_Hz"); _geometry(scattering_geometry,f0); ions=_species(ion_species,ion_fractions)
    electron_flow_seed = 0.0 if electron_velocity_projection_m_s is None else float(_array([electron_velocity_projection_m_s], "electron-flow optimizer seed")[0])
    if abs(electron_flow_seed) >= C:
        raise ValueError("electron-flow optimizer seed must be subluminal")
    response=_array(instrument_response,"instrument_response",2)
    if response.shape!=(f.size,f.size) or np.any(response<0) or np.any(response.sum(axis=0)<=0): raise ValueError("invalid instrument_response matrix")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError("measured instrument_response columns must already sum to one; the calibrated response operator is not renormalized")
    # Optional calibrated off-pulse/reference spectra define a fixed PCA
    # nuisance subspace. References are already on the same calibrated
    # frequency grid and detector scale; this routine never re-calibrates them.
    background_basis = None
    background_reference_mean = None
    background_pca_components_used = 0
    if background_reference_spectra is not None:
        reference = _array(background_reference_spectra, "background_reference_spectra", 2)
        if reference.shape[1] != f.size or reference.shape[0] < 2:
            raise ValueError("background_reference_spectra must contain at least two calibrated reference spectra on the same frequency grid")
        if np.any(reference < 0):
            raise ValueError("calibrated background reference spectra must be non-negative")
        if isinstance(background_components, (bool, np.bool_)) or not isinstance(background_components, (int, np.integer)):
            raise ValueError("background_components must be an explicit positive integer when background_reference_spectra is supplied")
        if not 1 <= int(background_components) <= min(reference.shape[0]-1, f.size):
            raise ValueError("background_components must be between 1 and min(n_reference_spectra-1, n_frequency)")
        background_reference_mean = np.mean(reference, axis=0)
        centered_reference = reference - background_reference_mean
        _, singular_values, right_vectors = np.linalg.svd(centered_reference, full_matrices=False)
        numerical_rank = int(np.count_nonzero(singular_values > np.finfo(float).eps * max(centered_reference.shape) * (singular_values[0] if singular_values.size else 0.0)))
        if int(background_components) > numerical_rank:
            raise ValueError(f"background_components={int(background_components)} exceeds the numerical rank {numerical_rank} of the supplied reference spectra")
        background_pca_components_used = int(background_components)
        background_basis = np.column_stack((background_reference_mean, right_vectors[:background_pca_components_used].T))
    unc=None if signal_uncertainty is None else _array(signal_uncertainty,"signal_uncertainty",1)
    if unc is not None and (unc.shape!=y.shape or np.any(unc<=0)): raise ValueError("signal_uncertainty must match spectrum and be positive")
    if unc is not None and spectral_covariance is not None:
        raise ValueError("Supply either signal_uncertainty or spectral_covariance, not both")
    if parameter_bounds is not None and not isinstance(parameter_bounds, Mapping):
        raise ValueError("parameter_bounds must be a mapping")
    b=parameter_bounds or {}
    allowed_bound_names={"electron_density_m3","electron_temperature_eV","ion_temperature_eV","electron_velocity_projection_m_s","ion_velocity_projection_m_s"} | {f"ion_temperature_eV_{i}" for i in range(len(ions))}
    unknown_bound_names=set(b)-allowed_bound_names
    if unknown_bound_names:
        raise ValueError(f"parameter_bounds contains unknown parameter names: {sorted(unknown_bound_names)}")
    def positive_bounds(name):
        if name not in b:
            return None
        pair=np.asarray(b[name],dtype=float)
        if pair.shape!=(2,) or np.any(~np.isfinite(pair)) or np.any(pair<=0) or pair[0]>=pair[1]:
            raise ValueError(f"parameter_bounds[{name!r}] must be a positive finite increasing pair")
        return float(pair[0]),float(pair[1])
    nb=positive_bounds("electron_density_m3")
    tb=positive_bounds("electron_temperature_eV")
    ib=positive_bounds("ion_temperature_eV")
    species_bound_keys = [f"ion_temperature_eV_{i}" for i in range(len(ions))]
    if ib is not None and any(key in b for key in species_bound_keys):
        raise ValueError("Use either the shared ion_temperature_eV bounds or species-specific ion_temperature_eV_i bounds, not both.")
    ib_species = [
        positive_bounds(key) if key in b else ib
        for key in species_bound_keys
    ]
    physical_flow_limit=float(np.nextafter(C,0.0))
    q_fit = _maximum_geometry_wave_number(scattering_geometry, f0)
    spectral_flow_limit = min(physical_flow_limit, 2.0*np.pi*float(np.max(np.abs(f)))/q_fit)
    eub=tuple(map(float,b.get("electron_velocity_projection_m_s",(-spectral_flow_limit,spectral_flow_limit))))
    if len(eub)!=2 or not np.all(np.isfinite(eub)) or eub[0]>=eub[1] or max(map(abs,eub))>=C:
        raise ValueError("electron-flow bounds must be finite, increasing and subluminal")
    if electron_velocity_projection_m_s is None:
        electron_flow_seed=float(np.clip(0.0,eub[0],eub[1]))
    elif not eub[0]<=electron_flow_seed<=eub[1]:
        raise ValueError("electron-flow optimizer seed must lie inside its supplied parameter_bounds")
    ub=tuple(map(float,b.get("ion_velocity_projection_m_s",(-physical_flow_limit,physical_flow_limit))))
    if len(ub)!=2 or not np.all(np.isfinite(ub)) or ub[0]>=ub[1] or max(map(abs,ub))>=C:
        raise ValueError("ion-flow bounds must be finite, increasing and within the physical subluminal domain")
    automatic_seeds = estimate_cts_optimizer_initial_parameters(
        f, y, probe_frequency_Hz=f0, scattering_geometry=scattering_geometry,
        ion_species=ion_species, ion_fractions=ion_fractions,
    ) if any(value is None for value in (
        electron_density_initial_m3, electron_temperature_initial_eV, ion_temperature_initial_eV
    )) else None
    us=1e6
    initial_density=_positive(electron_density_initial_m3 if electron_density_initial_m3 is not None else automatic_seeds["electron_density_m3"],"initial density")
    initial_te=_positive(electron_temperature_initial_eV if electron_temperature_initial_eV is not None else automatic_seeds["electron_temperature_eV"],"initial Te")
    initial_ti_raw = (automatic_seeds["ion_temperature_eV"] if ion_temperature_initial_eV is None else ion_temperature_initial_eV)
    initial_ti_raw_array = np.asarray(initial_ti_raw)
    if initial_ti_raw_array.dtype.kind not in ("i", "u", "f"):
        raise ValueError("ion_temperature_initial_eV must contain real numeric values")
    initial_ti = initial_ti_raw_array.astype(float, copy=False)
    if initial_ti.ndim == 0:
        initial_ti = np.full(len(ions), float(initial_ti), dtype=float)
    if initial_ti.shape != (len(ions),) or np.any(~np.isfinite(initial_ti)) or np.any(initial_ti <= 0):
        raise ValueError("ion_temperature_initial_eV must be a positive scalar or one positive initial temperature per supplied ion species")
    ti_values = initial_ti.copy()
    # A zero flow value is only a numerical optimizer seed. The flow remains
    # an unconstrained fitted parameter and is not fixed to a stationary state.
    initial_ion_flow=0.0 if ion_velocity_initial_m_s is None else float(_array([ion_velocity_initial_m_s],"ion_velocity_initial_m_s")[0])
    if abs(initial_ion_flow)>=C:
        raise ValueError("ion_velocity_initial_m_s must be subluminal")
    if automatic_seeds is not None:
        if electron_density_initial_m3 is None and nb is not None:
            initial_density=float(np.clip(initial_density,nb[0],nb[1]))
        if electron_temperature_initial_eV is None and tb is not None:
            initial_te=float(np.clip(initial_te,tb[0],tb[1]))
        if ion_temperature_initial_eV is None:
            for species_index, pair in enumerate(ib_species):
                if pair is not None:
                    initial_ti[species_index] = np.clip(initial_ti[species_index], pair[0], pair[1])
    for name,value,pair in (
        ("electron_density_m3",initial_density,nb),
        ("electron_temperature_eV",initial_te,tb),
        *[(f"ion_temperature_eV_{j}", float(value), ib_species[j]) for j, value in enumerate(initial_ti)],
    ):
        if pair is not None and not pair[0]<=value<=pair[1]:
            raise ValueError(f"initial value for {name} must lie inside its supplied parameter_bounds")
    if not ub[0]<=initial_ion_flow<=ub[1]:
        raise ValueError("initial ion flow must lie inside its supplied parameter_bounds")
    initialization_record = {
        "electron_density_m3": initial_density,
        "electron_temperature_eV": initial_te,
        "ion_temperature_eV": (float(initial_ti[0]) if len(ions) == 1 else initial_ti.tolist()),
        "method": automatic_seeds["method"] if automatic_seeds is not None else "researcher_supplied",
        "automatically_estimated_parameters": [
            name for name, value in (
                ("electron_density_m3", electron_density_initial_m3),
                ("electron_temperature_eV", electron_temperature_initial_eV),
                ("ion_temperature_eV", ion_temperature_initial_eV),
            ) if value is None
        ],
        "role": "optimizer_initial_estimates_only",
    }
    if automatic_seeds is not None:
        initialization_record["density_seed_assumption"] = automatic_seeds["density_seed_assumption"]
    n_species = len(ions)
    fast_offset = 4 + n_species
    p0=np.r_[np.log10(initial_density), np.log10(initial_te), np.log10(initial_ti),
             electron_flow_seed/us, initial_ion_flow/us]
    # Finite numerical guards prevent 10**p from underflowing to zero or
    # overflowing during unconstrained local optimization. These are machine-
    # representability limits, not scientific plasma-parameter bounds.
    log10_min = float(np.log10(np.finfo(float).tiny) + 2.0)
    log10_max = float(np.log10(np.finfo(float).max) - 2.0)
    lo=np.r_[
        np.log10(nb[0]) if nb is not None else log10_min,
        np.log10(tb[0]) if tb is not None else log10_min,
        np.asarray([np.log10(pair[0]) if pair is not None else log10_min for pair in ib_species], dtype=float),
        eub[0]/us, ub[0]/us,
    ]
    hi=np.r_[
        np.log10(nb[1]) if nb is not None else log10_max,
        np.log10(tb[1]) if tb is not None else log10_max,
        np.asarray([np.log10(pair[1]) if pair is not None else log10_max for pair in ib_species], dtype=float),
        eub[1]/us, ub[1]/us,
    ]
    if not isinstance(global_search,(bool,np.bool_)):
        raise ValueError("global_search must be boolean")
    if global_search and inversion_method in {"differential_evolution","improved_genetic"} and (
        nb is None or tb is None or any(pair is None for pair in ib_species)
        or "electron_velocity_projection_m_s" not in b
        or "ion_velocity_projection_m_s" not in b
    ):
        raise ValueError(
            "Global search requires explicit finite parameter_bounds supplied by the researcher "
            "for density, both temperatures, electron flow and ion flow; "
            "local least-squares inference uses numerical representability guards only."
        )
    fast=None
    if fast_ion is not None:
        if not isinstance(fast_ion,Mapping): raise ValueError("fast_ion must be a mapping")
        idx=fast_ion.get("species_index")
        if isinstance(idx,bool) or not isinstance(idx,(int,np.integer)) or not 0<=idx<len(ions): raise ValueError("invalid fast-ion species_index")
        idx=int(idx)
        kind=fast_ion.get("kind")
        if kind not in {"maxwellian","tabulated_1d","infer_tabulated_1d"}:
            raise ValueError("fast_ion.kind must be explicitly set to maxwellian, tabulated_1d or infer_tabulated_1d")
        flo,fhi=map(float,fast_ion.get("fraction_bounds",(0.0,1.0)))
        if fast_ion.get("fraction_initial") is None:
            raise ValueError("fast_ion.fraction_initial is required; no fast-ion fraction is assumed")
        fi=float(fast_ion["fraction_initial"])
        if not np.isfinite(flo+fhi+fi) or not 0.0<=flo<fi<fhi<=1.0:
            raise ValueError("fast-ion fraction initial value and bounds must satisfy 0 <= lower < initial < upper <= 1")
        if kind=="tabulated_1d":
            vdf=_regularized_tabulated_fast_ion(fast_ion.get("velocity_m_s"),fast_ion.get("probability_density_m_s_inverse"),
                                                smoothness=fast_ion.get("smoothness",0.0))
            fast={"species_index":idx,"kind":"tabulated_1d","vdf":vdf}
            p0=np.r_[p0,fi]; lo=np.r_[lo,flo]; hi=np.r_[hi,fhi]
        elif kind=="infer_tabulated_1d":
            if fast_ion.get("smoothness") is None:
                raise ValueError("fast_ion.smoothness is required for inferred tabulated VDF regularization")
            vdf=_regularized_tabulated_fast_ion(fast_ion.get("velocity_m_s"),fast_ion.get("initial_probability_density_m_s_inverse"),
                                                smoothness=fast_ion["smoothness"])
            weights=_trapezoid_node_weights(vdf["velocity_m_s"])
            masses=vdf["probability_density_m_s_inverse"]*weights
            logits=np.clip(np.log(np.maximum(masses[:-1],1e-12)/max(masses[-1],1e-12)),-19.0,19.0)
            if inversion_method!="regularized" or regularization_strength<=0:
                raise ValueError("infer_tabulated_1d fast-ion VDF requires regularized inversion with positive regularization_strength")
            fast={"species_index":idx,"kind":"infer_tabulated_1d","vdf":vdf}
            p0=np.r_[p0,fi,logits]; lo=np.r_[lo,flo,np.full(logits.size,-20.0)]; hi=np.r_[hi,fhi,np.full(logits.size,20.0)]
        else:
            if fast_ion.get("temperature_initial_eV") is None:
                raise ValueError("fast_ion.temperature_initial_eV is required for a Maxwellian fast-ion component")
            ft=_positive(fast_ion["temperature_initial_eV"],"fast_ion.temperature_initial_eV")
            temperature_bounds=fast_ion.get("temperature_bounds_eV")
            if temperature_bounds is None:
                tlo_log,thi_log=-np.inf,np.inf
            else:
                pair=np.asarray(temperature_bounds,dtype=float)
                if pair.shape!=(2,) or np.any(~np.isfinite(pair)) or np.any(pair<=0) or pair[0]>=pair[1] or not pair[0]<=ft<=pair[1]:
                    raise ValueError("fast_ion.temperature_bounds_eV must be a positive increasing pair containing temperature_initial_eV")
                tlo_log,thi_log=np.log10(pair)
            fast={"species_index":idx,"kind":"maxwellian"}
            p0=np.r_[p0,fi,np.log10(ft)]
            lo=np.r_[lo,flo,tlo_log]
            hi=np.r_[hi,fhi,thi_log]
    noise=unc if unc is not None else np.ones_like(y)
    weighting_status="measured_pointwise_uncertainty" if unc is not None else "unweighted_no_measurement_uncertainty_supplied"
    if inversion_method not in {"local_least_squares", "differential_evolution", "improved_genetic", "regularized"}:
        raise ValueError("Select inversion_method explicitly: local_least_squares, differential_evolution, improved_genetic or regularized")
    if inversion_method in {"differential_evolution","improved_genetic"} and not global_search:
        raise ValueError("differential_evolution and improved_genetic require global_search=True")
    if inversion_method in {"local_least_squares","regularized"} and global_search:
        raise ValueError("global_search must be disabled for local_least_squares and regularized inversion")
    regularization_strength=float(regularization_strength)
    if not np.isfinite(regularization_strength) or regularization_strength<0:
        raise ValueError("regularization_strength must be finite and non-negative")
    if inversion_method=="regularized" and (not np.isfinite(regularization_strength) or regularization_strength<=0):
        raise ValueError("regularized inversion requires regularization_strength to be finite and > 0")
    if inversion_method=="regularized" and (fast is None or fast["kind"]!="infer_tabulated_1d"):
        raise ValueError("regularized inversion requires fast_ion.kind='infer_tabulated_1d'; initial plasma guesses are not regularization priors")
    if fast and fast["kind"]=="infer_tabulated_1d" and fast["vdf"]["smoothness_parameter"]<=0:
        raise ValueError("infer_tabulated_1d requires a positive VDF smoothness parameter")
    independent_discrepancy_covariance = None
    effective_covariance_supplied = (spectral_covariance is not None or unc is not None)
    if model_discrepancy_covariance is not None:
        independent_discrepancy_covariance = _array(
            model_discrepancy_covariance, "model_discrepancy_covariance", 2
        )
        if (independent_discrepancy_covariance.shape != (f.size, f.size)
                or not np.allclose(independent_discrepancy_covariance,
                                   independent_discrepancy_covariance.T,
                                   rtol=1e-10, atol=1e-13)):
            raise ValueError("model_discrepancy_covariance must be symmetric and match the CTS frequency grid")
        eig = np.linalg.eigvalsh(independent_discrepancy_covariance)
        scale = max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
        if float(np.min(eig)) < -1e-10 * scale:
            raise ValueError("model_discrepancy_covariance must be positive semidefinite")
        base_covariance = (
            _array(spectral_covariance, "spectral_covariance", 2)
            if spectral_covariance is not None else
            np.diag(unc**2) if unc is not None else
            np.zeros((f.size, f.size), dtype=float)
        )
        if base_covariance.shape != (f.size, f.size) or not np.allclose(base_covariance, base_covariance.T, rtol=1e-9, atol=1e-12):
            raise ValueError("measurement covariance must be symmetric and match the CTS frequency grid")
        effective_covariance = base_covariance + independent_discrepancy_covariance
        try:
            chol = np.linalg.cholesky(effective_covariance)
        except np.linalg.LinAlgError as exc:
            raise ValueError("measurement plus independent model-discrepancy covariance must be positive definite") from exc
        whiten = lambda z: np.linalg.solve(chol, z)
        effective_covariance_supplied = True
        weighting_status = "measurement_covariance_plus_independent_correlated_model_discrepancy"
    elif spectral_covariance is not None:
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
    if external_parameter_covariance is None and external_parameter_order is not None:
        raise ValueError("external_parameter_order is only valid with external_parameter_covariance")
    if external_parameter_covariance is not None and ext_unc:
        raise ValueError("external_parameter_covariance and external_parameter_uncertainties are mutually exclusive")
    external_covariance = None
    external_order = []
    external_root = None
    if external_parameter_covariance is not None:
        if not isinstance(external_parameter_order, (list, tuple)) or not external_parameter_order:
            raise ValueError("external_parameter_order is required with external_parameter_covariance")
        external_order = [str(item) for item in external_parameter_order]
        allowed_covariance_names = {"probe_frequency_Hz", "scattering_angle_rad"} | {
            f"ion_fraction_{i}" for i in range(len(ions))
        } | {f"ion_mass_amu_{i}" for i in range(len(ions))} | {
            f"ion_charge_state_{i}" for i in range(len(ions))
        } | {f"ion_temperature_eV_{i}" for i in range(len(ions))}
        if len(set(external_order)) != len(external_order) or set(external_order)-allowed_covariance_names:
            raise ValueError("external_parameter_order contains duplicate or unsupported names")
        external_covariance = _array(external_parameter_covariance, "external_parameter_covariance", 2)
        if external_covariance.shape != (len(external_order), len(external_order)) or not np.allclose(external_covariance, external_covariance.T, rtol=1e-9, atol=1e-12):
            raise ValueError("external_parameter_covariance must be symmetric and match external_parameter_order")
        eigenvalues, eigenvectors = np.linalg.eigh(external_covariance)
        covariance_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
        if float(np.min(eigenvalues)) < -1e-10*covariance_scale:
            raise ValueError("external_parameter_covariance must be positive semidefinite")
        external_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
        if "scattering_angle_rad" in external_order and ("incident_direction" in scattering_geometry or "scattered_direction" in scattering_geometry):
            raise ValueError("scattering_angle_rad covariance requires scalar-angle geometry, not fixed beam-direction vectors")
    if not isinstance(ext_unc,Mapping):
        raise ValueError("external_parameter_uncertainties must be a mapping")
    allowed_external={"probe_frequency_Hz","scattering_angle_rad","ion_fractions"}
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
        ti = 10.0**p[2:2+n_species]
        d={"electron_density_m3":10**p[0],"electron_temperature_eV":10**p[1],
           "ion_temperature_eV":(float(ti[0]) if n_species == 1 else ti.copy()),
           "electron_velocity_projection_m_s":p[2+n_species]*us,
           "ion_velocity_projection_m_s":p[3+n_species]*us}
        if fast:
            if fast["kind"] in {"tabulated_1d","infer_tabulated_1d"}:
                if fast["kind"]=="infer_tabulated_1d":
                    masses=_softmax(np.r_[p[fast_offset+1:],0.0])
                    weights=_trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
                    pdf=masses/weights
                else:
                    pdf=fast["vdf"]["probability_density_m_s_inverse"]
                d["fast_ion"]={"species_index":fast["species_index"],"fraction":p[fast_offset],"kind":"tabulated_1d",
                    "velocity_m_s":fast["vdf"]["velocity_m_s"],
                    "probability_density_m_s_inverse":pdf,
                    "smoothness":fast["vdf"]["smoothness_parameter"]}
            else:
                d["fast_ion"]={"species_index":fast["species_index"],"fraction":p[fast_offset],"kind":"maxwellian","temperature_eV":10**p[fast_offset+1]}
        return d
    def forward(p, response_matrix=None, probe_frequency=None, electron_flow=None, geometry_override=None, fractions_override=None, species_override=None, temperatures_override=None):
        d=decode(p); z=cts_spectral_density(f,f0 if probe_frequency is None else probe_frequency,
          scattering_geometry if geometry_override is None else geometry_override,electron_density_m3=d["electron_density_m3"],
          electron_temperature_eV=d["electron_temperature_eV"],ion_temperature_eV=d["ion_temperature_eV"] if temperatures_override is None else temperatures_override,
          ion_species=ion_species if species_override is None else species_override,ion_fractions=ion_fractions if fractions_override is None else fractions_override,
          ion_velocity_projection_m_s=d["ion_velocity_projection_m_s"],
          electron_velocity_projection_m_s=d["electron_velocity_projection_m_s"] if electron_flow is None else electron_flow,
          fast_ion=d.get("fast_ion"))
        return _convolve(response if response_matrix is None else response_matrix,z,f)
    def profile_amplitude(z):
        # Linear nuisance parameters are profiled jointly with the nonlinear
        # plasma parameters. Reference-derived background modes remain in
        # detector space; only the physical CTS spectrum is response-convolved.
        nuisance = np.ones((f.size, 1)) if background_basis is None else background_basis
        design = np.column_stack((z, nuisance))
        coefficients = np.linalg.lstsq(whiten(design), whiten(y), rcond=None)[0]
        amp = float(coefficients[0])
        if amp < 0.0:
            amp = 0.0
            coefficients = np.linalg.lstsq(whiten(nuisance), whiten(y), rcond=None)[0]
        background = nuisance @ coefficients
        return amp, background, coefficients
    def data_residual(p, response_matrix=None, probe_frequency=None, electron_flow=None, geometry_override=None, fractions_override=None, species_override=None, temperatures_override=None):
        z=forward(p,response_matrix,probe_frequency,electron_flow,geometry_override,fractions_override,species_override,temperatures_override)
        amp,background,_=profile_amplitude(z)
        return whiten(amp*z+background-y)
    def regularization_terms(p):
        if inversion_method!="regularized":
            return []
        masses=_softmax(np.r_[p[fast_offset+1:],0.0])
        pdf=masses/_trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
        d2=_second_derivative_matrix(fast["vdf"]["velocity_m_s"])*float(np.ptp(fast["vdf"]["velocity_m_s"]))**2
        scale=max(float(np.max(pdf)),np.finfo(float).tiny)
        strength = float(regularization_strength) * float(fast["vdf"]["smoothness_parameter"])
        curvature_penalty = np.sqrt(strength) * (d2 @ pdf) / scale
        # The second-derivative operator has an affine null space. A weak,
        # explicit reference-shape Tikhonov term on the supplied initial VDF
        # closes that null space without imposing a prior on plasma parameters.
        reference_masses = (
            fast["vdf"]["probability_density_m_s_inverse"]
            * _trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
        )
        reference_masses = reference_masses / np.sum(reference_masses)
        initial_logits = np.clip(
            np.log(np.maximum(reference_masses[:-1], 1e-12)
                   / max(float(reference_masses[-1]), 1e-12)),
            -19.0, 19.0,
        )
        reference_penalty = np.sqrt(strength) * (p[fast_offset+1:] - initial_logits)
        return [curvature_penalty, reference_penalty]

    def residual(p):
        data=data_residual(p)
        penalties=regularization_terms(p)
        return np.concatenate([data,*penalties]) if penalties else data
    if global_search and inversion_method in {"differential_evolution","improved_genetic"} and (
        not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi))
    ):
        raise ValueError("Global search requires finite explicit bounds for every fitted parameter, including any fast-ion component; use local least-squares inference when no scientific bounds are supplied.")
    score=None; genetic_history=None
    if global_search and inversion_method=="improved_genetic":
        start,score,genetic_history=_improved_genetic_search(
            lambda p:float(residual(p)@residual(p)),list(zip(lo,hi)),seed=int(random_seed))
    elif global_search and inversion_method=="differential_evolution":
        de=differential_evolution(lambda p:float(residual(p)@residual(p)),list(zip(lo,hi)),seed=int(random_seed),maxiter=35,popsize=7,polish=False,workers=1,tol=1e-5)
        start=de.x; score=float(de.fun)
    else: start=p0
    fit=least_squares(residual,start,bounds=(lo,hi),x_scale="jac",max_nfev=6000)
    d=decode(fit.x); z=forward(fit.x); amp,background_fit,background_coefficients=profile_amplitude(z)
    fitted=amp*z+background_fit; res=y-fitted; data_fit_residual=data_residual(fit.x)
    nuisance = np.ones((f.size, 1)) if background_basis is None else background_basis
    nuisance_parameter_count=nuisance.shape[1]
    dof=max(1,y.size-len(fit.x)-1-nuisance_parameter_count)

    # Conditional covariance of the fitted detector-space background
    # coefficients.  This quantifies uncertainty in the linear nuisance fit
    # while keeping the researcher-supplied PCA reference subspace fixed.
    # It does not claim to propagate uncertainty in the reference ensemble
    # or in the PCA basis itself.
    if amp > 0.0:
        linear_design = np.column_stack((z, nuisance))
        linear_covariance_slice = slice(1, 1 + nuisance_parameter_count)
    else:
        linear_design = nuisance
        linear_covariance_slice = slice(0, nuisance_parameter_count)
    whitened_linear_design = whiten(linear_design)
    linear_normal = whitened_linear_design.T @ whitened_linear_design
    linear_covariance = np.linalg.pinv(linear_normal)
    if not effective_covariance_supplied:
        linear_covariance *= float(data_fit_residual @ data_fit_residual / dof)
    background_coefficient_covariance = linear_covariance[
        linear_covariance_slice, linear_covariance_slice
    ]
    background_spectrum_covariance = (
        nuisance @ background_coefficient_covariance @ nuisance.T
    )
    background_spectrum_uncertainty = np.sqrt(
        np.maximum(np.diag(background_spectrum_covariance), 0.0)
    )
    # Identifiability must be assessed after column scaling.  The optimizer
    # parameters mix logarithms, a dimensional flow projection and (for an
    # inferred VDF) dimensionless logits; a raw condition number is therefore
    # dominated by arbitrary parameter units rather than independent spectral
    # information.
    jacobian = np.asarray(fit.jac, dtype=float)
    column_norms = np.linalg.norm(jacobian, axis=0)
    scaled_jacobian = np.divide(
        jacobian, column_norms[None, :],
        out=np.zeros_like(jacobian),
        where=column_norms[None, :] > np.finfo(float).tiny,
    )
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    rank_tolerance = (
        np.finfo(float).eps * max(scaled_jacobian.shape)
        * (singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    cond = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0 else float("inf")
    )
    if not fit.success: raise ValueError(f"CTS refinement failed: {fit.message}")
    if rank<len(fit.x) or not np.isfinite(cond) or cond>1e12:
        raise ValueError(
            "CTS fit rejected: plasma parameters are not jointly identifiable "
            f"(scaled Jacobian rank {rank}/{len(fit.x)}, condition number {cond:.6g})"
        )
    cov0=np.linalg.pinv(fit.jac.T@fit.jac)
    if not effective_covariance_supplied: cov0*=float(data_fit_residual@data_fit_residual/dof)
    trans=np.eye(len(fit.x))
    trans[0,0]=np.log(10)*d["electron_density_m3"]
    trans[1,1]=np.log(10)*d["electron_temperature_eV"]
    fitted_ion_temperatures = np.atleast_1d(d["ion_temperature_eV"]).astype(float)
    for species_index, temperature in enumerate(fitted_ion_temperatures):
        trans[2+species_index,2+species_index]=np.log(10)*float(temperature)
    trans[2+n_species,2+n_species]=us
    trans[3+n_species,3+n_species]=us
    if fast and fast["kind"]=="maxwellian":
        trans[fast_offset+1,fast_offset+1]=np.log(10)*d["fast_ion"]["temperature_eV"]
    cov=trans@cov0@trans.T; su=np.sqrt(np.maximum(np.diag(cov),0)); q,qhat,angle=_geometry(scattering_geometry,f0)
    alpha=1/(q*np.sqrt(EPS0*d["electron_temperature_eV"]*E/(d["electron_density_m3"]*E**2)))
    ion_names = ["ion_temperature_eV"] if n_species == 1 else [
        f"ion_temperature_{ions[i]['name']}" if "name" in ions[i] else f"ion_temperature_species_{i}"
        for i in range(n_species)
    ]
    names = ["electron_density_m3", "electron_temperature_eV"] + ion_names + [
        "electron_velocity_projection_m_s", "ion_velocity_projection_m_s"
    ]
    out={**d,"optimizer_initial_estimates":initialization_record,
      "electron_velocity_projection_is_fitted":True,"electron_velocity_projection_uncertainty_m_s":float(su[2+n_species]),
      "electron_density_uncertainty_m3":float(su[0]),"electron_temperature_uncertainty_eV":float(su[1]),
      "ion_temperature_uncertainty_eV":(float(su[2]) if n_species == 1 else su[2:2+n_species].copy()),
      "ion_temperature_species_uncertainties_eV":su[2:2+n_species].copy(),
      "ion_velocity_projection_uncertainty_m_s":float(su[3+n_species]),
      "parameter_names":names,"fit_covariance":cov,"fitted_spectrum":fitted,"residual":res,
      "reduced_chi_square":float(data_fit_residual@data_fit_residual/dof) if effective_covariance_supplied else None,"fit_residual_rms":float(np.sqrt(np.mean(res**2))),
      "optimizer_success":bool(fit.success),"global_search_score":score,"genetic_best_score_history":genetic_history,"inversion_method":inversion_method,"regularization_strength":regularization_strength,"spectral_weighting_status":weighting_status,"spectral_uncertainty_supplied":bool(unc is not None or spectral_covariance is not None),"regularization_scope":"fast-ion VDF curvature and supplied reference-shape regularization only; no plasma-parameter prior","jacobian_rank":rank,"jacobian_condition_number":cond,
      "degrees_of_freedom":int(dof),"scattering_wave_number_m-1":q,"scattering_wavevector_direction":qhat,
      "scattering_angle_rad":angle,"collective_parameter_alpha":float(alpha),"scattering_regime":"collective" if alpha>1 else "weakly_collective_or_noncollective",
      "ion_species":ions,"ion_fractions_charge_density":[float(v) for v in ion_fractions],"probe_frequency_Hz":f0,
      "model":"unmagnetized electrostatic longitudinal microwave CTS",
      "species_resolved_ion_temperature_input_supported":True,
      "species_resolved_ion_temperature_inference_supported":True,
      "external_species_covariance_parameter_families":[
          "ion_fraction", "ion_mass_amu", "ion_charge_state", "ion_temperature_eV"
      ],
      "background_model":"constant_offset" if background_basis is None else "reference_spectrum_PCA",
      "background_spectrum":background_fit.copy(),
      "background_coefficients":background_coefficients.copy(),
      "background_coefficient_names":(
          ["constant_offset"] if background_basis is None else
          ["reference_mean"] + [f"pca_component_{i+1}" for i in range(background_pca_components_used)]
      ),
      "background_coefficient_covariance":background_coefficient_covariance.copy(),
      "background_spectrum_covariance":background_spectrum_covariance.copy(),
      "background_spectrum_uncertainty":background_spectrum_uncertainty.copy(),
      "background_uncertainty_status":(
          "conditional_linear_nuisance_covariance_from_supplied_spectral_uncertainty"
          if unc is not None or spectral_covariance is not None else
          "conditional_linear_nuisance_covariance_scaled_by_fit_residual_variance"
      ),
      "background_reference_basis_uncertainty_propagated":False,
      "background_reference_basis_covariance":None,
      "background_reference_basis_parameter_order":[],
      "background_reference_basis_uncertainty_status":"not_applicable_no_reference_PCA" if background_basis is None else "not_computed",
      "background_reference_mean":None if background_reference_mean is None else background_reference_mean.copy(),
      "background_pca_components_used":int(background_pca_components_used),
      "background_provenance":"single fitted constant offset nuisance" if background_basis is None else "PCA nuisance subspace derived only from supplied calibrated reference spectra on the same frequency grid; no reference data are modified or recalibrated",
      "parameter_provenance":{"electron_density":"inferred from measured CTS spectral shape","electron_temperature":"inferred from measured CTS spectral shape",
       "ion_temperature":"one independently inferred Maxwellian ion temperature per supplied species",
       "ion_velocity_projection":"one common projected ion flow inferred from the measured ion-feature Doppler shift and shared by all supplied species",
       "ion_composition":"externally supplied fixed composition",
       "electron_velocity_projection":"fitted scalar projection along the scattering wavevector from the measured CTS spectrum"},
      "uncertainty_model":"Local Jacobian covariance with optional full spectral covariance whitening; response-matrix uncertainty is propagated by Monte Carlo when supplied.",
      "limitations":["Not a full magnetized electromagnetic CTS tensor or ray-tracing model.","Absolute power transfer and polarization coupling are not inverted.",
       "Ion composition is fixed externally.",
       "The electrostatic inversion independently infers one Maxwellian ion temperature per supplied species; projected ion flow remains a shared scalar because independent species flows are not generally identifiable from one spectrum without additional constraints.",
       "Only the electron-flow component projected along the scattering wavevector is inferred; transverse vector components are not identifiable from one scattering geometry.",
       "Fast-ion VDF support is one-dimensional; fixed-tabulated mode treats shape as external, while infer_tabulated_1d reconstructs it under grid and regularization constraints.",
       "Model discrepancy remains a separate diagnostic and is not automatically a confidence interval.","Response-matrix uncertainty is propagated by Monte Carlo when supplied.",
       "The optional structured background uses PCA modes from explicitly supplied calibrated reference spectra; it is conditional on those references and does not infer or calibrate missing background measurements."]}
    if background_basis is not None:
        if isinstance(background_reference_basis_mc_samples, (bool, np.bool_)) or not isinstance(background_reference_basis_mc_samples, (int, np.integer)) or background_reference_basis_mc_samples < 30:
            raise ValueError("background_reference_basis_mc_samples must be an integer >=30 when PCA reference background is selected")
        from .advanced_inference import bootstrap_reference_pca_basis
        basis_uncertainty = bootstrap_reference_pca_basis(
            reference, components=background_pca_components_used,
            bootstrap_samples=int(background_reference_basis_mc_samples),
            random_seed=int(background_reference_basis_mc_seed),
        )
        out["background_reference_basis_covariance"] = basis_uncertainty["bootstrap_parameter_covariance"]
        out["background_reference_basis_parameter_order"] = basis_uncertainty["bootstrap_parameter_order"]
        out["background_reference_basis_relative_singular_value_gaps"] = basis_uncertainty["component_relative_singular_value_gaps"]
        out["background_reference_basis_weakly_identified_components"] = basis_uncertainty["weakly_identified_components"]
        out["background_reference_basis_bootstrap_samples"] = basis_uncertainty["bootstrap_samples"]
        out["background_reference_basis_uncertainty_status"] = "bootstrap_covariance_of_reference_mean_and_PCA_basis;_not_yet_propagated_to_plasma_parameter_covariance"
        out["background_reference_basis_uncertainty_propagated"] = False
    if independent_discrepancy_covariance is not None:
        out["independent_model_discrepancy_covariance"] = independent_discrepancy_covariance.copy()
        out["independent_model_discrepancy_supplied"] = True
        out["model_discrepancy_model"] = "researcher_supplied_correlated_covariance_independent_of_reduced_chi_square"
        out["model_discrepancy_parameter_covariance_status"] = "included_as_additive_spectral_covariance_in_GLS_whitening"
    else:
        out["independent_model_discrepancy_covariance"] = None
        out["independent_model_discrepancy_supplied"] = False
    if out["reduced_chi_square"] is not None and independent_discrepancy_covariance is None:
        discrepancy_scale=max(float(out["reduced_chi_square"])-1.0,0.0)
        discrepancy_covariance=discrepancy_scale*cov
        out["model_discrepancy_scale"]=discrepancy_scale
        out["model_discrepancy_covariance"]=discrepancy_covariance
        out["total_fit_covariance_including_discrepancy"]=cov+discrepancy_covariance
        total_sigma=np.sqrt(np.maximum(np.diag(cov+discrepancy_covariance),0.0))
        out["electron_density_uncertainty_m3"]=float(total_sigma[0])
        out["electron_temperature_uncertainty_eV"]=float(total_sigma[1])
        out["ion_temperature_uncertainty_eV"]=(float(total_sigma[2]) if n_species == 1 else total_sigma[2:2+n_species].copy())
        out["ion_temperature_species_uncertainties_eV"]=total_sigma[2:2+n_species].copy()
        out["electron_velocity_projection_uncertainty_m_s"]=float(total_sigma[2+n_species])
        out["ion_velocity_projection_uncertainty_m_s"]=float(total_sigma[3+n_species])
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
        out["fast_ion_fraction_uncertainty"]=float(total_sigma[fast_offset] if out.get("model_discrepancy_scale") is not None else su[fast_offset])
        if fast["kind"]=="maxwellian":
            out["fast_ion_temperature_uncertainty_eV"]=float(total_sigma[fast_offset+1] if out.get("model_discrepancy_scale") is not None else su[fast_offset+1])
        elif fast["kind"]=="infer_tabulated_1d":
            logits=fit.x[fast_offset+1:]
            def decode_pdf(values):
                return _softmax(np.r_[values,0.0])/_trapezoid_node_weights(fast["vdf"]["velocity_m_s"])
            pdf_jac=np.empty((fast["vdf"]["velocity_m_s"].size,logits.size))
            for j in range(logits.size):
                step=1e-5
                plus=logits.copy(); minus=logits.copy()
                plus[j]+=step; minus[j]-=step
                pdf_jac[:,j]=(decode_pdf(plus)-decode_pdf(minus))/(2.0*step)
            vdf_parameter_covariance=cov[fast_offset+1:,fast_offset+1:]
            if out.get("model_discrepancy_scale") is not None:
                vdf_parameter_covariance=(cov+discrepancy_covariance)[fast_offset+1:,fast_offset+1:]
            vdf_covariance=pdf_jac@vdf_parameter_covariance@pdf_jac.T
            out["fast_ion_vdf_covariance"]=vdf_covariance
            out["fast_ion_vdf_uncertainty_m_s_inverse"]=np.sqrt(np.maximum(np.diag(vdf_covariance),0.0))
    if response_sigma is not None or ext_unc or external_covariance is not None:
        rng=np.random.default_rng(int(random_seed)+991); draws=[]
        out["external_parameter_covariance_supplied"] = external_covariance is not None
        if external_covariance is not None:
            out["external_parameter_covariance_order"] = list(external_order)
            out["external_parameter_covariance"] = external_covariance.copy()
        for _ in range(int(response_mc_samples)):
            sampled=response
            if response_sigma is not None:
                sampled=np.maximum(response+rng.normal(0.0,response_sigma),0.0)
                sums=sampled.sum(axis=0)
                if np.any(sums<=0): continue
                sampled=sampled/sums[None,:]
            sampled_f0=f0+rng.normal(0.0,float(ext_unc.get("probe_frequency_Hz",0.0)))
            sampled_electron_flow=None
            sampled_geometry=scattering_geometry
            sampled_fractions=np.asarray(ion_fractions,dtype=float).copy()
            sampled_species=[dict(item) for item in ion_species]
            sampled_temperatures=fitted_ion_temperatures.copy()
            if external_covariance is not None:
                external_draw=external_root @ rng.normal(size=len(external_order))
                sampled_angle=None
                for key, perturbation in zip(external_order, external_draw):
                    if key == "probe_frequency_Hz":
                        sampled_f0 += float(perturbation)
                    elif key == "scattering_angle_rad":
                        sampled_angle=float(scattering_geometry["scattering_angle_rad"])+float(perturbation)
                    elif key.startswith("ion_fraction_"):
                        index=int(key.rsplit("_",1)[1])
                        sampled_fractions[index] += float(perturbation)
                    elif key.startswith("ion_mass_amu_"):
                        index=int(key.rsplit("_",1)[1])
                        sampled_species[index]["mass_amu"]=float(sampled_species[index]["mass_amu"])+float(perturbation)
                    elif key.startswith("ion_charge_state_"):
                        index=int(key.rsplit("_",1)[1])
                        sampled_species[index]["charge_state"]=float(sampled_species[index]["charge_state"])+float(perturbation)
                    elif key.startswith("ion_temperature_eV_"):
                        index=int(key.rsplit("_",1)[1])
                        sampled_temperatures[index]+=float(perturbation)
                if (any(float(item["mass_amu"])<=0 or float(item["charge_state"])<=0 for item in sampled_species)
                        or np.any(sampled_temperatures<=0)):
                    continue
                if sampled_angle is not None:
                    if not 0<sampled_angle<=np.pi: continue
                    sampled_geometry=dict(scattering_geometry)
                    sampled_geometry["scattering_angle_rad"]=sampled_angle
                sampled_fractions=np.maximum(sampled_fractions,1e-12)
                sampled_fractions=sampled_fractions/sampled_fractions.sum()
            else:
                if "scattering_angle_rad" in ext_unc:
                    sampled_geometry=dict(scattering_geometry)
                    sampled_geometry["scattering_angle_rad"]=float(scattering_geometry["scattering_angle_rad"])+rng.normal(0.0,float(ext_unc["scattering_angle_rad"]))
                    if not 0<sampled_geometry["scattering_angle_rad"]<=np.pi: continue
                if "ion_fractions" in ext_unc:
                    sampled_fractions=np.maximum(sampled_fractions+rng.normal(0.0,np.asarray(ext_unc["ion_fractions"],dtype=float)),1e-12)
                    sampled_fractions=sampled_fractions/sampled_fractions.sum()
            if sampled_f0<=0: continue
            try:
                def sampled_residual(p):
                    data=data_residual(p,sampled,sampled_f0,sampled_electron_flow,sampled_geometry,sampled_fractions,sampled_species,sampled_temperatures)
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
            out["ion_temperature_uncertainty_eV"]=(float(total[2]) if n_species == 1 else total[2:2+n_species].copy())
            out["ion_temperature_species_uncertainties_eV"]=total[2:2+n_species].copy()
            out["electron_velocity_projection_uncertainty_m_s"]=float(total[2+n_species])
            out["ion_velocity_projection_uncertainty_m_s"]=float(total[3+n_species])
            if fast:
                out["fast_ion_fraction_uncertainty"]=float(total[fast_offset])
                if fast["kind"]=="maxwellian":
                    out["fast_ion_temperature_uncertainty_eV"]=float(total[fast_offset+1])
                elif fast["kind"]=="infer_tabulated_1d":
                    response_vdf_covariance=pdf_jac@response_cov[fast_offset+1:,fast_offset+1:]@pdf_jac.T
                    out["fast_ion_vdf_covariance"]=out["fast_ion_vdf_covariance"]+response_vdf_covariance
                    out["fast_ion_vdf_uncertainty_m_s_inverse"]=np.sqrt(np.maximum(np.diag(out["fast_ion_vdf_covariance"]),0.0))
            out["response_uncertainty_propagated"]=response_sigma is not None
            out["external_parameter_uncertainty_propagated"]=bool(ext_unc) or external_covariance is not None
            out["external_parameter_covariance_propagated"]=external_covariance is not None
        else:
            out["response_uncertainty_propagated"]=False
            out["external_parameter_uncertainty_propagated"]=False
            out["external_parameter_covariance_propagated"]=False
            out["response_uncertainty_warning"]="Too few successful response/external perturbation refits; no Monte Carlo contribution was added."
    else:
        out["response_uncertainty_propagated"]=False
        out["external_parameter_uncertainty_propagated"]=False
        out["external_parameter_covariance_propagated"]=False
    from .thomson_identifiability import cts_electrostatic_identifiability
    out["parameter_identifiability"] = cts_electrostatic_identifiability(
        fast_ion_model=fast.get("kind") if fast else None
    )
    return out


# Separate magnetized electromagnetic CTS path; the electrostatic API remains unchanged.
from .cts_full_em import full_em_cts_spectral_density, analyze_full_em_cts_spectrum


def analyze_cts_spectrum_model(model=None, **kwargs):
    """Dispatch to exactly one CTS model using a common public spectrum argument.

    The public contract names the calibrated measured spectral-power vector
    ``spectrum`` for both models. The Full-EM implementation retains its
    descriptive internal parameter name ``measured_spectrum``; this adapter
    performs that keyword translation and rejects ambiguous duplicate inputs.
    """
    if model is None:
        raise ValueError("Select model='electrostatic' or model='full_electromagnetic' explicitly; no default model is selected.")
    if model == "electrostatic":
        return analyze_cts_spectrum(**kwargs)
    if model == "full_electromagnetic":
        if "spectrum" in kwargs and "measured_spectrum" in kwargs:
            raise ValueError("Supply the calibrated CTS spectrum once, as 'spectrum' or 'measured_spectrum', not both.")
        if "spectrum" in kwargs:
            kwargs["measured_spectrum"] = kwargs.pop("spectrum")
        return analyze_full_em_cts_spectrum(**kwargs)
    raise ValueError("model must be 'electrostatic' or 'full_electromagnetic'; no automatic model substitution is performed")

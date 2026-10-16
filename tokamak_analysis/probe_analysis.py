"""Basic Langmuir, ball-pen and multi-pin probe data analysis."""
import numpy as np
def analyze_probe_voltage_current(voltage_V,current_A,*,electron_fit_range=None,ion_fit_range=None):
    v=np.asarray(voltage_V,dtype=float); i=np.asarray(current_A,dtype=float)
    if v.ndim!=1 or i.shape!=v.shape or len(v)<5 or not np.all(np.isfinite(v)) or not np.all(np.isfinite(i)): raise ValueError("V and I must be finite equal-length 1-D arrays with >=5 points")
    order=np.argsort(v); v=v[order]; i=i[order]
    if np.any(np.diff(v)<=0): raise ValueError("Voltage values must be unique")
    dvdi=np.gradient(i,v,edge_order=2)
    # Floating potential is the voltage where I(V)=0. Interpolate in voltage order;
    # sorting the current itself would fabricate a crossing for non-monotonic/noisy I-V data.
    zero = (i == 0.0)
    candidates = []
    zero_idx = np.flatnonzero(zero)
    if zero_idx.size:
        groups = np.split(zero_idx, np.flatnonzero(np.diff(zero_idx) > 1) + 1)
        candidates.extend(float(0.5 * (v[g[0]] + v[g[-1]])) for g in groups)
    for k in range(len(i) - 1):
        if not zero[k] and not zero[k + 1] and i[k] * i[k + 1] < 0:
            candidates.append(float(v[k] - i[k] * (v[k + 1] - v[k]) / (i[k + 1] - i[k])))
    candidates = sorted(candidates)
    unique_candidates = []
    for candidate in candidates:
        if not unique_candidates or not np.isclose(candidate, unique_candidates[-1], rtol=1e-10, atol=1e-12):
            unique_candidates.append(candidate)
    vf = unique_candidates[0] if len(unique_candidates) == 1 else None
    out={"voltage_V":v,"current_A":i,"dI_dV_S":dvdi,"floating_potential_V":vf,"floating_potential_candidates_V":unique_candidates,"ion_saturation_current_A":float(np.median(i[v<=np.quantile(v,.2)])),"electron_saturation_current_A":float(np.median(i[v>=np.quantile(v,.8)])),"caveat":"These are descriptive I-V metrics only; reliable Te and plasma-potential inference requires physically selected fit regions and probe-specific theory."}
    if electron_fit_range is not None:
        lo,hi=map(float,electron_fit_range)
        if not np.all(np.isfinite([lo,hi])) or hi<=lo: raise ValueError("electron_fit_range must have finite increasing bounds")
        ion_baseline=out["ion_saturation_current_A"]
        electron_current=i-ion_baseline
        m=(v>=lo)&(v<=hi)&(electron_current>0)
        if np.count_nonzero(m)>=3:
            slope=np.polyfit(v[m],np.log(electron_current[m]),1)[0]
            out["electron_temperature_estimate_eV"]=1/slope if slope>0 else None
            out["electron_fit_points"]=int(np.count_nonzero(m))
            out["electron_current_used_A"]=electron_current[m]
            out["electron_fit_note"]="Ion-saturation baseline subtracted before log-linear fitting; verify current polarity and fit interval."
    if ion_fit_range is not None:
        lo,hi=map(float,ion_fit_range)
        if not np.all(np.isfinite([lo,hi])) or hi<=lo: raise ValueError("ion_fit_range must have finite increasing bounds")
        m=(v>=lo)&(v<=hi)
        out["ion_fit_linear_slope_A_V"]=float(np.polyfit(v[m],i[m],1)[0]) if np.count_nonzero(m)>=2 else None
    return out

def electron_retarding_current(V_probe_V, plasma_potential_V, electron_saturation_current_A, electron_temperature_eV):
    """Maxwellian electron-retarding branch: Ie=Ies*exp((V-Vp)/Te[eV])."""
    v=np.asarray(V_probe_V,dtype=float); vp=float(plasma_potential_V); ies=float(electron_saturation_current_A); te=float(electron_temperature_eV)
    if not np.all(np.isfinite(v)) or not np.isfinite(vp) or not np.isfinite(ies) or not np.isfinite(te) or ies<=0 or te<=0:
        raise ValueError("Inputs must be finite; Ies and Te must be positive")
    if np.any(v>vp): raise ValueError("This relation applies only to the electron-retarding region V <= Vp")
    current=ies*np.exp((v-vp)/te)
    return {"electron_current_A":current,"equation":"Ie=Ies*exp((Vprobe-Vplasma)/Te_eV)",
            "assumption":"Ideal Maxwellian electron-retarding branch; sheath geometry, collisions, and ion current are not included."}

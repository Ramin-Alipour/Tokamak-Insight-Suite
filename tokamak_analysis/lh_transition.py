"""L-H transition candidate detection from time-series change points."""
import numpy as np
def detect_lh_transition(time_s, signal, *, baseline_window=None, threshold_sigma=3.0, persistence=5):
    t=np.asarray(time_s,dtype=float); x=np.asarray(signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<10 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Invalid transition signal")
    if not np.isfinite(threshold_sigma) or threshold_sigma<0: raise ValueError("threshold_sigma must be finite and non-negative")
    if not np.isfinite(persistence) or int(persistence)!=persistence or int(persistence)<1: raise ValueError("persistence must be a positive integer")
    if baseline_window is None: base=x[:max(5,len(x)//5)]
    else:
        lo,hi=map(float,baseline_window)
        if not np.all(np.isfinite([lo,hi])) or hi<=lo: raise ValueError("Baseline window must have finite increasing bounds")
        base=x[(t>=lo)&(t<=hi)]
        if len(base)<3: raise ValueError("Baseline window has too few samples")
    mu=float(np.median(base)); sigma=float(1.4826*np.median(np.abs(base-mu)))
    threshold=mu+float(threshold_sigma)*max(sigma,np.finfo(float).eps)
    active=x>threshold; p=max(1,int(persistence)); idx=None
    for i in range(0,len(x)-p+1):
        if np.all(active[i:i+p]): idx=i; break
    return {"transition_candidate_time_s":float(t[idx]) if idx is not None else None,"threshold":threshold,"baseline_median":mu,"baseline_robust_sigma":sigma,"caveat":"Single-signal threshold is a candidate detector, not definitive L-H classification; combine density, radiation, D-alpha and confinement evidence."}

def martin_lh_threshold_power_MW(line_average_density_m3, toroidal_field_T, plasma_surface_area_m2):
    """Martin 2008 empirical L-H threshold scaling: PLH=0.0488*n20^0.717*B^0.803*S^0.941 MW."""
    n,B,S=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (line_average_density_m3,toroidal_field_T,plasma_surface_area_m2)])
    if any(not np.all(np.isfinite(v)) for v in (n,B,S)) or np.any(n<=0) or np.any(B<=0) or np.any(S<=0):
        raise ValueError("Density, field, and surface area must be finite and positive")
    n20=n/1e20; power=0.0488*n20**0.717*B**0.803*S**0.941
    return {"lh_threshold_power_MW":power,"equation":"P_LH[MW]=0.0488*n20^0.717*B_T^0.803*S_m2^0.941",
            "caveat":"Empirical multi-machine scaling; applicability and uncertainty depend on plasma configuration, isotope, and parameter definitions."}

"""L-H transition candidate detection from time-series change points."""
import numpy as np
def detect_lh_transition(time_s, signal, *, baseline_window=None, threshold_sigma=3.0, persistence=5):
    t=np.asarray(time_s,dtype=float); x=np.asarray(signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<10 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Invalid transition signal")
    if baseline_window is None: base=x[:max(5,len(x)//5)]
    else:
        lo,hi=baseline_window; base=x[(t>=lo)&(t<=hi)]
        if len(base)<3: raise ValueError("Baseline window has too few samples")
    mu=float(np.median(base)); sigma=float(1.4826*np.median(np.abs(base-mu)))
    threshold=mu+float(threshold_sigma)*max(sigma,np.finfo(float).eps)
    active=x>threshold; p=max(1,int(persistence)); idx=None
    for i in range(0,len(x)-p+1):
        if np.all(active[i:i+p]): idx=i; break
    return {"transition_candidate_time_s":float(t[idx]) if idx is not None else None,"threshold":threshold,"baseline_median":mu,"baseline_robust_sigma":sigma,"caveat":"Single-signal threshold is a candidate detector, not definitive L-H classification; combine density, radiation, D-alpha and confinement evidence."}
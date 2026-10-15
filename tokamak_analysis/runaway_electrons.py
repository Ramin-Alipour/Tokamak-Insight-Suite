"""Runaway-electron event metrics from hard-X-ray/current time series."""
import numpy as np
def analyze_runaway_events(time_s, hard_xray_signal, *, threshold=None, plasma_current_A=None):
    t=np.asarray(time_s,dtype=float); x=np.asarray(hard_xray_signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Finite increasing time and matching HXR signal required")
    thr=float(threshold) if threshold is not None else float(np.mean(x)+3*np.std(x))
    mask=x>=thr; edges=np.diff(np.r_[False,mask,False].astype(int)); starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
    events=[{"start_s":float(t[a]),"end_s":float(t[min(b-1,len(t)-1)]),"duration_s":float(t[min(b-1,len(t)-1)]-t[a]),"peak_signal":float(np.max(x[a:b]))} for a,b in zip(starts,ends)]
    out={"threshold":thr,"events":events,"event_count":len(events),"above_threshold_fraction":float(np.mean(mask))}
    if plasma_current_A is not None:
        ip=np.asarray(plasma_current_A,dtype=float)
        if ip.shape!=t.shape or not np.all(np.isfinite(ip)): raise ValueError("Plasma current must match time")
        out["plasma_current_A"]=ip
    out["caveat"]="HXR bursts are a proxy for energetic-electron activity, not unique proof of runaway electrons or a direct energy distribution."
    return out
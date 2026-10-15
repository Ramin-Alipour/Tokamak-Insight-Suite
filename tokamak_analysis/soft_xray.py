"""Soft X-ray channel time-series and radiation-burst analysis."""
import numpy as np
def analyze_soft_xray(time_s, channel_signals, *, baseline_window=None):
    t=np.asarray(time_s,dtype=float); x=np.asarray(channel_signals,dtype=float)
    if t.ndim!=1 or x.ndim not in (1,2) or x.shape[0]!=len(t) or len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Finite time and matching 1-D/2-D channels required")
    if baseline_window is None: base=np.median(x[:max(1,len(t)//10)],axis=0)
    else:
        lo,hi=baseline_window; m=(t>=lo)&(t<=hi)
        if not np.any(m): raise ValueError("No samples in baseline window")
        base=np.median(x[m],axis=0)
    signal=x-base
    return {"time_s":t,"baseline":base,"baseline_subtracted_signal":signal,"peak_by_channel":np.max(signal,axis=0),"integrated_signal_by_channel":np.trapezoid(signal,t,axis=0),"caveat":"Without detector response and line-of-sight inversion, signals are not absolute emissivity or temperature."}
"""Synchronize diagnostic channels onto a common time grid."""
import numpy as np
def synchronize_channels(time_arrays, signal_arrays, *, target_time=None):
    if len(time_arrays)!=len(signal_arrays) or len(time_arrays)<1: raise ValueError("Provide matching nonempty time and signal lists")
    ts=[]; ys=[]
    for t0,y0 in zip(time_arrays,signal_arrays):
        t=np.asarray(t0,dtype=float); y=np.asarray(y0,dtype=float)
        if t.ndim!=1 or y.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(np.diff(t)<=0): raise ValueError("Each channel needs finite increasing time and matching signal")
        ts.append(t); ys.append(y)
    common=np.asarray(target_time,dtype=float) if target_time is not None else ts[0][(ts[0]>=max(t[0] for t in ts))&(ts[0]<=min(t[-1] for t in ts))]
    if common.ndim!=1 or len(common)<1 or not np.all(np.isfinite(common)): raise ValueError("Common time grid is empty or invalid")
    synced=np.vstack([np.interp(common,t,y) for t,y in zip(ts,ys)])
    return {"time_s":common,"synchronized_signals":synced,"channel_count":len(ts),"caveat":"Linear interpolation assumes no significant aliasing and does not correct hardware latency; determine timing offsets independently."}
"""Synchronize diagnostic channels onto a common time grid."""
import numpy as np
def synchronize_channels(time_arrays, signal_arrays, *, target_time=None):
    if len(time_arrays)!=len(signal_arrays) or len(time_arrays)<1: raise ValueError("Provide matching nonempty time and signal lists")
    ts=[]; ys=[]
    for t0,y0 in zip(time_arrays,signal_arrays):
        t=np.asarray(t0,dtype=float); y=np.asarray(y0,dtype=float)
        if t.ndim!=1 or y.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(np.diff(t)<=0): raise ValueError("Each channel needs finite increasing time and matching signal")
        ts.append(t); ys.append(y)
    overlap_start=max(t[0] for t in ts); overlap_stop=min(t[-1] for t in ts)
    common=np.asarray(target_time,dtype=float) if target_time is not None else ts[0][(ts[0]>=overlap_start)&(ts[0]<=overlap_stop)]
    if common.ndim!=1 or len(common)<1 or not np.all(np.isfinite(common)) or np.any(np.diff(common)<=0): raise ValueError("Common time grid must be finite, nonempty, and strictly increasing")
    if common[0]<overlap_start or common[-1]>overlap_stop: raise ValueError("Target time grid must lie inside the channels’ shared time interval; extrapolation is not allowed")
    synced=np.vstack([np.interp(common,t,y) for t,y in zip(ts,ys)])
    return {"time_s":common,"synchronized_signals":synced,"channel_count":len(ts),"caveat":"Linear interpolation assumes no significant aliasing and does not correct hardware latency; determine timing offsets independently."}

def estimate_channel_time_lag(reference_time_s, reference_signal, target_time_s, target_signal):
    """Estimate relative delay by maximizing discrete cross-correlation on the common uniform grid."""
    from scipy.signal import correlate, correlation_lags
    t1=np.asarray(reference_time_s,dtype=float); x1=np.asarray(reference_signal,dtype=float)
    t2=np.asarray(target_time_s,dtype=float); x2=np.asarray(target_signal,dtype=float)
    if t1.ndim!=1 or x1.shape!=t1.shape or t2.ndim!=1 or x2.shape!=t2.shape or min(len(t1),len(t2))<3:
        raise ValueError("Each channel requires matching 1-D time and signal arrays")
    if any(not np.all(np.isfinite(v)) for v in (t1,x1,t2,x2)) or np.any(np.diff(t1)<=0) or np.any(np.diff(t2)<=0):
        raise ValueError("Inputs must be finite with strictly increasing time")
    dt=max(float(np.median(np.diff(t1))),float(np.median(np.diff(t2))))
    start=max(t1[0],t2[0]); stop=min(t1[-1],t2[-1]); grid=np.arange(start,stop+0.5*dt,dt)
    if len(grid)<3: raise ValueError("Insufficient overlapping time interval")
    a=np.interp(grid,t1,x1); b=np.interp(grid,t2,x2); a=a-np.mean(a); b=b-np.mean(b)
    corr=correlate(b,a,mode="full",method="auto"); lags=correlation_lags(len(b),len(a),mode="full")
    lag=int(lags[np.argmax(corr)])
    return {"target_delay_relative_to_reference_s":float(lag*dt),"lag_samples":lag,"sample_interval_s":dt,
            "equation":"lag=argmax_tau sum(reference(t)*target(t+tau))",
            "caveat":"Cross-correlation lag is an estimate; periodic signals, different waveforms and limited overlap can create ambiguous peaks."}

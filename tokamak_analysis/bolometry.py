"""Bolometer signal integration and radiated-power estimate."""
import numpy as np
def analyze_bolometry(time_s, channel_power_W_m2, *, channel_weights=None):
    t=np.asarray(time_s,dtype=float); p=np.asarray(channel_power_W_m2,dtype=float)
    if t.ndim!=1 or p.ndim not in (1,2) or p.shape[0]!=len(t) or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or np.any(np.diff(t)<=0): raise ValueError("Finite increasing time and matching signals required")
    energy=np.trapezoid(p,t,axis=0)
    out={"time_s":t,"signal_W_m2":p,"time_integrated_signal_J_m2":energy,"peak_signal_W_m2":np.max(p,axis=0)}
    if p.ndim==2 and channel_weights is not None:
        w=np.asarray(channel_weights,dtype=float)
        if w.shape!=(p.shape[1],) or not np.all(np.isfinite(w)): raise ValueError("Weights must match channel count")
        out["weighted_total_signal"]=p@w; out["weighted_integral"]=float(np.trapezoid(p@w,t))
    out["caveat"]="Bolometer power requires detector calibration, view factors and geometric inversion; channel sums alone are not total plasma radiated power."
    return out
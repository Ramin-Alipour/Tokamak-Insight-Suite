"""Hard X-ray count-spectrum analysis and calibration-aware energy summaries."""
import numpy as np
def analyze_hard_xray(energy_keV, counts, *, live_time_s=None, background_counts=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0): raise ValueError("Energy bins increasing; counts finite and nonnegative")
    net=c.copy()
    if background_counts is not None:
        bg=np.asarray(background_counts,dtype=float)
        if bg.shape!=c.shape or not np.all(np.isfinite(bg)) or np.any(bg<0): raise ValueError("Background must match counts and be nonnegative")
        net=np.maximum(c-bg,0)
    total=float(np.sum(net)); out={"energy_keV":e,"net_counts":net,"total_net_counts":total,"peak_energy_keV":float(e[np.argmax(net)]),"count_weighted_mean_energy_keV":float(np.sum(e*net)/total) if total else float("nan")}
    if live_time_s is not None:
        t=float(live_time_s)
        if not np.isfinite(t) or t<=0: raise ValueError("Live time must be positive")
        out["count_rate_s-1"]=total/t
    out["caveat"]="Energy calibration, detector efficiency, dead time and background uncertainty are not corrected unless separately supplied."
    return out
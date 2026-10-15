"""Ball-pen probe potential comparison and fluctuation metrics."""
import numpy as np
def analyze_ball_pen_probe(time_s, ball_pen_V, floating_V=None):
    t=np.asarray(time_s,dtype=float); v=np.asarray(ball_pen_V,dtype=float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or np.any(np.diff(t)<=0): raise ValueError("Finite equal-length data and strictly increasing time required")
    out={"time_s":t,"ball_pen_potential_V":v,"mean_V":float(np.mean(v)),"std_V":float(np.std(v,ddof=1)),"peak_to_peak_V":float(np.ptp(v))}
    if floating_V is not None:
        vf=np.asarray(floating_V,dtype=float)
        if vf.shape!=v.shape or not np.all(np.isfinite(vf)): raise ValueError("Floating-potential signal must match ball-pen signal")
        out["floating_potential_V"]=vf; out["potential_difference_V"]=v-vf; out["mean_potential_difference_V"]=float(np.mean(v-vf))
    out["caveat"]="Ball-pen potential interpretation depends on probe geometry, emissive/collection physics, and calibration."
    return out
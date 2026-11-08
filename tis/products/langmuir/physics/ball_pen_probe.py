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

def plasma_potential_from_ball_pen(ball_pen_potential_V, electron_temperature_eV, calibrated_offset_factor):
    """Apply calibrated ball-pen correction: Vp=V_BPP+alpha*Te[eV]."""
    vb,te,alpha=np.broadcast_arrays(np.asarray(ball_pen_potential_V,dtype=float),np.asarray(electron_temperature_eV,dtype=float),np.asarray(calibrated_offset_factor,dtype=float))
    if any(not np.all(np.isfinite(x)) for x in (vb,te,alpha)) or np.any(te<0):
        raise ValueError("Inputs must be finite and electron temperature non-negative")
    vp=vb+alpha*te
    return {"plasma_potential_V":vp,"ball_pen_potential_V":vb,"calibrated_offset_factor":alpha,
            "equation":"Vp=V_BPP+alpha*Te_eV",
            "caveat":"alpha must be independently calibrated for the probe geometry and plasma regime; it is not universal."}

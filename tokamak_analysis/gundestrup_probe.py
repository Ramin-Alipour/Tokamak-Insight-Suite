"""Directional Gundestrup probe analysis using angular current harmonics."""
import numpy as np
def analyze_gundestrup(angles_rad, collector_currents_A):
    th=np.asarray(angles_rad,dtype=float); cur=np.asarray(collector_currents_A,dtype=float)
    if th.ndim!=1 or cur.shape!=th.shape or len(th)<3 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(cur)): raise ValueError("At least three finite angle/current samples required")
    X=np.column_stack([np.ones_like(th),np.cos(th),np.sin(th)])
    coef,_,rank,_=np.linalg.lstsq(X,cur,rcond=None)
    if rank<3: raise ValueError("Collector angles do not constrain first harmonic")
    amp=float(np.hypot(coef[1],coef[2])); phase=float(np.arctan2(coef[2],coef[1]))
    fit=X@coef
    return {"offset_current_A":float(coef[0]),"first_harmonic_amplitude_A":amp,"flow_phase_rad":phase,"fit_current_A":fit,"residual_rms_A":float(np.sqrt(np.mean((cur-fit)**2))),"caveat":"Flow velocity requires probe calibration and a plasma-sheath/collection model; harmonic phase alone is not a velocity."}
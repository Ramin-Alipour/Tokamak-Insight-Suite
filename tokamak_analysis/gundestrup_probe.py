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

def calibrated_mach_from_current_harmonics(offset_current_A, first_harmonic_amplitude_A, current_ratio_per_mach):
    """Infer projected Mach number from calibrated harmonic ratio M=(I1/I0)/C."""
    i0=float(offset_current_A); i1=float(first_harmonic_amplitude_A); c=float(current_ratio_per_mach)
    if not np.all(np.isfinite([i0,i1,c])) or i0<=0 or i1<0 or c<=0:
        raise ValueError("Require finite positive offset current and calibration, and non-negative harmonic amplitude")
    ratio=i1/i0; mach=ratio/c
    return {"first_harmonic_fraction":ratio,"projected_mach_number":mach,
            "equation":"M_projected=(I1/I0)/C_calibrated",
            "caveat":"C_calibrated depends on probe geometry, ion collection, sheath physics, and flow orientation; this is not a universal conversion."}

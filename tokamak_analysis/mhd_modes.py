"""Toroidal/poloidal mode-number estimation from multi-channel phase."""
import numpy as np
def estimate_mode_number(angle_rad, complex_amplitudes):
    th=np.asarray(angle_rad,dtype=float); a=np.asarray(complex_amplitudes)
    if th.ndim!=1 or a.shape!=th.shape or len(th)<3 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(a)): raise ValueError("At least three matching finite angular channels required")
    phase=np.unwrap(np.angle(a)); coeff=np.polyfit(th,phase,1)
    return {"mode_number_estimate":float(coeff[0]),"nearest_integer_mode_number":int(np.rint(coeff[0])),"phase_rad":phase,"fit_residual_rms_rad":float(np.sqrt(np.mean((phase-np.polyval(coeff,th))**2))),"caveat":"Channel geometry, phase calibration, aliasing and mode mixing can bias mode-number estimates."}
"""Toroidal/poloidal mode-number estimation from multi-channel phase."""
import numpy as np
def estimate_mode_number(angle_rad, complex_amplitudes):
    th=np.asarray(angle_rad,dtype=float); a=np.asarray(complex_amplitudes)
    if th.ndim!=1 or a.shape!=th.shape or len(th)<3 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(a)): raise ValueError("At least three matching finite angular channels required")
    if np.any(np.abs(a)<=np.finfo(float).tiny): raise ValueError("Mode phase is undefined for zero-amplitude channels")
    order=np.argsort(th); th=th[order]; a=a[order]
    if np.any(np.diff(th)<=0): raise ValueError("Sensor angles must be unique")
    phase=np.unwrap(np.angle(a)); coeff=np.polyfit(th,phase,1)
    return {"mode_number_estimate":float(coeff[0]),"nearest_integer_mode_number":int(np.rint(coeff[0])),"phase_rad":phase,"fit_residual_rms_rad":float(np.sqrt(np.mean((phase-np.polyval(coeff,th))**2))),"caveat":"Channel geometry, phase calibration, aliasing and mode mixing can bias mode-number estimates."}

def helical_mode_phase(poloidal_mode_number, toroidal_mode_number, poloidal_angle_rad, toroidal_angle_rad, angular_frequency_rad_s, time_s, *, phase_offset_rad=0.0):
    """Ideal helical perturbation phase: phase=m*theta-n*phi-omega*t+phase0."""
    m,n,theta,phi,omega,t,phase0=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (poloidal_mode_number,toroidal_mode_number,poloidal_angle_rad,toroidal_angle_rad,angular_frequency_rad_s,time_s,phase_offset_rad)])
    if any(not np.all(np.isfinite(v)) for v in (m,n,theta,phi,omega,t,phase0)) or np.any(m!=np.floor(m)) or np.any(n!=np.floor(n)):
        raise ValueError("Mode numbers must be integers and all inputs finite")
    return {"helical_phase_rad":m*theta-n*phi-omega*t+phase0,
            "equation":"phase=m*theta-n*phi-omega*t+phase0"}

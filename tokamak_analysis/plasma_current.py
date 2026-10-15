"""Plasma-current signal analysis from calibrated Rogowski/loop-voltage-derived data."""
import numpy as np

def analyze_plasma_current(time_s, current_A, *, uncertainty_A=None, smoothing_window=1):
    """Compute current summary, dIp/dt, and ramp/flat-top intervals from calibrated Ip.

    This function does not infer current from raw coil voltage; Rogowski integration
    and instrument calibration must be performed separately.
    """
    t=np.asarray(time_s,dtype=float); ip=np.asarray(current_A,dtype=float)
    if t.ndim!=1 or ip.ndim!=1 or len(t)!=len(ip) or len(t)<3: raise ValueError("time and current must be equal-length 1-D arrays with at least 3 points")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(ip)) or np.any(np.diff(t)<=0): raise ValueError("Inputs must be finite and time strictly increasing")
    if int(smoothing_window)<1 or int(smoothing_window)>len(ip): raise ValueError("smoothing_window must be between 1 and data length")
    if uncertainty_A is not None:
        u=np.asarray(uncertainty_A,dtype=float)
        if u.shape!=ip.shape or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("uncertainty_A must be finite, non-negative, and match current")
    else: u=None
    if smoothing_window>1:
        w=int(smoothing_window); kernel=np.ones(w)/w; pad_left=w//2; pad_right=w-1-pad_left
        smooth=np.convolve(np.pad(ip,(pad_left,pad_right),mode="edge"),kernel,mode="valid")
    else: smooth=ip.copy()
    derivative=np.gradient(smooth,t,edge_order=2)
    peak_idx=int(np.argmax(np.abs(smooth))); peak=float(smooth[peak_idx])
    flat_tol=max(float(np.nanpercentile(np.abs(derivative),25)),np.finfo(float).eps)
    flat=np.abs(derivative)<=flat_tol
    return {"time_s":t.copy(),"current_A":ip.copy(),"smoothed_current_A":smooth,"dIp_dt_A_s":derivative,"peak_signed_current_A":peak,"peak_absolute_current_A":float(np.max(np.abs(smooth))),"peak_time_s":float(t[peak_idx]),"time_integral_A_s":float(np.trapezoid(smooth,t)),"ramp_up_indices":np.flatnonzero(derivative>flat_tol),"ramp_down_indices":np.flatnonzero(derivative < -flat_tol),"flat_top_candidate_indices":np.flatnonzero(flat),"uncertainty_A":u,"caveat":"Flat-top detection is a simple derivative threshold; select thresholds based on noise and machine-specific criteria."}
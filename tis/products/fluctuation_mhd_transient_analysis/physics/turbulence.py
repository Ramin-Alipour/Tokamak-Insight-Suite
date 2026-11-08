"""Plasma fluctuation, cross-phase and covariance metrics."""
import numpy as np
from scipy.signal import welch, coherence
def analyze_turbulence(time_s, signal_a, signal_b=None):
    t=np.asarray(time_s,dtype=float); a=np.asarray(signal_a,dtype=float)
    if t.ndim!=1 or a.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(a)) or np.any(np.diff(t)<=0): raise ValueError("Invalid time series")
    dt=np.median(np.diff(t))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    out={"mean":float(np.mean(a)),"rms_fluctuation":float(np.std(a,ddof=1)),"relative_rms":float(np.std(a,ddof=1)/abs(np.mean(a))) if np.mean(a)!=0 else float("nan")}
    f,p=welch(a-np.mean(a),fs=1/dt,nperseg=min(512,len(a))); out.update({"frequency_Hz":f,"psd":p})
    if signal_b is not None:
        b=np.asarray(signal_b,dtype=float)
        if b.shape!=a.shape or not np.all(np.isfinite(b)): raise ValueError("Second signal must match")
        out["cross_covariance"]=float(np.mean((a-np.mean(a))*(b-np.mean(b))))
        fc,coh=coherence(a,b,fs=1/dt,nperseg=min(512,len(a))); out["coherence_frequency_Hz"]=fc; out["coherence"]=coh
    return out

def mixing_length_diffusivity(growth_rate_s, perpendicular_wavenumber_m):
    """Mixing-length estimate D~gamma/k_perp^2 from linear growth rate and perpendicular wavenumber."""
    gamma,k=np.broadcast_arrays(np.asarray(growth_rate_s,dtype=float),np.asarray(perpendicular_wavenumber_m,dtype=float))
    if not np.all(np.isfinite(gamma)) or not np.all(np.isfinite(k)) or np.any(gamma<0) or np.any(k<=0):
        raise ValueError("Growth rate must be non-negative and perpendicular wavenumber positive")
    return {"mixing_length_diffusivity_m2_s":gamma/k**2,
            "equation":"D_mix~gamma/k_perp^2",
            "caveat":"Order-of-magnitude mixing-length estimate, not a measured transport coefficient."}

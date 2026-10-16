"""Dominant magnetic-mode frequency and spectral peak analysis."""
import numpy as np
from scipy.signal import welch, find_peaks
def magnetic_mode_frequency(time_s, signal, *, frequency_band_Hz=None):
    t=np.asarray(time_s,dtype=float); x=np.asarray(signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Invalid time series")
    dt=np.median(np.diff(t))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    f,p=welch(x-np.mean(x),fs=1/dt,nperseg=min(2048,len(x))); m=f>0
    if frequency_band_Hz is not None:
        lo,hi=map(float,frequency_band_Hz)
        if not np.all(np.isfinite([lo,hi])) or lo<0 or hi<=lo: raise ValueError("Frequency band must have finite increasing non-negative bounds")
        m &= (f>=lo)&(f<=hi)
    if not np.any(m): raise ValueError("No frequency bins in requested range")
    inds=np.flatnonzero(m); idx=inds[np.argmax(p[m])]
    return {"dominant_frequency_Hz":float(f[idx]),"frequency_Hz":f,"power_spectral_density":p,"peak_power":float(p[idx])}

def frequency_from_unwrapped_phase(time_s, phase_rad):
    """Estimate coherent mode frequency from phase: f=(1/2pi)*d(phase)/dt."""
    t=np.asarray(time_s,dtype=float); phase=np.asarray(phase_rad,dtype=float)
    if t.ndim!=1 or phase.shape!=t.shape or len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(phase)) or np.any(np.diff(t)<=0):
        raise ValueError("Need matching finite arrays with strictly increasing time")
    coeff=np.polyfit(t,phase,1); freq=float(coeff[0]/(2*np.pi))
    residual=phase-np.polyval(coeff,t)
    return {"mode_frequency_Hz":freq,"angular_frequency_rad_s":float(coeff[0]),"phase_residual_rms_rad":float(np.sqrt(np.mean(residual**2))),
            "equation":"f=(1/(2*pi))*d(phase_unwrapped)/dt"}

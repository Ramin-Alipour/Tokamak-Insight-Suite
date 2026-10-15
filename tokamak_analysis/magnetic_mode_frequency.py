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
        lo,hi=map(float,frequency_band_Hz); m &= (f>=lo)&(f<=hi)
    if not np.any(m): raise ValueError("No frequency bins in requested range")
    inds=np.flatnonzero(m); idx=inds[np.argmax(p[m])]
    return {"dominant_frequency_Hz":float(f[idx]),"frequency_Hz":f,"power_spectral_density":p,"peak_power":float(p[idx])}
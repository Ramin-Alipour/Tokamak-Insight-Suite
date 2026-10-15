"""Density fluctuation spectrum, band power, and propagation estimates."""
import numpy as np
from scipy.signal import welch
def analyze_density_waves(time_s,density_m3,*,band_Hz=None):
    t=np.asarray(time_s,dtype=float); n=np.asarray(density_m3,dtype=float)
    if t.ndim!=1 or n.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(n)) or np.any(np.diff(t)<=0): raise ValueError("Need >=8 finite samples and strictly increasing time")
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    f,p=welch(n-np.mean(n),fs=1/dt,nperseg=min(256,len(n)))
    out={"frequency_Hz":f,"power_spectral_density":p,"rms_fluctuation":float(np.std(n,ddof=1)),"relative_rms":float(np.std(n,ddof=1)/abs(np.mean(n))) if np.mean(n)!=0 else float("nan")}
    if band_Hz is not None:
        lo,hi=map(float,band_Hz)
        if lo<0 or hi<=lo: raise ValueError("Invalid frequency band")
        m=(f>=lo)&(f<=hi); out["band_power"]=float(np.trapezoid(p[m],f[m])) if np.count_nonzero(m)>1 else 0.0
    return out
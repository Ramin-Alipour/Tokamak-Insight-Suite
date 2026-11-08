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
        if not np.all(np.isfinite([lo,hi])) or lo<0 or hi<=lo: raise ValueError("Frequency band must have finite increasing non-negative bounds")
        m=(f>=lo)&(f<=hi); bin_count=int(np.count_nonzero(m)); out["band_power_bin_count"]=bin_count
        out["band_power"]=float(getattr(np, "trapezoid", np.trapz)(p[m],f[m])) if bin_count>1 else None
        if bin_count<=1: out["band_power_note"]="Insufficient spectral bins to integrate this band; zero power is not inferred."
    return out

def wave_phase_velocity(frequency_Hz, wavelength_m):
    """Phase velocity from measured frequency and wavelength: v_ph=f*lambda=omega/k."""
    f,w=np.broadcast_arrays(np.asarray(frequency_Hz,dtype=float),np.asarray(wavelength_m,dtype=float))
    if not np.all(np.isfinite(f)) or not np.all(np.isfinite(w)) or np.any(f<0) or np.any(w<=0):
        raise ValueError("Frequency must be non-negative and wavelength positive; inputs finite")
    return {"phase_velocity_m_s":f*w,"angular_frequency_rad_s":2*np.pi*f,"wavenumber_rad_m":2*np.pi/w,
            "equation":"v_phase=omega/k=f*lambda"}

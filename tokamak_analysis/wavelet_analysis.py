"""Morlet continuous wavelet transform for nonstationary plasma signals."""
import numpy as np
from scipy.signal import fftconvolve
def analyze_wavelet(time_s, signal, *, frequencies_Hz=None, wavelet_width=6.0):
    t=np.asarray(time_s,dtype=float); x=np.asarray(signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Invalid signal")
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    f=np.asarray(frequencies_Hz if frequencies_Hz is not None else np.geomspace(1/(len(t)*dt),0.45/dt,64),dtype=float)
    if f.ndim!=1 or len(f)<1 or not np.all(np.isfinite(f)) or np.any(f<=0) or np.any(f>=0.5/dt): raise ValueError("Frequencies must lie between zero and Nyquist")
    if not np.isfinite(wavelet_width) or wavelet_width<=0: raise ValueError("wavelet_width must be positive")
    centered=x-np.mean(x); coefficients=[]
    for freq in f:
        sigma=wavelet_width/(2*np.pi*freq)
        half=max(4,int(np.ceil(4*sigma/dt)))
        tau=np.arange(-half,half+1)*dt
        wave=np.exp(1j*2*np.pi*freq*tau)*np.exp(-0.5*(tau/sigma)**2)
        norm=np.sqrt(np.sum(np.abs(wave)**2))
        coefficients.append(fftconvolve(centered,np.conjugate(wave[::-1]),mode="same")/max(norm,np.finfo(float).tiny))
    coef=np.asarray(coefficients)
    return {"time_s":t,"frequencies_Hz":f,"coefficients":coef,"power":np.abs(coef)**2,"caveat":"Finite-window edge effects and wavelet-width choice affect interpretation."}

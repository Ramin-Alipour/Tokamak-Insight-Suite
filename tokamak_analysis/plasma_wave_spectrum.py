"""Frequency and spectral analysis for plasma wave time series."""
import numpy as np
from scipy.signal import welch, spectrogram
def analyze_wave_signal(time_s, signal, *, nperseg=None):
    t=np.asarray(time_s,dtype=float); y=np.asarray(signal,dtype=float)
    if t.ndim!=1 or y.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.any(np.diff(t)<=0): raise ValueError("Need matching finite signals and increasing time")
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform time sampling required")
    fs=1/dt; n=min(len(y),int(nperseg or min(256,len(y))))
    f,p=welch(y-np.mean(y),fs=fs,nperseg=n)
    f2,tt,sxx=spectrogram(y,fs=fs,nperseg=n)
    idx=int(np.argmax(p[1:])+1) if len(p)>1 else 0
    return {"frequency_Hz":f,"power_spectral_density":p,"dominant_frequency_Hz":float(f[idx]),"spectrogram_frequency_Hz":f2,"spectrogram_time_s":tt+t[0],"spectrogram_power":sxx,"sampling_frequency_Hz":fs}


def langmuir_wave_frequency(ne_m3, Te_eV, wavenumber_rad_m):
    """Warm-electron Langmuir dispersion: omega^2=omega_pe^2+3*k^2*kB*Te/me."""
    E=1.602176634e-19; ME=9.1093837139e-31; EPS0=8.8541878128e-12
    ne,te,k=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (ne_m3,Te_eV,wavenumber_rad_m)])
    if any(not np.all(np.isfinite(v)) for v in (ne,te,k)) or np.any(ne<0) or np.any(te<0) or np.any(k<0):
        raise ValueError("Density, temperature, and wavenumber must be finite and non-negative")
    omega_pe=np.sqrt(ne*E**2/(EPS0*ME))
    omega=np.sqrt(omega_pe**2+3*k**2*E*te/ME)
    return {"angular_frequency_rad_s":omega,"frequency_Hz":omega/(2*np.pi),"electron_plasma_angular_frequency_rad_s":omega_pe,
            "equation":"omega^2=omega_pe^2+3*k^2*e*Te_eV/me",
            "assumption":"Unmagnetized, weakly damped Maxwellian electron plasma in the long-wavelength fluid approximation."}

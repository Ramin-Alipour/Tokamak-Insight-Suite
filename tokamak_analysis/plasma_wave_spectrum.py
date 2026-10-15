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

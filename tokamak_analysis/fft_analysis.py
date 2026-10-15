"""FFT amplitude and phase spectrum with explicit sampling validation."""
import numpy as np
def analyze_fft(time_s, signal, *, detrend=True, window="hann"):
    t=np.asarray(time_s,dtype=float); x=np.asarray(signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Invalid time series")
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    y=x-np.mean(x) if detrend else x.copy()
    if window=="hann": w=np.hanning(len(y))
    elif window=="rectangular": w=np.ones(len(y))
    else: raise ValueError("window must be hann or rectangular")
    spec=np.fft.rfft(y*w); f=np.fft.rfftfreq(len(y),dt); amp=2*np.abs(spec)/max(np.sum(w),np.finfo(float).tiny)
    return {"frequency_Hz":f,"amplitude":amp,"phase_rad":np.angle(spec),"sampling_frequency_Hz":1/dt,"window":window}

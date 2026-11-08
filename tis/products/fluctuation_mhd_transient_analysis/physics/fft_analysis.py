"""FFT amplitude and phase spectrum with explicit sampling validation."""
import numpy as np
def analyze_fft(time_s, signal, *, detrend=True, window="hann"):
    t=np.asarray(time_s,dtype=float); x=np.asarray(signal,dtype=float)
    if t.ndim!=1 or x.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Invalid time series")
    dt=float(np.median(np.diff(t)))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    y=x-np.mean(x) if detrend else x.copy()
    if window=="hann":
        if len(y)<3: raise ValueError("Hann window requires at least 3 samples")
        w=np.hanning(len(y))
    elif window=="rectangular": w=np.ones(len(y))
    else: raise ValueError("window must be hann or rectangular")
    spec=np.fft.rfft(y*w); f=np.fft.rfftfreq(len(y),dt)
    amp=np.abs(spec)/max(np.sum(w),np.finfo(float).tiny)
    if len(amp)>1:
        amp[1:-1 if len(y)%2==0 else None] *= 2.0
    return {"frequency_Hz":f,"amplitude":amp,"phase_rad":np.angle(spec),"sampling_frequency_Hz":1/dt,"window":window}


def frequency_resolution_Hz(time_s):
    """Finite-record Fourier frequency spacing: delta_f=1/T_record."""
    t=np.asarray(time_s,dtype=float)
    if t.ndim!=1 or len(t)<2 or not np.all(np.isfinite(t)) or np.any(np.diff(t)<=0):
        raise ValueError("Time must be finite, 1-D, length >=2, strictly increasing")
    intervals=np.diff(t); dt=float(np.median(intervals))
    if not np.allclose(intervals,dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required for FFT frequency resolution")
    sample_span=float(t[-1]-t[0]); effective_duration=len(t)*dt
    return {"frequency_resolution_Hz":1.0/effective_duration,"record_duration_s":effective_duration,"sample_time_span_s":sample_span,
            "equation":"delta_f=fs/N=1/(N*dt); DFT effective record duration is N*dt"}

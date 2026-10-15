"""Reflectometry cutoff density and frequency-spectrum summaries."""
import numpy as np
EPS0=8.8541878128e-12; ME=9.1093837139e-31; E=1.602176634e-19
def cutoff_density(frequency_Hz, mode="O"):
    f=np.asarray(frequency_Hz,dtype=float)
    if not np.all(np.isfinite(f)) or np.any(f<=0): raise ValueError("Frequency must be finite and positive")
    if mode.upper()=="O": n=EPS0*ME*(2*np.pi*f)**2/E**2
    elif mode.upper()=="X": n=EPS0*ME*(2*np.pi*f)**2/E**2*0.5
    else: raise ValueError("mode must be O or simplified X")
    return {"cutoff_density_m-3":n,"mode":mode.upper(),"caveat":"X-mode cutoff depends on magnetic field and branch; the returned X-mode expression is only a rough placeholder and should not be used for scientific interpretation."}
def analyze_reflectometry_spectrum(frequency_Hz, amplitude):
    f=np.asarray(frequency_Hz,dtype=float); a=np.asarray(amplitude,dtype=float)
    if f.ndim!=1 or a.shape!=f.shape or len(f)<3 or not np.all(np.isfinite(f)) or not np.all(np.isfinite(a)) or np.any(np.diff(f)<=0): raise ValueError("Finite increasing frequency and matching amplitude required")
    idx=int(np.argmax(np.abs(a)))
    return {"peak_frequency_Hz":float(f[idx]),"peak_amplitude":float(a[idx]),"frequency_Hz":f,"amplitude":a,"caveat":"Cutoff interpretation and density-profile inversion require a propagation model and instrument geometry."}
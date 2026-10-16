"""Soft X-ray channel time-series and radiation-burst analysis."""
import numpy as np
def analyze_soft_xray(time_s, channel_signals, *, baseline_window=None):
    t=np.asarray(time_s,dtype=float); x=np.asarray(channel_signals,dtype=float)
    if t.ndim!=1 or x.ndim not in (1,2) or x.shape[0]!=len(t) or len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0): raise ValueError("Finite time and matching 1-D/2-D channels required")
    if baseline_window is None: base=np.median(x[:max(1,len(t)//10)],axis=0)
    else:
        lo,hi=baseline_window; m=(t>=lo)&(t<=hi)
        if not np.any(m): raise ValueError("No samples in baseline window")
        base=np.median(x[m],axis=0)
    signal=x-base
    return {"time_s":t,"baseline":base,"baseline_subtracted_signal":signal,"peak_by_channel":np.max(signal,axis=0),"integrated_signal_by_channel":getattr(np, "trapezoid", np.trapz)(signal,t,axis=0),"caveat":"Without detector response and line-of-sight inversion, signals are not absolute emissivity or temperature."}

def maxwellian_bremsstrahlung_spectral_shape(photon_energy_eV, electron_temperature_eV):
    """Relative non-relativistic free-free spectral shape proportional to Te^-1/2 exp(-E_gamma/Te)."""
    eg,te=np.broadcast_arrays(np.asarray(photon_energy_eV,dtype=float),np.asarray(electron_temperature_eV,dtype=float))
    if not np.all(np.isfinite(eg)) or not np.all(np.isfinite(te)) or np.any(eg<0) or np.any(te<=0):
        raise ValueError("Photon energy must be non-negative and electron temperature positive")
    shape=np.exp(-eg/te)/np.sqrt(te)
    return {"relative_bremsstrahlung_spectral_shape":shape,
            "equation":"j_E proportional to Z_eff*n_e^2*Te_eV^(-1/2)*exp(-E_gamma_eV/Te_eV)",
            "caveat":"Relative non-relativistic Maxwellian free-free shape only; Gaunt factor, detector response, impurities, line emission, and absolute normalization are omitted."}

"""X-ray spectral moments and background-subtracted spectral summaries."""
import numpy as np
def analyze_xray_spectrum(energy_keV, counts, *, background=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(e<0): raise ValueError("Finite, increasing, non-negative energy bins and matching counts required")
    if np.any(c<0): raise ValueError("Counts cannot be negative")
    net=c.copy()
    if background is not None:
        b=np.asarray(background,dtype=float)
        if b.shape!=c.shape or not np.all(np.isfinite(b)) or np.any(b<0): raise ValueError("Background invalid")
        net=np.maximum(c-b,0)
    total=np.sum(net); mean=float(np.sum(e*net)/total) if total>0 else float("nan")
    variance=float(np.sum((e-mean)**2*net)/total) if total>0 else float("nan")
    return {"energy_keV":e,"net_counts":net,"total_counts":float(total),"mean_energy_keV":mean,"energy_std_keV":float(np.sqrt(max(variance,0))) if np.isfinite(variance) else float("nan"),"peak_energy_keV":float(e[np.argmax(net)]),"caveat":"Detector response unfolding and energy calibration must be applied for physical photon spectra."}

def electron_temperature_from_xray_continuum(energy_keV, counts, *, fit_range_keV=None):
    """Estimate continuum slope using I(E)~exp(-E/Te): Te_keV=-1/slope of ln(counts)."""
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<3 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0):
        raise ValueError("Require finite increasing energies and matching non-negative counts")
    mask=c>0
    if fit_range_keV is not None:
        lo,hi=map(float,fit_range_keV)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi<=lo: raise ValueError("Invalid fit range")
        mask &= (e>=lo)&(e<=hi)
    if np.count_nonzero(mask)<3: raise ValueError("At least three positive-count bins are needed in fit range")
    slope,intercept=np.polyfit(e[mask],np.log(c[mask]),1)
    if slope>=0: raise ValueError("Continuum log-slope must be negative to infer positive temperature")
    return {"electron_temperature_keV":float(-1/slope),"log_spectrum_slope_per_keV":float(slope),"fit_intercept":float(intercept),"fit_bin_count":int(np.count_nonzero(mask)),
            "equation":"ln(I_E)=constant-E/Te_keV; Te_keV=-1/(d ln I_E/dE)",
            "caveat":"Simplified exponential continuum fit; detector response, line emission, background, and relativistic corrections can bias temperature."}

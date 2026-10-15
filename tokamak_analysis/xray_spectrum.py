"""X-ray spectral moments and background-subtracted spectral summaries."""
import numpy as np
def analyze_xray_spectrum(energy_keV, counts, *, background=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0): raise ValueError("Finite increasing energy bins and matching counts required")
    if np.any(c<0): raise ValueError("Counts cannot be negative")
    net=c.copy()
    if background is not None:
        b=np.asarray(background,dtype=float)
        if b.shape!=c.shape or not np.all(np.isfinite(b)) or np.any(b<0): raise ValueError("Background invalid")
        net=np.maximum(c-b,0)
    total=np.sum(net); mean=float(np.sum(e*net)/total) if total>0 else float("nan")
    variance=float(np.sum((e-mean)**2*net)/total) if total>0 else float("nan")
    return {"energy_keV":e,"net_counts":net,"total_counts":float(total),"mean_energy_keV":mean,"energy_std_keV":float(np.sqrt(max(variance,0))) if np.isfinite(variance) else float("nan"),"peak_energy_keV":float(e[np.argmax(net)]),"caveat":"Detector response unfolding and energy calibration must be applied for physical photon spectra."}
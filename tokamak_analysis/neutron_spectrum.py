"""Neutron energy-spectrum moments and peak extraction."""
import numpy as np
def analyze_neutron_spectrum(energy_keV, counts, *, background=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0): raise ValueError("Invalid energy spectrum")
    net=c.copy()
    if background is not None:
        b=np.asarray(background,dtype=float)
        if b.shape!=c.shape or not np.all(np.isfinite(b)) or np.any(b<0): raise ValueError("Invalid background")
        net=np.maximum(c-b,0)
    total=np.sum(net); mean=float(np.sum(e*net)/total) if total else float("nan")
    return {"energy_keV":e,"net_counts":net,"peak_energy_keV":float(e[np.argmax(net)]),"mean_energy_keV":mean,"total_counts":float(total),"caveat":"Detector unfolding and neutron-response calibration are required for a physical incident neutron spectrum."}
"""Binned energetic-particle distribution and loss/flux estimates."""
import numpy as np
def analyze_energy_distribution(energy_keV, counts, *, normalize=True):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(e<0) or np.any(c<0): raise ValueError("Energy and counts must be finite, nonnegative 1-D arrays")
    total=float(np.sum(c)); dist=c/total if normalize and total>0 else c.copy()
    return {"energy_keV":e,"distribution":dist,"total_counts":total,"mean_energy_keV":float(np.sum(e*c)/total) if total>0 else float("nan"),"peak_energy_keV":float(e[np.argmax(c)]),"normalized":bool(normalize and total>0)}
def particle_loss_rate(time_s, loss_counts, *, detection_efficiency=1.0):
    t=np.asarray(time_s,dtype=float); c=np.asarray(loss_counts,dtype=float); eff=float(detection_efficiency)
    if t.ndim!=1 or c.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)) or np.any(np.diff(t)<=0) or np.any(c<0) or not 0<eff<=1: raise ValueError("Invalid time/counts/efficiency")
    return {"loss_rate_s-1":np.gradient(c,t,edge_order=2 if len(t)>2 else 1)/eff,"total_detected_counts":float(np.sum(c)),"caveat":"Absolute particle loss requires detector efficiency, acceptance, and background corrections."}
def energetic_particle_flux(time_s, counts, effective_area_m2, *, efficiency=1.0):
    t=np.asarray(time_s,dtype=float); c=np.asarray(counts,dtype=float); area=float(effective_area_m2); eff=float(efficiency)
    if t.ndim!=1 or c.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)) or np.any(np.diff(t)<=0) or np.any(c<0) or area<=0 or not 0<eff<=1: raise ValueError("Invalid inputs")
    return {"particle_flux_m-2_s-1":np.gradient(c,t,edge_order=2 if len(t)>2 else 1)/(area*eff),"caveat":"Flux is a detector-normalized estimate and requires validated geometry and efficiency."}
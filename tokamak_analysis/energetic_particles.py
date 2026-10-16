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

def relativistic_kinetic_energy_keV(momentum_kg_m_s, particle_mass_kg):
    """Relativistic kinetic energy K=sqrt((pc)^2+(mc^2)^2)-mc^2."""
    p,m=np.broadcast_arrays(np.asarray(momentum_kg_m_s,dtype=float),np.asarray(particle_mass_kg,dtype=float))
    c=299792458.0; joule_per_keV=1.602176634e-16
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(m)) or np.any(p<0) or np.any(m<=0):
        raise ValueError("Momentum must be non-negative and mass positive; inputs finite")
    rest=m*c*c; total=np.hypot(p*c,rest)
    # Stable form avoids cancellation at low momentum.
    kinetic=(p*c)**2/(total+rest)
    return {"kinetic_energy_keV":kinetic/joule_per_keV,"gamma_factor":total/rest,
            "equation":"K=sqrt((p*c)^2+(m*c^2)^2)-m*c^2"}


def exponential_confinement_loss_rate(particle_inventory, confinement_time_s):
    """First-order loss model: instantaneous loss rate Gamma_loss=N/tau_p."""
    n,tau=np.broadcast_arrays(np.asarray(particle_inventory,dtype=float),np.asarray(confinement_time_s,dtype=float))
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(tau)) or np.any(n<0) or np.any(tau<=0):
        raise ValueError("Inventory non-negative and confinement time positive; inputs finite")
    return {"particle_loss_rate_s-1":n/tau,"equation":"Gamma_loss=N_particles/tau_p",
            "assumption":"Single exponential particle-loss model with constant confinement time."}

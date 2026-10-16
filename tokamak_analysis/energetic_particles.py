"""Binned energetic-particle distribution and loss/flux estimates."""
import numpy as np
def analyze_energy_distribution(energy_keV, counts, *, normalize=True):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(e<0) or np.any(c<0): raise ValueError("Energy and counts must be finite, nonnegative 1-D arrays")
    total=float(np.sum(c)); dist=c/total if normalize and total>0 else c.copy()
    return {"energy_keV":e,"distribution":dist,"total_counts":total,"mean_energy_keV":float(np.sum(e*c)/total) if total>0 else float("nan"),"peak_energy_keV":float(e[np.argmax(c)]) if total>0 else None,"normalized":bool(normalize and total>0)}
def particle_loss_rate(time_s, loss_counts, *, detection_efficiency=1.0, live_time_s=None, counts_are_cumulative=False):
    t=np.asarray(time_s,dtype=float); c=np.asarray(loss_counts,dtype=float); eff=float(detection_efficiency)
    if t.ndim!=1 or c.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)) or np.any(np.diff(t)<=0) or np.any(c<0) or not np.isfinite(eff) or not 0<eff<=1:
        raise ValueError("Invalid time/counts/efficiency")
    if counts_are_cumulative:
        if live_time_s is not None or np.any(np.diff(c)<0):
            raise ValueError("Cumulative counts must be nondecreasing and cannot use live_time_s")
        rate=np.gradient(c,t,edge_order=2 if len(t)>2 else 1)
        total=float(c[-1]-c[0])
        method="derivative of cumulative counter"
    else:
        width=np.gradient(t) if live_time_s is None else np.asarray(live_time_s,dtype=float)
        if not np.all(np.isfinite(width)) or np.any(width<=0):
            raise ValueError("Live times/bin widths must be finite and positive")
        try: width=np.broadcast_to(width,c.shape)
        except ValueError as exc: raise ValueError("live_time_s must be scalar or match loss_counts") from exc
        rate=c/width
        total=float(np.sum(c))
        method="counts divided by supplied live time" if live_time_s is not None else "counts divided by estimated time-bin width"
    return {"loss_rate_s-1":rate/eff,"total_detected_counts":total,"rate_method":method,
            "caveat":"Absolute particle loss requires detector efficiency, acceptance, background and dead-time corrections."}


def energetic_particle_flux(time_s, counts, effective_area_m2, *, efficiency=1.0, live_time_s=None, counts_are_cumulative=False):
    t=np.asarray(time_s,dtype=float); c=np.asarray(counts,dtype=float); area=float(effective_area_m2); eff=float(efficiency)
    if t.ndim!=1 or c.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)) or np.any(np.diff(t)<=0) or np.any(c<0) or not np.isfinite(area) or area<=0 or not np.isfinite(eff) or not 0<eff<=1:
        raise ValueError("Invalid time, counts, area or efficiency")
    if counts_are_cumulative:
        if live_time_s is not None or np.any(np.diff(c)<0):
            raise ValueError("Cumulative counts must be nondecreasing and cannot use live_time_s")
        rate=np.gradient(c,t,edge_order=2 if len(t)>2 else 1)
        method="derivative of cumulative counter"
    else:
        width=np.gradient(t) if live_time_s is None else np.asarray(live_time_s,dtype=float)
        if not np.all(np.isfinite(width)) or np.any(width<=0):
            raise ValueError("Live times/bin widths must be finite and positive")
        try: width=np.broadcast_to(width,c.shape)
        except ValueError as exc: raise ValueError("live_time_s must be scalar or match counts") from exc
        rate=c/width
        method="counts divided by supplied live time" if live_time_s is not None else "counts divided by estimated time-bin width"
    return {"particle_flux_m-2_s-1":rate/(area*eff),"count_rate_method":method,
            "caveat":"Flux requires validated detector geometry, efficiency, background, dead-time and response corrections."}

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

"""Divertor heat/particle flux and plasma-facing component load metrics."""
import numpy as np
def analyze_divertor_flux(position_m, heat_flux_W_m2, *, particle_flux_m2_s=None, surface_area_m2=None):
    x=np.asarray(position_m,dtype=float); q=np.asarray(heat_flux_W_m2,dtype=float)
    if x.ndim!=1 or q.shape!=x.shape or len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)) or np.any(np.diff(x)<=0): raise ValueError("Invalid divertor profile")
    out={"position_m":x,"heat_flux_W_m2":q,"peak_heat_flux_W_m2":float(np.max(q)),"integrated_heat_flux_W_m":float(np.trapezoid(q,x))}
    if particle_flux_m2_s is not None:
        g=np.asarray(particle_flux_m2_s,dtype=float)
        if g.shape!=x.shape or not np.all(np.isfinite(g)) or np.any(g<0): raise ValueError("Particle flux must match and be nonnegative")
        out["particle_flux_m2_s"]=g; out["integrated_particle_flux_m_s-1"]=float(np.trapezoid(g,x))
    if surface_area_m2 is not None:
        area=float(surface_area_m2)
        if area<=0 or not np.isfinite(area): raise ValueError("Area must be positive")
        out["peak_power_load_W"]=out["peak_heat_flux_W_m2"]*area
    return out
def sputtering_yield(incident_ions_s, sputtered_atoms_s):
    inc=float(incident_ions_s); sput=float(sputtered_atoms_s)
    if not np.isfinite(inc) or not np.isfinite(sput) or inc<=0 or sput<0: raise ValueError("Incident rate positive and sputtered rate nonnegative")
    return {"sputtering_yield_atoms_per_ion":sput/inc}
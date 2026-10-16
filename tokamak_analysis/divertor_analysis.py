"""Divertor heat/particle flux and plasma-facing component load metrics."""
import numpy as np
def analyze_divertor_flux(position_m, heat_flux_W_m2, *, particle_flux_m2_s=None, surface_area_m2=None):
    x=np.asarray(position_m,dtype=float); q=np.asarray(heat_flux_W_m2,dtype=float)
    if x.ndim!=1 or q.shape!=x.shape or len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)) or np.any(np.diff(x)<=0) or np.any(q<0): raise ValueError("Divertor heat flux must be finite, non-negative, and ordered by increasing position")
    out={"position_m":x,"heat_flux_W_m2":q,"peak_heat_flux_W_m2":float(np.max(q)),"integrated_heat_flux_W_m":float(getattr(np, "trapezoid", np.trapz)(q,x))}
    if particle_flux_m2_s is not None:
        g=np.asarray(particle_flux_m2_s,dtype=float)
        if g.shape!=x.shape or not np.all(np.isfinite(g)) or np.any(g<0): raise ValueError("Particle flux must match and be nonnegative")
        out["particle_flux_m-2_s-1"]=g; out["integrated_particle_flux_m-1_s-1"]=float(getattr(np, "trapezoid", np.trapz)(g,x))
    if surface_area_m2 is not None:
        area=float(surface_area_m2)
        if area<=0 or not np.isfinite(area): raise ValueError("Area must be positive")
        out["peak_power_load_W"]=out["peak_heat_flux_W_m2"]*area
    return out
def sputtering_yield(incident_ions_s, sputtered_atoms_s):
    inc=float(incident_ions_s); sput=float(sputtered_atoms_s)
    if not np.isfinite(inc) or not np.isfinite(sput) or inc<=0 or sput<0: raise ValueError("Incident rate positive and sputtered rate nonnegative")
    return {"sputtering_yield_atoms_per_ion":sput/inc}

def sheath_heat_flux_to_divertor(ne_m3, Te_eV, ion_mass_amu, sheath_heat_transmission, Bp_T, Btotal_T):
    """Approximate target heat flux q_t=gamma_s*n_e*Te*e*c_s*(B_p/B), c_s=sqrt(e*Te/m_i)."""
    ne,te,mass,gamma,bp,bt=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (ne_m3,Te_eV,ion_mass_amu,sheath_heat_transmission,Bp_T,Btotal_T)])
    if any(not np.all(np.isfinite(v)) for v in (ne,te,mass,gamma,bp,bt)) or np.any(ne<0) or np.any(te<0) or np.any(mass<=0) or np.any(gamma<0) or np.any(bp<0) or np.any(bt<=0) or np.any(bp>bt):
        raise ValueError("Invalid density, temperature, mass, sheath factor or magnetic fields")
    e=1.602176634e-19; u=1.66053906660e-27
    cs=np.sqrt(e*te/(mass*u)); qpar=gamma*ne*te*e*cs; qtarget=qpar*bp/bt
    return {"sound_speed_m_s":cs,"parallel_sheath_heat_flux_W_m2":qpar,"target_heat_flux_W_m2":qtarget,
            "equation":"q_target~=gamma_s*n_e*Te_eV*e*c_s*(B_p/B), c_s=sqrt(e*Te_eV/m_i)",
            "assumption":"Simplified sheath-limited mapping; neglects Ti contribution, flux expansion geometry, radiation and cross-field transport."}


def total_pfc_power_from_surface_profile(position_m, heat_flux_W_m2, toroidal_radius_m):
    """Axisymmetric surface power integral P=integral q(s)*2*pi*R(s) ds."""
    s=np.asarray(position_m,dtype=float); q=np.asarray(heat_flux_W_m2,dtype=float); R=np.asarray(toroidal_radius_m,dtype=float)
    if s.ndim!=1 or q.shape!=s.shape or R.shape!=s.shape or len(s)<2 or any(not np.all(np.isfinite(v)) for v in (s,q,R)) or np.any(np.diff(s)<=0) or np.any(q<0) or np.any(R<=0):
        raise ValueError("Require matching finite profiles, increasing position, nonnegative heat flux and positive radius")
    return {"total_pfc_power_W":float(getattr(np, "trapezoid", np.trapz)(q*2*np.pi*R,s)),"equation":"P=integral q(s)*2*pi*R(s) ds"}


def divertor_ion_particle_flux_from_sheath(ion_density_m3, electron_temperature_eV, ion_mass_amu, *, ion_temperature_eV=0.0, ion_thermal_factor=1.0):
    """Approximate Bohm-sheath particle flux: Gamma_i=n_i*c_s, c_s=sqrt(e*(Te+gamma_i*Ti)/m_i)."""
    ni,te,mass,ti,gamma=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (ion_density_m3,electron_temperature_eV,ion_mass_amu,ion_temperature_eV,ion_thermal_factor)])
    if any(not np.all(np.isfinite(v)) for v in (ni,te,mass,ti,gamma)) or np.any(ni<0) or np.any(te<0) or np.any(ti<0) or np.any(mass<=0) or np.any(gamma<0):
        raise ValueError("Density and temperatures must be non-negative; mass positive; thermal factor non-negative; inputs finite")
    cs=np.sqrt((te+gamma*ti)*1.602176634e-19/(mass*1.66053906660e-27))
    return {"sound_speed_m_s":cs,"divertor_ion_particle_flux_m-2_s-1":ni*cs,
            "equation":"Gamma_i~=n_i*c_s; c_s=sqrt(e*(Te_eV+gamma_i*Ti_eV)/m_i)",
            "assumption":"Approximate Bohm-sheath entry speed; gamma_i sets the assumed ion thermal contribution. Use Te-only (Ti=0) only when the cold-ion approximation is justified; geometry and sheath transmission remain separate."}


def analyze_divertor_particle_flux(position_m, particle_flux_m2_s):
    """Integrate a calibrated divertor particle-flux profile independently of heat-flux inputs."""
    x=np.asarray(position_m,dtype=float); g=np.asarray(particle_flux_m2_s,dtype=float)
    if x.ndim!=1 or g.shape!=x.shape or len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(g)) or np.any(np.diff(x)<=0) or np.any(g<0):
        raise ValueError("Position must be finite/increasing and particle flux finite, non-negative, and shape-matched")
    return {"position_m":x,"particle_flux_m-2_s-1":g,
            "peak_particle_flux_m-2_s-1":float(np.max(g)),
            "integrated_particle_flux_m-1_s-1":float(getattr(np, "trapezoid", np.trapz)(g,x)),
            "caveat":"Integrated profile is per unit toroidal length unless surface geometry is included; absolute particle rate needs the relevant area and geometry."}


def plasma_facing_component_heat_load(position_m, heat_flux_W_m2, surface_area_m2):
    """Estimate peak heat load from a measured flux profile and specified exposed area."""
    area=float(surface_area_m2)
    if not np.isfinite(area) or area<=0:
        raise ValueError("surface_area_m2 must be finite and positive")
    result=analyze_divertor_flux(position_m, heat_flux_W_m2, surface_area_m2=area)
    result["surface_area_m2"]=area
    result["caveat"]="Peak flux multiplied by the supplied exposed area is a peak-based estimate, not an area-integrated load for a nonuniform surface."
    return result

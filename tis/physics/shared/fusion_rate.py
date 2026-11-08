"""Fusion reaction-rate estimates from density and reactivity inputs."""
import numpy as np
def fusion_reaction_rate(density_a_m3,density_b_m3,reactivity_m3_s,*,identical_species=False,volume_m3=None):
    na=np.asarray(density_a_m3,dtype=float); nb=np.asarray(density_b_m3,dtype=float); sv=np.asarray(reactivity_m3_s,dtype=float)
    if any(not np.all(np.isfinite(x)) for x in (na,nb,sv)) or np.any(na<0) or np.any(nb<0) or np.any(sv<0): raise ValueError("Densities/reactivity must be finite and nonnegative")
    rate=na*nb*sv/(2 if identical_species else 1)
    out={"reaction_rate_density_m-3_s-1":rate}
    if volume_m3 is not None:
        V=float(volume_m3)
        if not np.isfinite(V) or V<=0: raise ValueError("Volume must be positive")
        out["total_reactions_s-1"]=rate*V
    out["caveat"]="Reactivity must correspond to the correct ion distributions and reaction channel; spatially varying profiles require volume integration."
    return out

def fusion_power_from_reaction_rate(reaction_rate_density_m3_s, volume_m3, energy_release_MeV):
    """Fusion power: P=R_density*V*E_release, with MeV converted to joules."""
    rate,vol,energy=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (reaction_rate_density_m3_s,volume_m3,energy_release_MeV)])
    if any(not np.all(np.isfinite(v)) for v in (rate,vol,energy)) or np.any(rate<0) or np.any(vol<=0) or np.any(energy<0):
        raise ValueError("Rate and energy must be non-negative, volume positive, all finite")
    power=rate*vol*energy*1.602176634e-13
    return {"fusion_power_W":power,"total_reaction_rate_s-1":rate*vol,
            "equation":"P_fusion=R_density*V*E_release_MeV*1.602176634e-13 J/MeV",
            "caveat":"Assumes uniform reaction-rate density and that energy_release refers to the intended product energy accounting."}

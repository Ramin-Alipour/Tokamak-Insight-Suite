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
"""Plasma/floating potential summaries and electric-field derivation."""
import numpy as np
def analyze_potentials(potential_V, floating_potential_V, *, coordinate_m=None):
    vp=np.asarray(potential_V,dtype=float); vf=np.asarray(floating_potential_V,dtype=float)
    if not np.all(np.isfinite(vp)) or not np.all(np.isfinite(vf)): raise ValueError("Potentials must be finite")
    vp,vf=np.broadcast_arrays(vp,vf); out={"plasma_potential_V":vp,"floating_potential_V":vf,"difference_V":vp-vf,"mean_difference_V":float(np.mean(vp-vf))}
    if coordinate_m is not None:
        x=np.asarray(coordinate_m,dtype=float)
        if vp.ndim!=1 or x.shape!=vp.shape or np.any(np.diff(x)<=0) or not np.all(np.isfinite(x)): raise ValueError("Coordinate must match 1-D potential profile and increase")
        out["electric_field_V_m"]=-np.gradient(vp,x,edge_order=2 if len(x)>2 else 1)
    out["caveat"]="Plasma potential must be measured or inferred by a justified diagnostic model; it is not interchangeable with floating potential."
    return out
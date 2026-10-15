"""Edge potential gradients and electric-field profiles."""
import numpy as np
def analyze_edge_field(position_m, potential_V, *, floating_potential_V=None):
    x=np.asarray(position_m,dtype=float); v=np.asarray(potential_V,dtype=float)
    if x.ndim!=1 or v.shape!=x.shape or len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(v)) or np.any(np.diff(x)<=0): raise ValueError("Invalid edge potential profile")
    out={"position_m":x,"potential_V":v,"electric_field_V_m":-np.gradient(v,x,edge_order=2 if len(x)>2 else 1)}
    if floating_potential_V is not None:
        vf=np.asarray(floating_potential_V,dtype=float)
        if vf.shape!=v.shape or not np.all(np.isfinite(vf)): raise ValueError("Floating potential must match")
        out["floating_potential_V"]=vf; out["plasma_minus_floating_V"]=v-vf
    return out
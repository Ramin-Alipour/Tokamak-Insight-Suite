"""Safety-factor estimates from supplied equilibrium geometry and fields."""
import numpy as np
def estimate_safety_factor(R_m, minor_radius_m, B_toroidal_T, B_poloidal_T, *, q_profile=None):
    R=np.asarray(R_m,dtype=float); a=np.asarray(minor_radius_m,dtype=float); bt=np.asarray(B_toroidal_T,dtype=float); bp=np.asarray(B_poloidal_T,dtype=float)
    if any(not np.all(np.isfinite(x)) for x in (R,a,bt,bp)) or np.any(R<=0) or np.any(a<=0) or np.any(bp<=0): raise ValueError("Geometry and field values must be finite; R, a and Bp positive")
    q=R*bt/(a*bp)
    out={"q_cylindrical_estimate":q,"caveat":"Large-aspect-ratio cylindrical estimate only; accurate q(r) requires equilibrium flux-surface integration."}
    if q_profile is not None:
        qp=np.asarray(q_profile,dtype=float)
        if qp.ndim!=1 or not np.all(np.isfinite(qp)): raise ValueError("q_profile must be a finite 1-D array")
        out["provided_q_profile"]=qp; out["q_min_provided"]=float(np.min(qp)); out["q95_provided"]=float(np.interp(.95,np.linspace(0,1,len(qp)),qp))
    return out
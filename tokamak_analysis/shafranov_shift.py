"""Shafranov-shift geometric estimate."""
import numpy as np
def shafranov_shift(R_axis_m,R_geometric_m,*,minor_radius_m=None,elongation=None,beta_p=None,li=None):
    vals=[R_axis_m,R_geometric_m]
    if not all(np.isfinite(v) for v in vals) or R_axis_m<=0 or R_geometric_m<=0: raise ValueError("Radii must be finite and positive")
    out={"measured_axis_shift_m":float(R_axis_m-R_geometric_m),"absolute_shift_m":float(abs(R_axis_m-R_geometric_m))}
    if minor_radius_m is not None:
        a=float(minor_radius_m)
        if not np.isfinite(a) or a<=0: raise ValueError("minor_radius_m must be positive")
        out["normalized_axis_shift"]=out["measured_axis_shift_m"]/a
    if beta_p is not None and li is not None:
        bp=float(beta_p); internal=float(li)
        if not np.isfinite(bp) or not np.isfinite(internal): raise ValueError("beta_p and li must be finite")
        out["large_aspect_ratio_circular_prediction_m"]=minor_radius_m**2/(2*R_geometric_m)*(bp+internal/2) if minor_radius_m is not None else None
    out["caveat"]="The radius difference is a geometric estimate; analytic prediction assumes simplified equilibrium geometry."
    return out
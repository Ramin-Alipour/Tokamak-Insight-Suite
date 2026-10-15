"""Spectral line-ratio calculations with uncertainty propagation."""
import numpy as np
def analyze_line_ratio(intensity_numerator,intensity_denominator,*,uncertainty_numerator=None,uncertainty_denominator=None):
    a=np.asarray(intensity_numerator,dtype=float); b=np.asarray(intensity_denominator,dtype=float)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)) or np.any(a<0) or np.any(b<=0): raise ValueError("Numerator nonnegative and denominator positive; all finite")
    a,b=np.broadcast_arrays(a,b); ratio=a/b; out={"line_ratio":ratio}
    if uncertainty_numerator is not None or uncertainty_denominator is not None:
        if uncertainty_numerator is None or uncertainty_denominator is None: raise ValueError("Provide both intensity uncertainties")
        sa=np.asarray(uncertainty_numerator,dtype=float); sb=np.asarray(uncertainty_denominator,dtype=float)
        if np.any(sa<0) or np.any(sb<0) or not np.all(np.isfinite(sa)) or not np.all(np.isfinite(sb)): raise ValueError("Uncertainties must be finite and non-negative")
        sa,sb=np.broadcast_arrays(sa,sb); out["line_ratio_uncertainty"]=np.abs(ratio)*np.sqrt((sa/np.maximum(a,np.finfo(float).tiny))**2+(sb/b)**2)
    out["caveat"]="Physical interpretation requires line calibration, background subtraction, optical-depth assessment and atomic-rate modeling."
    return out
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

def boltzmann_temperature_from_line_ratio(intensity_ratio, line_prefactor, upper_energy_difference_eV):
    """Two-line Boltzmann estimate: R=C*exp(-DeltaE/Te), hence Te=DeltaE/ln(C/R)."""
    ratio,pref,delta=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (intensity_ratio,line_prefactor,upper_energy_difference_eV)])
    if any(not np.all(np.isfinite(v)) for v in (ratio,pref,delta)) or np.any(ratio<=0) or np.any(pref<=0) or np.any(delta<=0) or np.any(pref<=ratio):
        raise ValueError("Require finite positive ratio, prefactor and energy difference, with prefactor > ratio")
    te=delta/np.log(pref/ratio)
    return {"excitation_temperature_eV":te,"equation":"Te_eV=DeltaE_eV/ln(C_line/R)",
            "assumption":"Optically thin line emission and a Boltzmann population model; not generally valid for non-LTE plasmas without collisional-radiative modeling."}

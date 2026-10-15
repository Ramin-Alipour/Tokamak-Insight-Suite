"""Impurity concentration estimates from calibrated line emissivities."""
import numpy as np
def estimate_impurity_fraction(impurity_line_emissivity, impurity_excitation_coefficient, electron_density_m3, *, hydrogen_line_emissivity=None, hydrogen_excitation_coefficient=None):
    Lz=np.asarray(impurity_line_emissivity,dtype=float); Qz=np.asarray(impurity_excitation_coefficient,dtype=float); ne=np.asarray(electron_density_m3,dtype=float)
    if any(not np.all(np.isfinite(x)) for x in (Lz,Qz,ne)) or np.any(Lz<0) or np.any(Qz<=0) or np.any(ne<=0): raise ValueError("Emissivity must be nonnegative, coefficients and density positive")
    out={"impurity_density_estimate_m-3":Lz/(ne*Qz)}
    if hydrogen_line_emissivity is not None or hydrogen_excitation_coefficient is not None:
        if hydrogen_line_emissivity is None or hydrogen_excitation_coefficient is None: raise ValueError("Provide both hydrogen emissivity and coefficient")
        Lh=np.asarray(hydrogen_line_emissivity,dtype=float); Qh=np.asarray(hydrogen_excitation_coefficient,dtype=float)
        if np.any(Lh<0) or np.any(Qh<=0) or not np.all(np.isfinite(Lh)) or not np.all(np.isfinite(Qh)): raise ValueError("Hydrogen inputs invalid")
        nz=Lz/(ne*Qz); nh=Lh/(ne*Qh); out["impurity_fraction_estimate"]=nz/nh
    out["caveat"]="Requires calibrated optically thin emissivity and appropriate collisional-radiative coefficients; this simplified estimator omits transport and charge-state balance."
    return out
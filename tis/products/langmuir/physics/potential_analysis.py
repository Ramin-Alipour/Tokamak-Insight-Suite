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

def floating_potential_from_saturation_currents(plasma_potential_V, electron_temperature_eV, electron_saturation_current_A, ion_saturation_current_A):
    """Current-balance relation Vf=Vp-Te*ln(Ies/Iis), with Te expressed in eV."""
    vp,te,ies,iis=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (plasma_potential_V,electron_temperature_eV,electron_saturation_current_A,ion_saturation_current_A)])
    if any(not np.all(np.isfinite(v)) for v in (vp,te,ies,iis)) or np.any(te<=0) or np.any(ies<=0) or np.any(iis<=0):
        raise ValueError("All inputs must be finite; Te and saturation-current magnitudes must be positive")
    vf=vp-te*np.log(ies/iis)
    return {"floating_potential_V":vf,"equation":"Vf=Vp-Te_eV*ln(|Ies|/|Iis|)",
            "assumption":"Steady floating-current balance with Maxwellian electron retarding current and fixed ion-saturation current."}

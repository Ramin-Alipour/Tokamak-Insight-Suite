"""Magnetic-field metrics from calibrated toroidal field and equilibrium poloidal field."""
import numpy as np

def analyze_magnetic_field(B_toroidal_T, B_poloidal_T, *, R_m=None, B_radial_T=None, B_vertical_T=None):
    """Calculate total field magnitude and pitch angle for scalar or array inputs.

    Inputs must already be calibrated field components; no coil inversion is performed.
    """
    bt=np.asarray(B_toroidal_T,dtype=float); bp=np.asarray(B_poloidal_T,dtype=float)
    if not np.all(np.isfinite(bt)) or not np.all(np.isfinite(bp)): raise ValueError("Field inputs must be finite")
    bt,bp=np.broadcast_arrays(bt,bp); total=np.hypot(bt,bp)
    pitch=np.arctan2(bp,bt)
    out={"B_toroidal_T":bt,"B_poloidal_T":bp,"B_total_T":total,"field_pitch_rad":pitch,"field_pitch_deg":np.degrees(pitch),"max_total_field_T":float(np.max(total)),"rms_total_field_T":float(np.sqrt(np.mean(total**2))),"physical_relations":{"field_magnitude":"|B|=sqrt(Bt^2+Bp^2)","field_pitch":"alpha=atan2(Bp,Bt)","vacuum_toroidal_scaling":"Bt(R)*R=constant; Bt(R)=Bt_ref*R_ref/R (large-aspect-ratio vacuum approximation)"}}
    if R_m is not None:
        R=np.asarray(R_m,dtype=float)
        if np.any(~np.isfinite(R)) or np.any(R<=0): raise ValueError("R_m must be finite and positive")
        out["R_m"]=R.copy()
        if np.asarray(bt).ndim==0 and R.ndim>0: out["B_toroidal_R_scaled_T"]=bt*R[0]/R
    if B_radial_T is not None or B_vertical_T is not None:
        if B_radial_T is None or B_vertical_T is None: raise ValueError("Provide both radial and vertical components")
        br=np.asarray(B_radial_T,dtype=float); bz=np.asarray(B_vertical_T,dtype=float)
        if not np.all(np.isfinite(br)) or not np.all(np.isfinite(bz)): raise ValueError("Poloidal components must be finite")
        out["B_poloidal_from_components_T"]=np.hypot(br,bz)
    out["caveat"]="Toroidal-field R-scaling is only a large-aspect-ratio vacuum approximation; do not apply it to arbitrary shaped-plasma fields without justification."
    return out

def toroidal_field_vacuum_scaling(Bt_ref_T, R_ref_m, R_m):
    """Large-aspect-ratio vacuum toroidal-field relation Bt(R)=Bt_ref*R_ref/R."""
    bt0=float(Bt_ref_T); r0=float(R_ref_m); r=np.asarray(R_m,dtype=float)
    if not np.isfinite(bt0) or not np.isfinite(r0) or r0<=0:
        raise ValueError("Reference field must be finite and reference radius positive")
    if not np.all(np.isfinite(r)) or np.any(r<=0):
        raise ValueError("R_m must contain finite positive major radii")
    bt=bt0*r0/r
    return {"R_m":r.copy(),"B_toroidal_T":bt,
            "equation":"Bt(R)*R=constant; Bt(R)=Bt_ref*R_ref/R",
            "Bt_times_R_T_m":bt*r,
            "caveat":"Vacuum 1/R scaling; plasma currents, coils, ferromagnetic structures, and non-ideal geometry can alter the field."}

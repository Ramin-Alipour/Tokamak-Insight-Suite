"""Multi-pin probe channel synchronization, potential gradients and E-field estimate."""
import numpy as np
def analyze_multipin_probe(time_s, potentials_V, positions_m, *, axis="poloidal"):
    t=np.asarray(time_s,dtype=float); v=np.asarray(potentials_V,dtype=float); x=np.asarray(positions_m,dtype=float)
    if t.ndim!=1 or v.ndim!=2 or v.shape[0]!=len(t) or v.shape[1]!=len(x) or len(x)<2: raise ValueError("Potentials must have shape (time, pins) and match positions")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or not np.all(np.isfinite(x)) or np.any(np.diff(t)<=0) or np.any(np.diff(x)<=0): raise ValueError("Finite data and strictly increasing time/position required")
    grad=np.gradient(v,x,axis=1,edge_order=2 if len(x)>2 else 1)
    return {"time_s":t,"positions_m":x,"potential_V":v,"potential_gradient_V_m":grad,"electric_field_V_m":-grad,"axis":axis,"mean_potential_by_pin_V":np.mean(v,axis=0),"caveat":"Electric-field estimate assumes simultaneous calibrated potentials and known pin coordinates; sheath effects and spatial resolution remain."}

def exb_drift_velocity(E_x_V_m,E_y_V_m,E_z_V_m,B_x_T,B_y_T,B_z_T):
    """Compute ideal E-cross-B drift v=E×B/B^2, perpendicular to B."""
    ex,ey,ez,bx,by,bz=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (E_x_V_m,E_y_V_m,E_z_V_m,B_x_T,B_y_T,B_z_T)])
    if any(not np.all(np.isfinite(v)) for v in (ex,ey,ez,bx,by,bz)): raise ValueError("Field components must be finite")
    b2=bx*bx+by*by+bz*bz
    if np.any(b2<=0): raise ValueError("Magnetic field magnitude must be non-zero")
    vx=(ey*bz-ez*by)/b2; vy=(ez*bx-ex*bz)/b2; vz=(ex*by-ey*bx)/b2
    return {"v_exb_x_m_s":vx,"v_exb_y_m_s":vy,"v_exb_z_m_s":vz,
            "equation":"v_E=E×B/|B|^2",
            "caveat":"Ideal guiding-center E-cross-B drift; requires colocated calibrated E and B components."}

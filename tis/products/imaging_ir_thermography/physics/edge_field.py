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

def debye_length_m(electron_density_m3, electron_temperature_eV):
    """Electron Debye length lambda_D=sqrt(epsilon0*kB*Te[K]/(ne*e^2))=sqrt(epsilon0*Te[eV]*e/(ne*e^2))."""
    ne,te=np.broadcast_arrays(np.asarray(electron_density_m3,dtype=float),np.asarray(electron_temperature_eV,dtype=float))
    if not np.all(np.isfinite(ne)) or not np.all(np.isfinite(te)) or np.any(ne<=0) or np.any(te<0):
        raise ValueError("Electron density must be positive and temperature non-negative; inputs finite")
    E=1.602176634e-19; EPS0=8.8541878128e-12
    return {"debye_length_m":np.sqrt(EPS0*te/(ne*E)),"equation":"lambda_D=sqrt(epsilon0*Te_eV/(ne*e))"}

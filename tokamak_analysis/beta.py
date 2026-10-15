"""Plasma beta calculations from measured pressure and magnetic field."""
import numpy as np
MU0=4e-7*np.pi
def analyze_beta(pressure_Pa, magnetic_field_T, *, electron_pressure_Pa=None, ion_pressure_Pa=None):
    p=np.asarray(pressure_Pa,dtype=float); b=np.asarray(magnetic_field_T,dtype=float)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(b)) or np.any(p<0) or np.any(b<=0): raise ValueError("Pressure must be finite/non-negative and field finite/positive")
    beta=2*MU0*p/b**2
    out={"beta":beta,"beta_percent":100*beta,"pressure_Pa":p,"magnetic_field_T":b}
    if electron_pressure_Pa is not None or ion_pressure_Pa is not None:
        if electron_pressure_Pa is None or ion_pressure_Pa is None: raise ValueError("Provide both electron and ion pressure")
        pe=np.asarray(electron_pressure_Pa,dtype=float); pi=np.asarray(ion_pressure_Pa,dtype=float)
        if not np.all(np.isfinite(pe)) or not np.all(np.isfinite(pi)) or np.any(pe<0) or np.any(pi<0): raise ValueError("Species pressures must be finite and non-negative")
        out["beta_e"]=2*MU0*pe/b**2; out["beta_i"]=2*MU0*pi/b**2; out["beta_thermal"]=2*MU0*(pe+pi)/b**2
    out["caveat"]="Use consistent local or volume-averaged pressure and field definitions; this function does not reconstruct equilibrium."
    return out
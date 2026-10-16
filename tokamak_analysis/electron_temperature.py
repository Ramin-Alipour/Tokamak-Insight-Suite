"""Electron-temperature profile statistics from calibrated measurements."""
import numpy as np
def analyze_electron_temperature(temperature_eV, *, coordinate=None, uncertainty_eV=None):
    te=np.asarray(temperature_eV,dtype=float)
    if te.ndim!=1 or len(te)<2 or not np.all(np.isfinite(te)) or np.any(te<0): raise ValueError("Temperature must be finite, non-negative 1-D data")
    out={"temperature_eV":te,"mean_eV":float(np.mean(te)),"median_eV":float(np.median(te)),"peak_eV":float(np.max(te)),"peak_index":int(np.argmax(te)),"standard_deviation_eV":float(np.std(te,ddof=1))}
    if coordinate is not None:
        x=np.asarray(coordinate,dtype=float)
        if x.shape!=te.shape or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0): raise ValueError("coordinate must match and increase strictly")
        out["coordinate"]=x
        out["gradient_eV_per_coordinate"]=np.gradient(te,x,edge_order=2 if len(te)>2 else 1)
    if uncertainty_eV is not None:
        u=np.asarray(uncertainty_eV,dtype=float)
        if u.shape!=te.shape or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Uncertainties must be finite, non-negative, and match data")
        out["uncertainty_eV"]=u
    out["caveat"]="Statistics summarize calibrated Te measurements; they do not perform Thomson spectral fitting."
    return out

def electron_temperature_from_pressure_density(electron_pressure_Pa, electron_density_m3):
    """Ideal-gas electron temperature in eV: Te_eV=pe/(ne*e)."""
    pe,ne=np.broadcast_arrays(np.asarray(electron_pressure_Pa,dtype=float),np.asarray(electron_density_m3,dtype=float))
    if not np.all(np.isfinite(pe)) or not np.all(np.isfinite(ne)) or np.any(pe<0) or np.any(ne<=0):
        raise ValueError("Electron pressure must be non-negative and density positive; inputs finite")
    te=pe/(ne*1.602176634e-19)
    return {"electron_temperature_eV":te,"electron_pressure_Pa":pe,"electron_density_m3":ne,
            "equation":"Te[eV]=pe[Pa]/(ne[m^-3]*e[C])",
            "assumption":"Single-temperature ideal electron gas with pe=ne*kB*Te[K]=ne*e*Te[eV]."}

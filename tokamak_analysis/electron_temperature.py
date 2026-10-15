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
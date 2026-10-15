"""Ion-temperature profile analysis from already inferred measurements."""
import numpy as np
def analyze_ion_temperature(temperature_eV, *, coordinate=None, uncertainty_eV=None):
    ti=np.asarray(temperature_eV,dtype=float)
    if ti.ndim!=1 or len(ti)<2 or not np.all(np.isfinite(ti)) or np.any(ti<0): raise ValueError("Ion temperature must be finite, non-negative 1-D data")
    out={"temperature_i_eV":ti,"mean_eV":float(np.mean(ti)),"median_eV":float(np.median(ti)),"peak_eV":float(np.max(ti)),"peak_index":int(np.argmax(ti)),"standard_deviation_eV":float(np.std(ti,ddof=1))}
    if coordinate is not None:
        x=np.asarray(coordinate,dtype=float)
        if x.shape!=ti.shape or not np.all(np.isfinite(x)) or np.any(np.diff(x)<=0): raise ValueError("coordinate must match and increase strictly")
        out["coordinate"]=x; out["gradient_eV_per_coordinate"]=np.gradient(ti,x,edge_order=2 if len(ti)>2 else 1)
    if uncertainty_eV is not None:
        u=np.asarray(uncertainty_eV,dtype=float)
        if u.shape!=ti.shape or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Uncertainties must be finite, non-negative, and match data")
        out["uncertainty_eV"]=u
    out["caveat"]="Input must be a measured or independently inferred Ti profile; this does not fit spectral line widths."
    return out
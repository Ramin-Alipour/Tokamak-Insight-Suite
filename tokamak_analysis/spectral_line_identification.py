"""Candidate spectral-line identification by nearest reference wavelength."""
import numpy as np
def identify_spectral_lines(observed_nm, reference_lines, *, tolerance_nm=0.05):
    obs=np.asarray(observed_nm,dtype=float); tol=float(tolerance_nm)
    if obs.ndim!=1 or not np.all(np.isfinite(obs)) or not np.isfinite(tol) or tol<=0: raise ValueError("Observed wavelengths and tolerance must be finite; tolerance positive")
    refs=[]
    for item in reference_lines:
        if len(item)<2: raise ValueError("Each reference line must contain wavelength and label")
        wl=float(item[0]); label=str(item[1])
        if not np.isfinite(wl) or wl<=0: raise ValueError("Reference wavelengths must be positive")
        refs.append((wl,label))
    if not refs: raise ValueError("Reference line list cannot be empty")
    result=[]
    for wl in obs:
        ref,label=min(refs,key=lambda z:abs(z[0]-wl)); delta=float(wl-ref)
        result.append({"observed_nm":float(wl),"candidate_nm":ref,"label":label,"delta_nm":delta,"matched":abs(delta)<=tol})
    return {"matches":result,"tolerance_nm":tol,"caveat":"Nearest-wavelength matching is preliminary; use a trusted atomic database and account for calibration, blends, charge state, and plasma conditions."}
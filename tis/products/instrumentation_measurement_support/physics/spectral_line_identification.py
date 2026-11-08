"""Candidate spectral-line identification by nearest reference wavelength."""
import numpy as np
def identify_spectral_lines(observed_nm, reference_lines, *, tolerance_nm=0.05):
    obs=np.asarray(observed_nm,dtype=float); tol=float(tolerance_nm)
    if obs.ndim!=1 or obs.size<1 or not np.all(np.isfinite(obs)) or np.any(obs<=0) or not np.isfinite(tol) or tol<=0: raise ValueError("Observed wavelengths must be a nonempty finite positive 1-D array; tolerance must be positive")
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

def hydrogen_rydberg_wavelength_nm(n_upper, n_lower):
    """Hydrogen transition wavelength from 1/lambda=R_H*(1/n_lower^2-1/n_upper^2)."""
    nu, nl=np.broadcast_arrays(np.asarray(n_upper,dtype=float),np.asarray(n_lower,dtype=float))
    if not np.all(np.isfinite(nu)) or not np.all(np.isfinite(nl)) or np.any(nl<1) or np.any(nu<=nl) or np.any(nu!=np.floor(nu)) or np.any(nl!=np.floor(nl)):
        raise ValueError("Principal quantum numbers must be integers with n_upper > n_lower >= 1")
    R_H=10967758.340
    inv_lambda=R_H*(1/nl**2-1/nu**2)
    return {"wavelength_nm":1e9/inv_lambda,
            "equation":"1/lambda=R_H*(1/n_lower^2-1/n_upper^2)",
            "caveat":"Non-relativistic hydrogen Rydberg estimate; fine structure, isotope shifts, and plasma wavelength shifts are omitted."}

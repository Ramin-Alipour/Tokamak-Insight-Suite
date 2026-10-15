"""Gamma-ray peak and count-rate analysis from calibrated spectra."""
import numpy as np
def analyze_gamma_spectrum(energy_keV, counts, *, expected_lines_keV=None, tolerance_keV=5.0):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float); tol=float(tolerance_keV)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0) or tol<=0: raise ValueError("Invalid spectrum or tolerance")
    peaks=[{"energy_keV":float(e[i]),"counts":float(c[i])} for i in range(1,len(c)-1) if c[i]>=c[i-1] and c[i]>c[i+1]]
    matches=[]
    if expected_lines_keV is not None:
        for line in expected_lines_keV:
            target=float(line); idx=int(np.argmin(abs(e-target)))
            matches.append({"expected_keV":target,"observed_bin_keV":float(e[idx]),"counts":float(c[idx]),"within_tolerance":bool(abs(e[idx]-target)<=tol)})
    return {"peaks":peaks,"expected_line_matches":matches,"tolerance_keV":tol,"caveat":"Peak assignment is preliminary; detector calibration, efficiency, background, and nuclear-line databases are needed."}
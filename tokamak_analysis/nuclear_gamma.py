"""Gamma-ray peak and count-rate analysis from calibrated spectra."""
import numpy as np
def analyze_gamma_spectrum(energy_keV, counts, *, expected_lines_keV=None, tolerance_keV=5.0):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float); tol=float(tolerance_keV)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0) or not np.isfinite(tol) or tol<=0: raise ValueError("Invalid spectrum or finite positive tolerance required")
    peaks=[{"energy_keV":float(e[i]),"counts":float(c[i])} for i in range(1,len(c)-1) if c[i]>=c[i-1] and c[i]>c[i+1]]
    matches=[]
    if expected_lines_keV is not None:
        for line in expected_lines_keV:
            target=float(line)
            if not np.isfinite(target) or target<=0: raise ValueError("Expected gamma-line energies must be finite and positive")
            idx=int(np.argmin(abs(e-target)))
            matches.append({"expected_keV":target,"observed_bin_keV":float(e[idx]),"counts":float(c[idx]),"within_tolerance":bool(abs(e[idx]-target)<=tol)})
    return {"peaks":peaks,"expected_line_matches":matches,"tolerance_keV":tol,"caveat":"Peak assignment is preliminary; detector calibration, efficiency, background, and nuclear-line databases are needed."}

def activity_from_gamma_peak_rate(net_peak_count_rate_s, full_energy_peak_efficiency, emission_probability):
    """Gamma spectroscopy activity relation A=R_peak/(epsilon_peak*I_gamma)."""
    rate,eff,prob=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (net_peak_count_rate_s,full_energy_peak_efficiency,emission_probability)])
    if any(not np.all(np.isfinite(v)) for v in (rate,eff,prob)) or np.any(rate<0) or np.any(eff<=0) or np.any(eff>1) or np.any(prob<=0) or np.any(prob>1):
        raise ValueError("Rate non-negative; efficiency and emission probability in (0,1]; all finite")
    activity=rate/(eff*prob)
    return {"activity_Bq":activity,"equation":"A=R_peak/(epsilon_peak*I_gamma)",
            "assumption":"Requires net peak count rate, absolute full-energy peak efficiency, emission probability, and geometry/self-attenuation corrections."}

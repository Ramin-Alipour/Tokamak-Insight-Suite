"""Runaway-electron characteristic energy estimates from calibrated HXR data."""
import numpy as np
def estimate_runaway_energy(energy_keV, counts, *, detector_response_corrected=False, fit_range_keV=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<4 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(c<0): raise ValueError("Invalid HXR spectrum")
    m=np.ones_like(e,dtype=bool)
    if fit_range_keV is not None:
        lo,hi=map(float,fit_range_keV); m=(e>=lo)&(e<=hi)
    m &= c>0
    if np.count_nonzero(m)<3: raise ValueError("Need at least three positive bins in fit range")
    slope,intercept=np.polyfit(e[m],np.log(c[m]),1)
    temperature=-1/slope if slope<0 else float("nan")
    return {"exponential_slope_per_keV":float(slope),"characteristic_energy_keV":float(temperature),"fit_intercept":float(intercept),"response_corrected":bool(detector_response_corrected),"caveat":"Exponential-slope energy is a model-dependent characteristic scale, not a unique maximum runaway-electron energy; detector response correction is essential."}

def runaway_speed_from_kinetic_energy(kinetic_energy_keV):
    """Relativistic electron speed from kinetic energy: gamma=1+K/(m_e*c^2), beta=sqrt(1-gamma^-2)."""
    energy=np.asarray(kinetic_energy_keV,dtype=float); rest_energy_keV=510.99895; c=299792458.0
    if not np.all(np.isfinite(energy)) or np.any(energy<0): raise ValueError("Kinetic energy must be finite and non-negative")
    gamma=1+energy/rest_energy_keV; beta=np.sqrt(1-1/gamma**2)
    return {"gamma_factor":gamma,"speed_m_s":c*beta,"speed_over_c":beta,
            "equation":"gamma=1+K/(m_e*c^2); v/c=sqrt(1-1/gamma^2)"}

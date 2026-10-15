"""Line-of-sight flow velocity from calibrated Doppler line-center shift."""
import numpy as np
C=299792458.0
def doppler_flow_velocity(observed_wavelength_nm, rest_wavelength_nm, *, uncertainty_nm=None, relativistic=False):
    obs=np.asarray(observed_wavelength_nm,dtype=float); rest=float(rest_wavelength_nm)
    if not np.all(np.isfinite(obs)) or not np.isfinite(rest) or np.any(obs<=0) or rest<=0: raise ValueError("Wavelengths must be finite and positive")
    ratio=obs/rest
    if relativistic:
        if np.any(ratio<=0): raise ValueError("Invalid wavelength ratio")
        beta=(ratio**2-1)/(ratio**2+1); v=C*beta
    else: v=C*(ratio-1)
    out={"velocity_m_s":v,"wavelength_shift_nm":obs-rest,"relativistic":bool(relativistic)}
    if uncertainty_nm is not None:
        u=np.asarray(uncertainty_nm,dtype=float)
        if np.any(~np.isfinite(u)) or np.any(u<0): raise ValueError("uncertainty_nm must be finite and non-negative")
        out["velocity_uncertainty_m_s"]=C*u/rest
    out["caveat"]="Velocity is line-of-sight and depends on accurate wavelength calibration and rest-line identification."
    return out
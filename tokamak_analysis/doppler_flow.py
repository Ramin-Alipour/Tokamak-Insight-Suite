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
        if not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("uncertainty_nm must be finite and non-negative")
        try: u=np.broadcast_to(u,obs.shape)
        except ValueError as exc: raise ValueError("uncertainty_nm must be scalar or broadcast-compatible with observed wavelength") from exc
        if relativistic:
            out["velocity_uncertainty_m_s"]=C*(4*ratio/(ratio**2+1)**2)*(u/rest)
        else:
            out["velocity_uncertainty_m_s"]=C*u/rest
    out["caveat"]="Velocity is line-of-sight and depends on accurate wavelength calibration and rest-line identification."
    return out

def wavelength_from_line_of_sight_velocity(rest_wavelength_nm, velocity_m_s, *, relativistic=False):
    """Forward Doppler relation from line-of-sight velocity to observed wavelength."""
    rest,v=np.broadcast_arrays(np.asarray(rest_wavelength_nm,dtype=float),np.asarray(velocity_m_s,dtype=float))
    if not np.all(np.isfinite(rest)) or not np.all(np.isfinite(v)) or np.any(rest<=0) or np.any(np.abs(v)>=C):
        raise ValueError("Rest wavelength must be positive and |velocity|<c; inputs finite")
    beta=v/C
    ratio=np.sqrt((1+beta)/(1-beta)) if relativistic else (1+beta)
    if not relativistic and np.any(ratio<=0): raise ValueError("Non-relativistic wavelength factor is non-positive")
    return {"observed_wavelength_nm":rest*ratio,"equation":"lambda=lambda0*sqrt((1+beta)/(1-beta)) relativistic; lambda=lambda0*(1+beta) nonrelativistic" if relativistic else "lambda=lambda0*(1+v/c)"}

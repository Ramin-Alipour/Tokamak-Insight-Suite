"""Line-integrated density from calibrated interferometer phase shift."""
import numpy as np
C=299792458.0; E=1.602176634e-19; ME=9.1093837139e-31; EPS0=8.8541878128e-12
def interferometry_density(phase_shift_rad, probe_frequency_Hz, *, path_length_m=None, sign=1):
    phase=np.asarray(phase_shift_rad,dtype=float); f=float(probe_frequency_Hz)
    if not np.all(np.isfinite(phase)) or not np.isfinite(f) or f<=0 or sign not in (-1,1): raise ValueError("Phase/frequency invalid; sign must be ±1")
    omega=2*np.pi*f; re=E**2/(4*np.pi*EPS0*ME*C**2)
    line_density=sign*phase/(re*(C/f))
    out={"line_integrated_density_m-2":line_density,"probe_frequency_Hz":f}
    if path_length_m is not None:
        L=float(path_length_m)
        if not np.isfinite(L) or L<=0: raise ValueError("Path length must be positive")
        out["line_average_density_m-3"]=line_density/L
    out["caveat"]="Formula assumes a simple unmagnetized plasma phase model and correct fringe/sign calibration; use the diagnostic's validated refractive-index model where needed."
    return out

def phase_shift_from_line_density(line_integrated_density_m2, probe_frequency_Hz, *, sign=1):
    """Forward cold-plasma interferometer phase relation: Delta_phi=sign*r_e*lambda*integral(ne dl)."""
    neL=np.asarray(line_integrated_density_m2,dtype=float); f=float(probe_frequency_Hz)
    if not np.all(np.isfinite(neL)) or np.any(neL<0) or not np.isfinite(f) or f<=0 or sign not in (-1,1):
        raise ValueError("Line density must be finite/non-negative, frequency positive, sign ±1")
    re=E**2/(4*np.pi*EPS0*ME*C**2); wavelength=C/f
    phase=sign*re*wavelength*neL
    return {"phase_shift_rad":phase,"wavelength_m":wavelength,
            "equation":"Delta_phi=sign*r_e*lambda*integral(ne dl)",
            "caveat":"Simple unmagnetized cold-plasma approximation; magnetized propagation and instrument phase offsets may require a fuller model."}

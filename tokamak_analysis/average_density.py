"""Average density metrics from calibrated line-integrated or profile data."""
import numpy as np
def analyze_average_density(*, density_profile_m3=None, radius_m=None, line_integrated_density_m2=None, chord_length_m=None):
    out={}
    if density_profile_m3 is not None:
        ne=np.asarray(density_profile_m3,dtype=float)
        if ne.ndim!=1 or len(ne)<2 or not np.all(np.isfinite(ne)) or np.any(ne<0): raise ValueError("Density profile must be finite, non-negative 1-D data")
        out["sample_mean_density_m3"]=float(np.mean(ne)); out["peak_density_m3"]=float(np.max(ne))
        if radius_m is not None:
            r=np.asarray(radius_m,dtype=float)
            if r.shape!=ne.shape or not np.all(np.isfinite(r)) or np.any(np.diff(r)<=0): raise ValueError("radius_m must match density profile and increase strictly")
            out["radial_average_density_m3"]=float(np.trapezoid(ne,r)/(r[-1]-r[0]))
    if line_integrated_density_m2 is not None:
        if chord_length_m is None: raise ValueError("chord_length_m required for line-average density")
        nbar=float(line_integrated_density_m2); length=float(chord_length_m)
        if not np.isfinite(nbar) or nbar<0 or not np.isfinite(length) or length<=0: raise ValueError("Line integral must be non-negative and chord length positive")
        out["line_average_density_m3"]=nbar/length
    if not out: raise ValueError("Provide density_profile_m3 or line_integrated_density_m2")
    out["caveat"]="Line-average density assumes the supplied integral and chord length correspond to the same path."
    return out
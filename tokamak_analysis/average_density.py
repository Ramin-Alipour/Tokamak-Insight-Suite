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
            out["radial_average_density_m3"]=float(getattr(np, "trapezoid", np.trapz)(ne,r)/(r[-1]-r[0]))
    if line_integrated_density_m2 is not None:
        if chord_length_m is None: raise ValueError("chord_length_m required for line-average density")
        nbar=float(line_integrated_density_m2); length=float(chord_length_m)
        if not np.isfinite(nbar) or nbar<0 or not np.isfinite(length) or length<=0: raise ValueError("Line integral must be non-negative and chord length positive")
        out["line_average_density_m3"]=nbar/length
    if not out: raise ValueError("Provide density_profile_m3 or line_integrated_density_m2")
    out["caveat"]="Line-average density assumes the supplied integral and chord length correspond to the same path."
    return out

def cylindrical_volume_average_density(radius_m, density_profile_m3):
    """Flux-surface-independent cylindrical volume average: <n>=2/a^2 integral_0^a n(r) r dr."""
    r=np.asarray(radius_m,dtype=float); ne=np.asarray(density_profile_m3,dtype=float)
    if r.ndim!=1 or ne.shape!=r.shape or len(r)<2 or not np.all(np.isfinite(r)) or not np.all(np.isfinite(ne)):
        raise ValueError("Radius and density must be finite matching 1-D arrays with at least 2 points")
    if np.any(np.diff(r)<=0) or r[0]<0 or r[-1]<=0 or np.any(ne<0):
        raise ValueError("Radius must increase from non-negative values and density must be non-negative")
    if not np.isclose(r[0],0.0,atol=1e-12): raise ValueError("Profile must include the magnetic axis at r=0")
    a=float(r[-1]); avg=2*float(np.trapezoid(ne*r,r))/(a*a)
    return {"volume_average_density_m3":avg,"minor_radius_m":a,
            "equation":"<n>_V=(2/a^2)*integral_0^a n(r)*r dr",
            "caveat":"Assumes circular cylindrical cross-section and density represented as a function of minor radius."}

"""Bolometer signal integration and radiated-power estimate."""
import numpy as np
def analyze_bolometry(time_s, channel_power_W_m2, *, channel_weights=None):
    t=np.asarray(time_s,dtype=float); p=np.asarray(channel_power_W_m2,dtype=float)
    if t.ndim!=1 or p.ndim not in (1,2) or p.shape[0]!=len(t) or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or np.any(np.diff(t)<=0): raise ValueError("Finite increasing time and matching signals required")
    energy=(getattr(np, "trapezoid", None) or getattr(np, "trapz"))(p,t,axis=0)
    out={"time_s":t,"signal_W_m2":p,"time_integrated_signal_J_m2":energy,"peak_signal_W_m2":np.max(p,axis=0)}
    if p.ndim==2 and channel_weights is not None:
        w=np.asarray(channel_weights,dtype=float)
        if w.shape!=(p.shape[1],) or not np.all(np.isfinite(w)): raise ValueError("Weights must match channel count")
        out["weighted_total_signal"]=p@w; out["weighted_integral"]=float((getattr(np, "trapezoid", None) or getattr(np, "trapz"))(p@w,t))
    out["caveat"]="Bolometer power requires detector calibration, view factors and geometric inversion; channel sums alone are not total plasma radiated power."
    return out

def radiated_power_from_axisymmetric_emissivity(radius_m, emissivity_W_m3, major_radius_m):
    """Integrate toroidally symmetric emissivity: P_rad=4*pi^2*R0*integral(epsilon(r)*r dr)."""
    r=np.asarray(radius_m,dtype=float); eps=np.asarray(emissivity_W_m3,dtype=float)
    if not isinstance(major_radius_m, (int, float, np.integer, np.floating)) or isinstance(major_radius_m, (bool, np.bool_)):
        raise ValueError("Major radius must be a finite real scalar.")
    try:
        R0=float(major_radius_m)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Major radius must be a finite real scalar.") from exc
    if not np.isfinite(R0):
        raise ValueError("Major radius must be finite.")
    if r.ndim!=1 or eps.shape!=r.shape or len(r)<2 or not np.all(np.isfinite(r)) or not np.all(np.isfinite(eps)):
        raise ValueError("Radius and emissivity must be finite matching 1-D arrays.")
    if np.any(np.diff(r)<=0) or not np.isclose(r[0],0.0,atol=1e-12) or r[-1]<=0 or np.any(eps<0) or R0<=r[-1]:
        raise ValueError("Require increasing radius from zero, non-negative emissivity, and R0 greater than minor radius")
    power=4*np.pi**2*R0*float((getattr(np, "trapezoid", None) or getattr(np, "trapz"))(eps*r,r))
    return {"total_radiated_power_W":power,"equation":"P_rad=4*pi^2*R0*integral_0^a epsilon(r)*r dr",
            "assumption":"Axisymmetric circular torus and emissivity uniform on each flux surface."}

"""Hard X-ray count-spectrum analysis and calibration-aware energy summaries."""
import numpy as np
def analyze_hard_xray(energy_keV, counts, *, live_time_s=None, background_counts=None):
    e=np.asarray(energy_keV,dtype=float); c=np.asarray(counts,dtype=float)
    if e.ndim!=1 or c.shape!=e.shape or len(e)<2 or not np.all(np.isfinite(e)) or not np.all(np.isfinite(c)) or np.any(np.diff(e)<=0) or np.any(e<0) or np.any(c<0): raise ValueError("Energy bins must be finite, increasing and non-negative; counts finite and nonnegative")
    net=c.copy()
    if background_counts is not None:
        bg=np.asarray(background_counts,dtype=float)
        if bg.shape!=c.shape or not np.all(np.isfinite(bg)) or np.any(bg<0): raise ValueError("Background must match counts and be nonnegative")
        net=np.maximum(c-bg,0)
    total=float(np.sum(net)); out={"energy_keV":e,"net_counts":net,"total_net_counts":total,"peak_energy_keV":float(e[np.argmax(net)]) if total>0 else None,"count_weighted_mean_energy_keV":float(np.sum(e*net)/total) if total else float("nan")}
    if live_time_s is not None:
        t=float(live_time_s)
        if not np.isfinite(t) or t<=0: raise ValueError("Live time must be positive")
        out["count_rate_s-1"]=total/t
    out["caveat"]="Energy calibration, detector efficiency, dead time and background uncertainty are not corrected unless separately supplied."
    return out

def compton_scattered_photon_energy_keV(incident_energy_keV, scattering_angle_rad):
    """Compton energy-shift relation E'=E/[1+(E/511 keV)(1-cos(theta))]."""
    energy,theta=np.broadcast_arrays(np.asarray(incident_energy_keV,dtype=float),np.asarray(scattering_angle_rad,dtype=float))
    if not np.all(np.isfinite(energy)) or not np.all(np.isfinite(theta)) or np.any(energy<=0) or np.any(theta<0) or np.any(theta>np.pi):
        raise ValueError("Energy must be positive and scattering angle in [0, pi]")
    scattered=energy/(1+(energy/510.99895)*(1-np.cos(theta)))
    return {"scattered_photon_energy_keV":scattered,"energy_recoil_keV":energy-scattered,
            "equation":"E_prime=E/(1+(E/510.99895 keV)*(1-cos(theta)))"}

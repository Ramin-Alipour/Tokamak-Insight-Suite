"""Electron/ion heating power and energy deposition summaries."""
import numpy as np
def analyze_heating(time_s, power_W, *, stored_energy_J=None, species="electron"):
    t=np.asarray(time_s,dtype=float); p=np.asarray(power_W,dtype=float)
    if species not in ("electron","ion"): raise ValueError("species must be electron or ion")
    if t.ndim!=1 or p.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or np.any(np.diff(t)<=0) or np.any(p<0): raise ValueError("Invalid heating power time series")
    out={"species":species,"time_s":t,"power_W":p,"peak_power_W":float(np.max(p)),"injected_energy_J":float((getattr(np, "trapezoid", None) or getattr(np, "trapz"))(p,t))}
    if stored_energy_J is not None:
        w=np.asarray(stored_energy_J,dtype=float)
        if w.shape!=p.shape or not np.all(np.isfinite(w)) or np.any(w<0): raise ValueError("Stored energy must match and be nonnegative")
        out["stored_energy_change_J"]=float(w[-1]-w[0]); out["stored_energy_J"]=w
    out["caveat"]="Species-specific heating requires validated power partition and deposition model; input power alone does not establish absorbed plasma heating."
    return out

def electron_heating_power_from_partition(absorbed_power_W, electron_fraction):
    """Species power partition: P_e=f_e*P_abs."""
    power,f=np.broadcast_arrays(np.asarray(absorbed_power_W,dtype=float),np.asarray(electron_fraction,dtype=float))
    if not np.all(np.isfinite(power)) or not np.all(np.isfinite(f)) or np.any(power<0) or np.any(f<0) or np.any(f>1):
        raise ValueError("Absorbed power non-negative and electron fraction in [0,1]")
    return {"electron_heating_power_W":power*f,"equation":"P_e=f_e*P_abs"}


def ion_heating_power_from_partition(absorbed_power_W, electron_fraction):
    """Complementary species power partition: P_i=(1-f_e)*P_abs."""
    power,f=np.broadcast_arrays(np.asarray(absorbed_power_W,dtype=float),np.asarray(electron_fraction,dtype=float))
    if not np.all(np.isfinite(power)) or not np.all(np.isfinite(f)) or np.any(power<0) or np.any(f<0) or np.any(f>1):
        raise ValueError("Absorbed power non-negative and electron fraction in [0,1]")
    return {"ion_heating_power_W":power*(1-f),"equation":"P_i=(1-f_e)*P_abs"}


def analyze_electron_heating(time_s, power_W, *, stored_energy_J=None):
    """Electron-specific heating summary; absorbed electron power must be supplied."""
    return analyze_heating(time_s, power_W, stored_energy_J=stored_energy_J, species="electron")


def analyze_ion_heating(time_s, power_W, *, stored_energy_J=None):
    """Ion-specific heating summary; absorbed ion power must be supplied."""
    return analyze_heating(time_s, power_W, stored_energy_J=stored_energy_J, species="ion")

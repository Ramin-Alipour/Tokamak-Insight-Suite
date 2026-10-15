"""Electron/ion heating power and energy deposition summaries."""
import numpy as np
def analyze_heating(time_s, power_W, *, stored_energy_J=None, species="electron"):
    t=np.asarray(time_s,dtype=float); p=np.asarray(power_W,dtype=float)
    if t.ndim!=1 or p.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or np.any(np.diff(t)<=0) or np.any(p<0): raise ValueError("Invalid heating power time series")
    out={"species":species,"time_s":t,"power_W":p,"peak_power_W":float(np.max(p)),"injected_energy_J":float(np.trapezoid(p,t))}
    if stored_energy_J is not None:
        w=np.asarray(stored_energy_J,dtype=float)
        if w.shape!=p.shape or not np.all(np.isfinite(w)) or np.any(w<0): raise ValueError("Stored energy must match and be nonnegative")
        out["stored_energy_change_J"]=float(w[-1]-w[0]); out["stored_energy_J"]=w
    out["caveat"]="Species-specific heating requires validated power partition and deposition model; input power alone does not establish absorbed plasma heating."
    return out
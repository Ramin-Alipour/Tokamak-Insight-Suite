"""Gas-puff valve command and calibrated flow analysis."""
import numpy as np
def analyze_gas_puffing(time_s, valve_signal, *, calibrated_flow_mol_s=None):
    t=np.asarray(time_s,dtype=float); v=np.asarray(valve_signal,dtype=float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or np.any(np.diff(t)<=0): raise ValueError("Invalid valve time series")
    out={"time_s":t,"valve_signal":v,"integrated_valve_command_s":float(np.trapezoid(v,t)),"open_fraction":float(np.mean(v>0))}
    if calibrated_flow_mol_s is not None:
        q=np.asarray(calibrated_flow_mol_s,dtype=float)
        if q.shape!=t.shape or not np.all(np.isfinite(q)) or np.any(q<0): raise ValueError("Flow must match and be nonnegative")
        out["flow_mol_s"]=q; out["injected_amount_mol"]=float(np.trapezoid(q,t))
    out["caveat"]="Valve command alone is not a gas throughput measurement; calibrated flow and plumbing response are required."
    return out
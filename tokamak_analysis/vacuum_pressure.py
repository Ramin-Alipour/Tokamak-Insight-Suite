"""Vacuum pressure diagnostics and pressure-rise metrics."""
import numpy as np
def analyze_vacuum_pressure(time_s, pressure_Pa, *, base_pressure_Pa=None):
    t=np.asarray(time_s,dtype=float); p=np.asarray(pressure_Pa,dtype=float)
    if t.ndim!=1 or p.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or np.any(np.diff(t)<=0) or np.any(p<0): raise ValueError("Invalid pressure time series")
    dpdt=np.gradient(p,t,edge_order=2 if len(t)>2 else 1)
    out={"time_s":t,"pressure_Pa":p,"peak_pressure_Pa":float(np.max(p)),"mean_pressure_Pa":float(np.mean(p)),"pressure_rise_rate_Pa_s":dpdt,"pressure_time_integral_Pa_s":float(np.trapezoid(p,t))}
    if base_pressure_Pa is not None:
        base=float(base_pressure_Pa)
        if base<0 or not np.isfinite(base): raise ValueError("Base pressure must be nonnegative")
        out["pressure_above_base_Pa"]=p-base
    return out
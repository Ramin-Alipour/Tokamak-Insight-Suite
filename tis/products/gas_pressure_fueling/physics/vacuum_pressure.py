"""Vacuum pressure diagnostics and pressure-rise metrics."""
import numpy as np
def analyze_vacuum_pressure(time_s, pressure_Pa, *, base_pressure_Pa=None):
    t=np.asarray(time_s,dtype=float); p=np.asarray(pressure_Pa,dtype=float)
    if t.ndim!=1 or p.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(p)) or np.any(np.diff(t)<=0) or np.any(p<0): raise ValueError("Invalid pressure time series")
    dpdt=np.gradient(p,t,edge_order=2 if len(t)>2 else 1)
    out={"time_s":t,"pressure_Pa":p,"peak_pressure_Pa":float(np.max(p)),"mean_pressure_Pa":float(np.mean(p)),"pressure_rise_rate_Pa_s":dpdt,"pressure_time_integral_Pa_s":float((getattr(np, "trapezoid", None) or getattr(np, "trapz"))(p,t))}
    if base_pressure_Pa is not None:
        base=float(base_pressure_Pa)
        if base<0 or not np.isfinite(base): raise ValueError("Base pressure must be nonnegative")
        out["pressure_above_base_Pa"]=p-base
    return out

def pressure_rate_from_gas_load(gas_throughput_Pa_m3_s, pumping_speed_m3_s, pressure_Pa, chamber_volume_m3):
    """Lumped vacuum balance: V*dP/dt=Q-S*P."""
    Q,S,P,V=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (gas_throughput_Pa_m3_s,pumping_speed_m3_s,pressure_Pa,chamber_volume_m3)])
    if any(not np.all(np.isfinite(v)) for v in (Q,S,P,V)) or np.any(Q<0) or np.any(S<0) or np.any(P<0) or np.any(V<=0):
        raise ValueError("Gas load, pumping speed and pressure non-negative; volume positive; all finite")
    dpdt=(Q-S*P)/V
    return {"pressure_rate_Pa_s":dpdt,"equation":"dP/dt=(Q-S*P)/V",
            "assumption":"Well-mixed fixed-volume chamber with constant pumping speed and gas throughput."}

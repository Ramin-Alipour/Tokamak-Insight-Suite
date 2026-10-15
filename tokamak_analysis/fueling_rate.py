"""Fueling rate and injected-particle inventory estimates."""
import numpy as np
K_B=1.380649e-23
def analyze_fueling_rate(time_s, particle_rate_s, *, gas_temperature_K=None, pressure_Pa=None, volume_m3=None):
    t=np.asarray(time_s,dtype=float); rate=np.asarray(particle_rate_s,dtype=float)
    if t.ndim!=1 or rate.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(rate)) or np.any(np.diff(t)<=0) or np.any(rate<0): raise ValueError("Invalid particle-rate series")
    out={"time_s":t,"particle_rate_s-1":rate,"injected_particles":float(np.trapezoid(rate,t)),"peak_rate_s-1":float(np.max(rate))}
    if any(x is not None for x in (gas_temperature_K,pressure_Pa,volume_m3)):
        if any(x is None for x in (gas_temperature_K,pressure_Pa,volume_m3)): raise ValueError("Provide temperature, pressure and volume together")
        T,P,V=map(float,(gas_temperature_K,pressure_Pa,volume_m3))
        if min(T,P,V)<=0: raise ValueError("Temperature, pressure and volume must be positive")
        out["ideal_gas_inventory_particles"]=P*V/(K_B*T)
    return out
"""Neutral gas density from pressure and gas temperature."""
import numpy as np
K_B=1.380649e-23
def neutral_density(pressure_Pa, temperature_K):
    p=np.asarray(pressure_Pa,dtype=float); T=np.asarray(temperature_K,dtype=float)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(T)) or np.any(p<0) or np.any(T<=0): raise ValueError("Pressure nonnegative and temperature positive required")
    return {"neutral_density_m-3":p/(K_B*T),"pressure_Pa":p,"temperature_K":T,"model":"ideal gas"}

"""Neutral gas density from pressure and gas temperature."""
import numpy as np
K_B=1.380649e-23
def neutral_density(pressure_Pa, temperature_K):
    p=np.asarray(pressure_Pa,dtype=float); T=np.asarray(temperature_K,dtype=float)
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(T)) or np.any(p<0) or np.any(T<=0): raise ValueError("Pressure nonnegative and temperature positive required")
    return {"neutral_density_m-3":p/(K_B*T),"pressure_Pa":p,"temperature_K":T,"model":"ideal gas"}


def neutral_mean_free_path_m(pressure_Pa, temperature_K, molecular_diameter_m):
    """Hard-sphere mean free path: lambda=kB*T/(sqrt(2)*pi*d^2*P)."""
    p,T,d=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (pressure_Pa,temperature_K,molecular_diameter_m)])
    if any(not np.all(np.isfinite(v)) for v in (p,T,d)) or np.any(p<=0) or np.any(T<=0) or np.any(d<=0):
        raise ValueError("Pressure, temperature, and molecular diameter must be finite and positive")
    return {"mean_free_path_m":K_B*T/(np.sqrt(2)*np.pi*d**2*p),
            "equation":"lambda_mfp=kB*T/(sqrt(2)*pi*d^2*P)",
            "assumption":"Ideal gas of hard-sphere molecules with specified effective diameter."}

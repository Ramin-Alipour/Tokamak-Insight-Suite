"""Energy confinement time from stored energy and net heating power."""
import numpy as np
def energy_confinement_time(stored_energy_J, net_heating_power_W, *, uncertainty_energy_J=None, uncertainty_power_W=None):
    W=np.asarray(stored_energy_J,dtype=float); P=np.asarray(net_heating_power_W,dtype=float)
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(P)) or np.any(W<0) or np.any(P<=0): raise ValueError("Stored energy nonnegative and net power positive")
    tau=W/P; out={"energy_confinement_time_s":tau}
    if uncertainty_energy_J is not None or uncertainty_power_W is not None:
        if uncertainty_energy_J is None or uncertainty_power_W is None: raise ValueError("Provide both uncertainties")
        sW=np.asarray(uncertainty_energy_J,dtype=float); sP=np.asarray(uncertainty_power_W,dtype=float)
        out["uncertainty_s"]=np.abs(tau)*np.sqrt((sW/np.maximum(W,np.finfo(float).tiny))**2+(sP/P)**2)
    out["caveat"]="Net heating power must consistently include auxiliary input, radiation, transport and other relevant losses."
    return out
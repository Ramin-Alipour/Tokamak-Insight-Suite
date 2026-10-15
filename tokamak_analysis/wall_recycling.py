"""Wall recycling coefficient estimate from incident and returned particle fluxes."""
import numpy as np
def recycling_coefficient(incident_flux_m2_s, returned_flux_m2_s, *, uncertainties=None):
    inc=np.asarray(incident_flux_m2_s,dtype=float); ret=np.asarray(returned_flux_m2_s,dtype=float)
    if not np.all(np.isfinite(inc)) or not np.all(np.isfinite(ret)) or np.any(inc<=0) or np.any(ret<0): raise ValueError("Incident flux positive and returned flux nonnegative")
    inc,ret=np.broadcast_arrays(inc,ret); r=ret/inc
    out={"recycling_coefficient":r}
    if uncertainties is not None:
        si,sr=map(lambda x:np.asarray(x,dtype=float),uncertainties)
        if np.any(si<0) or np.any(sr<0) or not np.all(np.isfinite(si)) or not np.all(np.isfinite(sr)): raise ValueError("Uncertainties must be finite and nonnegative")
        out["uncertainty"]=np.abs(r)*np.sqrt((sr/np.maximum(ret,np.finfo(float).tiny))**2+(si/inc)**2)
    out["caveat"]="Requires matched incident and recycled fluxes on consistent surfaces and time windows."
    return out
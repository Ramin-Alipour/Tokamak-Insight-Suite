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
        try:
            si=np.broadcast_to(si,inc.shape); sr=np.broadcast_to(sr,inc.shape)
        except ValueError as exc: raise ValueError("Uncertainties must be scalar or broadcast-compatible with flux shape") from exc
        out["uncertainty"]=np.sqrt((sr/inc)**2+(ret*si/inc**2)**2)
    out["caveat"]="Requires matched incident and recycled fluxes on consistent surfaces and time windows."
    return out

def net_wall_particle_sink(incident_flux_m2_s, returned_flux_m2_s):
    """Net wall sink flux from particle balance: Gamma_sink=Gamma_incident-Gamma_returned."""
    inc,ret=np.broadcast_arrays(np.asarray(incident_flux_m2_s,dtype=float),np.asarray(returned_flux_m2_s,dtype=float))
    if not np.all(np.isfinite(inc)) or not np.all(np.isfinite(ret)) or np.any(inc<0) or np.any(ret<0):
        raise ValueError("Incident and returned fluxes must be finite and non-negative")
    net=inc-ret
    return {"net_particle_sink_flux_m-2_s-1":net,"net_wall_sink_flux_m-2_s-1":net,
            "equation":"Gamma_sink=Gamma_incident-Gamma_returned",
            "units":"m^-2 s^-1","caveat":"Returned and incident fluxes must use matched surfaces, species conventions and time windows."}

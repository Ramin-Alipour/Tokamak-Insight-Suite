"""Energy confinement time from stored energy and net heating power."""
import numpy as np
def energy_confinement_time(stored_energy_J, net_heating_power_W, *, uncertainty_energy_J=None, uncertainty_power_W=None):
    W=np.asarray(stored_energy_J,dtype=float); P=np.asarray(net_heating_power_W,dtype=float)
    if not np.all(np.isfinite(W)) or not np.all(np.isfinite(P)) or np.any(W<0) or np.any(P<=0): raise ValueError("Stored energy nonnegative and net power positive")
    tau=W/P; out={"energy_confinement_time_s":tau}
    if uncertainty_energy_J is not None or uncertainty_power_W is not None:
        if uncertainty_energy_J is None or uncertainty_power_W is None: raise ValueError("Provide both uncertainties")
        sW=np.asarray(uncertainty_energy_J,dtype=float); sP=np.asarray(uncertainty_power_W,dtype=float)
        try: sW,sP=np.broadcast_arrays(sW,W); sP=np.broadcast_to(sP,P.shape)
        except ValueError as exc: raise ValueError("Uncertainties must be broadcast-compatible with stored energy and power") from exc
        out["uncertainty_s"]=np.sqrt((sW/P)**2+(W*sP/P**2)**2)
    out["caveat"]="Net heating power must consistently include auxiliary input, radiation, transport and other relevant losses."
    return out

def confinement_time_from_energy_decay(time_s, stored_energy_J, *, fit_range_s=None):
    """Transient exponential decay estimate: W(t)=W0*exp(-t/tau), tau=-1/slope(ln W)."""
    t=np.asarray(time_s,dtype=float); W=np.asarray(stored_energy_J,dtype=float)
    if t.ndim!=1 or W.shape!=t.shape or len(t)<3 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(W)) or np.any(np.diff(t)<=0) or np.any(W<=0):
        raise ValueError("Need positive finite energy and matching increasing time arrays")
    mask=np.ones_like(t,dtype=bool)
    if fit_range_s is not None:
        lo,hi=map(float,fit_range_s)
        if hi<=lo: raise ValueError("Invalid fit range")
        mask=(t>=lo)&(t<=hi)
    if np.count_nonzero(mask)<3: raise ValueError("At least three points required in fit range")
    slope,intercept=np.polyfit(t[mask],np.log(W[mask]),1)
    if slope>=0: raise ValueError("Energy must decay over selected fit range")
    return {"energy_decay_confinement_time_s":float(-1/slope),"log_energy_slope_s-1":float(slope),"fit_intercept":float(intercept),
            "equation":"W(t)=W0*exp(-t/tau_E); tau_E=-1/(d ln W/dt)",
            "caveat":"Transient decay time equals global energy confinement time only under the applicable post-heating power-balance assumptions."}

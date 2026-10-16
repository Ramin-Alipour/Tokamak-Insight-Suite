"""Neutron detector count-rate and flux estimate."""
import numpy as np
def analyze_neutron_flux(time_s, counts, *, efficiency=None, effective_area_m2=None, live_time_s=None):
    t=np.asarray(time_s,dtype=float); c=np.asarray(counts,dtype=float)
    if t.ndim!=1 or c.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)) or np.any(np.diff(t)<=0) or np.any(c<0): raise ValueError("Finite increasing time and nonnegative counts required")
    rate=np.gradient(c,t,edge_order=2 if len(t)>2 else 1)
    out={"time_s":t,"counts":c,"count_rate_s-1":rate,"total_counts":float(np.sum(c))}
    if efficiency is not None or effective_area_m2 is not None:
        if efficiency is None or effective_area_m2 is None: raise ValueError("Provide detector efficiency and effective area")
        eff=float(efficiency); area=float(effective_area_m2)
        if not 0<eff<=1 or area<=0: raise ValueError("Efficiency must be (0,1] and area positive")
        out["flux_estimate_m-2_s-1"]=rate/(eff*area)
    out["caveat"]="Flux estimate requires detector geometry, efficiency, background, dead-time, and energy-response corrections."
    return out

def neutron_flux_from_integrated_counts(counts, live_time_s, detector_efficiency, effective_area_m2):
    """Integrated-count flux estimator Phi=N/(T*epsilon*A) with Poisson counting uncertainty."""
    n,t,eff,area=np.broadcast_arrays(*[np.asarray(v,dtype=float) for v in (counts,live_time_s,detector_efficiency,effective_area_m2)])
    if any(not np.all(np.isfinite(v)) for v in (n,t,eff,area)) or np.any(n<0) or np.any(t<=0) or np.any(eff<=0) or np.any(eff>1) or np.any(area<=0):
        raise ValueError("Counts non-negative; time/area positive; efficiency in (0,1]; inputs finite")
    denom=t*eff*area
    return {"neutron_flux_m2_s":n/denom,"poisson_uncertainty_m2_s":np.sqrt(n)/denom,
            "equation":"Phi=N/(T*epsilon*A); sigma_Phi=sqrt(N)/(T*epsilon*A)",
            "assumption":"Poisson counting statistics and constant efficiency/area over live time; background and dead-time corrections must be applied separately."}

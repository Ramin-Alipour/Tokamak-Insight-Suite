"""Neutron detector count-rate and flux estimate."""
import numpy as np
def analyze_neutron_flux(time_s, counts, *, efficiency=None, effective_area_m2=None, live_time_s=None, counts_are_cumulative=False):
    t=np.asarray(time_s,dtype=float); c=np.asarray(counts,dtype=float)
    if t.ndim!=1 or c.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(c)) or np.any(np.diff(t)<=0) or np.any(c<0): raise ValueError("Finite increasing time and nonnegative counts required")
    if counts_are_cumulative:
        if live_time_s is not None: raise ValueError("live_time_s applies to per-bin counts, not cumulative counters")
        if np.any(np.diff(c)<0): raise ValueError("Cumulative counts must be nondecreasing")
        rate=np.gradient(c,t,edge_order=2 if len(t)>2 else 1)
        total_counts=float(c[-1]-c[0])
        rate_method="derivative of cumulative counter"
    else:
        if live_time_s is None:
            bin_width=np.gradient(t)
        else:
            bin_width=np.asarray(live_time_s,dtype=float)
            if not np.all(np.isfinite(bin_width)) or np.any(bin_width<=0): raise ValueError("live_time_s must be finite and positive")
            try: bin_width=np.broadcast_to(bin_width,c.shape)
            except ValueError as exc: raise ValueError("live_time_s must be scalar or match counts") from exc
        rate=c/bin_width
        total_counts=float(np.sum(c))
        rate_method="counts divided by supplied live time" if live_time_s is not None else "counts divided by local time-bin width estimated from time coordinates"
    if efficiency is not None or effective_area_m2 is not None:
        if efficiency is None or effective_area_m2 is None: raise ValueError("Provide detector efficiency and effective area")
        eff=float(efficiency); area=float(effective_area_m2)
        if not 0<eff<=1 or area<=0: raise ValueError("Efficiency must be (0,1] and area positive")
        out["flux_estimate_m-2_s-1"]=rate/(eff*area)
    out["caveat"]="For per-bin counts, time_s is interpreted as bin-center times unless live_time_s is supplied; cumulative counters must be selected explicitly. Flux also requires detector geometry, efficiency, background, dead-time, and energy-response corrections."
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

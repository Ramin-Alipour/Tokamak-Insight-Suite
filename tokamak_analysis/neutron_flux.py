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
"""Heat-flux profile integration and exponential decay-length fitting."""
import numpy as np
from scipy.optimize import curve_fit
def analyze_heat_flux(position_m, heat_flux_W_m2):
    x=np.asarray(position_m,dtype=float); q=np.asarray(heat_flux_W_m2,dtype=float)
    if x.ndim!=1 or q.shape!=x.shape or len(x)<3 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(q)) or np.any(np.diff(x)<=0): raise ValueError("Invalid heat-flux profile")
    peak=int(np.argmax(q)); out={"position_m":x,"heat_flux_W_m2":q,"peak_heat_flux_W_m2":float(q[peak]),"peak_position_m":float(x[peak]),"integrated_heat_flux_W_m":float(np.trapezoid(q,x))}
    m=q>0
    if np.count_nonzero(m)>=3:
        def fn(xx,q0,l,offset): return q0*np.exp(-(xx-x[m][0])/l)+offset
        try:
            p,_=curve_fit(fn,x[m],q[m],p0=[float(np.max(q[m])),max((x[-1]-x[0])/10,1e-9),max(0,float(np.min(q[m])))],bounds=([0,1e-12,-np.inf],[np.inf,np.inf,np.inf]),maxfev=20000)
            out["exponential_decay_length_m"]=float(p[1]); out["fit_parameters"]=p
        except Exception: out["exponential_decay_length_m"]=None
    out["caveat"]="Profile integration yields power per unit toroidal length unless geometry factors are included."
    return out
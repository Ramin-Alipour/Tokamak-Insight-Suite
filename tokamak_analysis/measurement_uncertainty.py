"""Measurement uncertainty propagation for common derived quantities."""
import numpy as np
def propagate_product_uncertainty(values, uncertainties):
    v=np.asarray(values,dtype=float); u=np.asarray(uncertainties,dtype=float)
    if v.ndim!=1 or u.shape!=v.shape or len(v)<1 or not np.all(np.isfinite(v)) or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Matching finite values and nonnegative uncertainties required")
    product=float(np.prod(v)); rel=np.sqrt(np.sum((u/np.maximum(np.abs(v),np.finfo(float).tiny))**2))
    return {"value":product,"standard_uncertainty":abs(product)*float(rel),"relative_uncertainty":float(rel)}
def uncertainty_summary(measured, uncertainty):
    x=np.asarray(measured,dtype=float); u=np.asarray(uncertainty,dtype=float)
    if x.shape!=u.shape or x.size<1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Matching finite data and nonnegative uncertainty required")
    return {"mean":float(np.mean(x)),"mean_standard_uncertainty":float(np.sqrt(np.sum(u**2))/x.size),"mean_uncertainty_if_independent_random":float(np.sqrt(np.sum(u**2))/x.size),"expanded_uncertainty_k2":float(2*np.sqrt(np.sum(u**2))/x.size)}
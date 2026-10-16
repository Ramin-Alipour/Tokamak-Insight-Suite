"""Measurement uncertainty propagation for common derived quantities."""
import numpy as np
def propagate_product_uncertainty(values, uncertainties):
    v=np.asarray(values,dtype=float); u=np.asarray(uncertainties,dtype=float)
    if v.ndim!=1 or u.shape!=v.shape or len(v)<1 or not np.all(np.isfinite(v)) or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Matching finite values and nonnegative uncertainties required")
    product=float(np.prod(v))
    # First-order independent uncertainty propagation remains valid when one or more factors are zero.
    derivatives=np.ones_like(v)
    for i in range(len(v)):
        derivatives[i]=float(np.prod(np.delete(v,i)))
    standard=float(np.sqrt(np.sum((derivatives*u)**2)))
    relative=standard/abs(product) if product!=0 else float("nan")
    return {"value":product,"standard_uncertainty":standard,"relative_uncertainty":float(relative),
            "equation":"u_product=sqrt(sum_i((u_i*product_{j!=i}(x_j))^2)); relative uncertainty undefined when product is zero"}
def uncertainty_summary(measured, uncertainty):
    x=np.asarray(measured,dtype=float); u=np.asarray(uncertainty,dtype=float)
    if x.shape!=u.shape or x.size<1 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Matching finite data and nonnegative uncertainty required")
    return {"mean":float(np.mean(x)),"mean_standard_uncertainty":float(np.sqrt(np.sum(u**2))/x.size),"mean_uncertainty_if_independent_random":float(np.sqrt(np.sum(u**2))/x.size),"expanded_uncertainty_k2":float(2*np.sqrt(np.sum(u**2))/x.size)}

def propagate_sum_uncertainty(values, uncertainties, *, correlation_matrix=None):
    """First-order uncertainty for a sum; independent case u_sum=sqrt(sum(u_i^2))."""
    v=np.asarray(values,dtype=float); u=np.asarray(uncertainties,dtype=float)
    if v.ndim!=1 or u.shape!=v.shape or len(v)<1 or not np.all(np.isfinite(v)) or not np.all(np.isfinite(u)) or np.any(u<0):
        raise ValueError("Matching finite values and non-negative uncertainties required")
    if correlation_matrix is None: variance=float(np.sum(u**2))
    else:
        corr=np.asarray(correlation_matrix,dtype=float)
        if corr.shape!=(len(v),len(v)) or not np.all(np.isfinite(corr)) or not np.allclose(corr,corr.T,atol=1e-10) or not np.allclose(np.diag(corr),1.0,atol=1e-10):
            raise ValueError("Correlation matrix must be finite, symmetric, and have unit diagonal")
        covariance=np.outer(u,u)*corr
        variance=float(np.sum(covariance))
        if variance < -1e-12: raise ValueError("Correlation matrix yields negative sum variance")
        variance=max(variance,0.0)
    return {"sum_value":float(np.sum(v)),"sum_standard_uncertainty":float(np.sqrt(variance)),
            "equation":"u_sum=sqrt(1^T Cov 1); independent: sqrt(sum(u_i^2))"}

"""Singular-value decomposition for multi-channel fluctuation data."""
import numpy as np
def analyze_svd(signals, *, rank=None, center=True):
    x=np.asarray(signals,dtype=float)
    if x.ndim!=2 or min(x.shape)<2 or not np.all(np.isfinite(x)): raise ValueError("Signals must be finite 2-D (samples, channels) matrix")
    mean=np.mean(x,axis=0) if center else np.zeros(x.shape[1]); xc=x-mean
    U,s,Vh=np.linalg.svd(xc,full_matrices=False)
    if rank is not None and (not np.isfinite(rank) or int(rank)!=rank): raise ValueError("rank must be an integer")
    k=len(s) if rank is None else int(rank)
    if k<1 or k>len(s): raise ValueError("rank outside valid range")
    reconstructed=(U[:,:k]*s[:k])@Vh[:k]+mean
    energy=s**2; frac=energy/np.sum(energy) if np.sum(energy)>0 else np.zeros_like(energy)
    return {"singular_values":s,"explained_variance_fraction":frac,"spatial_modes":Vh,"temporal_coefficients":U*s,"reconstruction_rank":k,"reconstruction":reconstructed,"residual_rms":float(np.sqrt(np.mean((x-reconstructed)**2)))}

def svd_truncation_frobenius_error(singular_values, rank):
    """Eckart-Young bound: best rank-k Frobenius residual is sqrt(sum_{j>k} sigma_j^2)."""
    s=np.asarray(singular_values,dtype=float)
    if not np.isfinite(rank) or int(rank)!=rank: raise ValueError("rank must be an integer")
    k=int(rank)
    if s.ndim!=1 or len(s)<1 or not np.all(np.isfinite(s)) or np.any(s<0) or np.any(np.diff(s)>1e-12):
        raise ValueError("Singular values must be finite, non-negative, and non-increasing")
    if k<0 or k>len(s): raise ValueError("rank must lie between 0 and number of singular values")
    tail=float(np.sum(s[k:]**2))
    total=float(np.sum(s**2))
    return {"minimum_frobenius_error":float(np.sqrt(tail)),"retained_energy_fraction":float(np.sum(s[:k]**2)/total) if total else 1.0,
            "equation":"||X-X_k||_F=sqrt(sum_{j>k} sigma_j^2)"}

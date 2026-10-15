"""Singular-value decomposition for multi-channel fluctuation data."""
import numpy as np
def analyze_svd(signals, *, rank=None, center=True):
    x=np.asarray(signals,dtype=float)
    if x.ndim!=2 or min(x.shape)<2 or not np.all(np.isfinite(x)): raise ValueError("Signals must be finite 2-D (samples, channels) matrix")
    mean=np.mean(x,axis=0) if center else np.zeros(x.shape[1]); xc=x-mean
    U,s,Vh=np.linalg.svd(xc,full_matrices=False); k=len(s) if rank is None else int(rank)
    if k<1 or k>len(s): raise ValueError("rank outside valid range")
    reconstructed=(U[:,:k]*s[:k])@Vh[:k]+mean
    energy=s**2; frac=energy/np.sum(energy) if np.sum(energy)>0 else np.zeros_like(energy)
    return {"singular_values":s,"explained_variance_fraction":frac,"spatial_modes":Vh,"temporal_coefficients":U*s,"reconstruction_rank":k,"reconstruction":reconstructed,"residual_rms":float(np.sqrt(np.mean((x-reconstructed)**2)))}
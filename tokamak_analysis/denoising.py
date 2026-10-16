"""Signal denoising with median and Savitzky-Golay filters."""
import numpy as np
from scipy.signal import medfilt, savgol_filter
def denoise_signal(signal, *, method="savgol", window_length=9, polynomial_order=2):
    x=np.asarray(signal,dtype=float)
    if x.ndim!=1 or len(x)<3 or not np.all(np.isfinite(x)): raise ValueError("Signal must be finite 1-D data")
    w=int(window_length)
    if w<3: raise ValueError("window_length must be >=3")
    if method=="median":
        if w%2==0: w+=1
        y=medfilt(x,kernel_size=min(w, len(x) if len(x)%2 else len(x)-1))
    elif method=="savgol":
        if w%2==0: w+=1
        w=min(w,len(x) if len(x)%2 else len(x)-1)
        poly=int(polynomial_order)
        if poly<0 or w<=poly: raise ValueError("polynomial_order must be non-negative and smaller than the filter window")
        y=savgol_filter(x,w,poly)
    else: raise ValueError("method must be median or savgol")
    return {"filtered_signal":y,"residual":x-y,"residual_rms":float(np.sqrt(np.mean((x-y)**2))),"method":method,"window_length":w}

def robust_noise_sigma_from_residual(residual):
    """Robust Gaussian-noise scale estimate sigma~=1.4826*median(|r-median(r)|)."""
    r=np.asarray(residual,dtype=float)
    if r.ndim!=1 or len(r)<3 or not np.all(np.isfinite(r)): raise ValueError("Residual must be finite 1-D data with at least 3 samples")
    med=float(np.median(r)); mad=float(np.median(np.abs(r-med)))
    return {"robust_noise_sigma":1.4826*mad,"residual_median":med,"mad":mad,
            "equation":"sigma_hat=1.4826*median(|r-median(r)|)",
            "assumption":"Consistent Gaussian-noise estimate when residuals are predominantly noise and not structured signal."}

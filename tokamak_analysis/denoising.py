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
        if w<=polynomial_order: raise ValueError("Window must exceed polynomial order")
        y=savgol_filter(x,w,int(polynomial_order))
    else: raise ValueError("method must be median or savgol")
    return {"filtered_signal":y,"residual":x-y,"residual_rms":float(np.sqrt(np.mean((x-y)**2))),"method":method,"window_length":w}
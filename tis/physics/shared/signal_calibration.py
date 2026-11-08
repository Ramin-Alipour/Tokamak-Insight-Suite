"""Linear/polynomial signal calibration and calibration residuals."""
import numpy as np
def calibrate_signal(raw, coefficients, *, uncertainty_raw=None):
    x=np.asarray(raw,dtype=float); c=np.asarray(coefficients,dtype=float)
    if not np.all(np.isfinite(x)) or c.ndim!=1 or len(c)<2 or not np.all(np.isfinite(c)): raise ValueError("Finite raw values and >=2 polynomial coefficients required")
    y=np.polyval(c,x); out={"calibrated":y,"coefficients_high_to_low":c}
    if uncertainty_raw is not None:
        u=np.asarray(uncertainty_raw,dtype=float)
        if u.shape!=x.shape or not np.all(np.isfinite(u)) or np.any(u<0): raise ValueError("Input uncertainty invalid")
        deriv=np.polyval(np.polyder(c),x); out["propagated_uncertainty"]=np.abs(deriv)*u
    return out
def calibration_residuals(reference, measured):
    r=np.asarray(reference,dtype=float); m=np.asarray(measured,dtype=float)
    if r.shape!=m.shape or r.ndim!=1 or len(r)<2 or not np.all(np.isfinite(r)) or not np.all(np.isfinite(m)): raise ValueError("Matching finite calibration vectors required")
    d=m-r
    return {"residual":d,"bias":float(np.mean(d)),"rmse":float(np.sqrt(np.mean(d**2))),"max_absolute_error":float(np.max(np.abs(d)))}

def fit_linear_calibration(raw_reference, physical_reference):
    """Least-squares affine calibration y=gain*x+offset from reference pairs."""
    x=np.asarray(raw_reference,dtype=float); y=np.asarray(physical_reference,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.ptp(x)==0:
        raise ValueError("Need at least two matching finite calibration points with varying raw values")
    gain,offset=np.polyfit(x,y,1); predicted=gain*x+offset; residual=y-predicted
    return {"gain":float(gain),"offset":float(offset),"predicted_reference":predicted,"residual":residual,"rmse":float(np.sqrt(np.mean(residual**2))),
            "equation":"y_calibrated=gain*x_raw+offset"}

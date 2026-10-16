"""Mirnov-coil magnetic fluctuation signal analysis."""
import numpy as np
from scipy.signal import welch, find_peaks
def analyze_mirnov_signal(time_s, voltage_V, *, coil_area_m2=None, turns=None):
    t=np.asarray(time_s,dtype=float); v=np.asarray(voltage_V,dtype=float)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<8 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or np.any(np.diff(t)<=0): raise ValueError("Invalid Mirnov signal")
    dt=np.median(np.diff(t))
    if not np.allclose(np.diff(t),dt,rtol=1e-3,atol=1e-12): raise ValueError("Uniform sampling required")
    f,p=welch(v-np.mean(v),fs=1/dt,nperseg=min(1024,len(v))); inds,_=find_peaks(p)
    inds=inds[np.argsort(p[inds])[::-1]][:8]
    out={"frequency_Hz":f,"power_spectral_density":p,"dominant_peaks_Hz":f[inds],"rms_voltage_V":float(np.std(v,ddof=1))}
    if coil_area_m2 is not None or turns is not None:
        try:
            area = float(coil_area_m2)
            turns_value = float(turns)
        except (TypeError, ValueError) as exc:
            raise ValueError("Finite positive coil area and positive integer turns required together") from exc
        if (coil_area_m2 is None or turns is None or not np.isfinite(area) or area <= 0
                or not np.isfinite(turns_value) or turns_value < 1 or not turns_value.is_integer()):
            raise ValueError("Finite positive coil area and positive integer turns required together")
        out["approx_Bdot_T_s"]=-v/(int(turns_value)*area)
    out["caveat"]="Coil transfer function, orientation, integration constant and calibration are not corrected automatically."
    return out

def magnetic_perturbation_from_mirnov_voltage(time_s, coil_voltage_V, turns, coil_area_m2, *, initial_delta_B_T=0.0):
    """Faraday law for a fixed coil: V=-N*A*dB/dt; integrate to recover delta-B."""
    t=np.asarray(time_s,dtype=float); v=np.asarray(coil_voltage_V,dtype=float); n=int(turns); area=float(coil_area_m2); b0=float(initial_delta_B_T)
    if t.ndim!=1 or v.shape!=t.shape or len(t)<2 or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v)) or np.any(np.diff(t)<=0):
        raise ValueError("Need finite matching voltage/time arrays and increasing time")
    if n<1 or not np.isfinite(area) or area<=0 or not np.isfinite(b0): raise ValueError("Turns and area must be positive; initial field finite")
    dbdt=-v/(n*area); increments=0.5*(dbdt[1:]+dbdt[:-1])*np.diff(t)
    db=np.r_[b0,b0+np.cumsum(increments)]
    return {"delta_B_T":db,"dB_dt_T_s":dbdt,"equation":"V=-N*A*dB/dt; delta_B=integral(-V/(N*A) dt)",
            "caveat":"Requires coil orientation, area, turns, transfer function and baseline offsets to be known."}

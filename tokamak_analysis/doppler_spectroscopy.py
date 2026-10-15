"""Spectral-line fitting and Doppler shift analysis for calibrated spectra."""
import numpy as np
from scipy.optimize import curve_fit
C_M_S=299792458.0
def analyze_doppler_spectrum(wavelength_nm,intensity,*,rest_wavelength_nm=None,model="gaussian"):
    x=np.asarray(wavelength_nm,dtype=float); y=np.asarray(intensity,dtype=float)
    if x.ndim!=1 or y.shape!=x.shape or len(x)<7 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x)<=0): raise ValueError("Spectrum requires finite increasing wavelength and matching intensity")
    baseline=float(np.percentile(y,10)); amp=float(np.max(y)-baseline); center0=float(x[np.argmax(y)])
    if amp<=0: raise ValueError("Spectrum has no positive line contrast")
    def gauss(z,a,c,s,b): return b+a*np.exp(-.5*((z-c)/s)**2)
    def lorentz(z,a,c,g,b): return b+a/(1+((z-c)/g)**2)
    fn=gauss if model.lower()=="gaussian" else lorentz if model.lower()=="lorentzian" else None
    if fn is None: raise ValueError("model must be gaussian or lorentzian")
    p0=[amp,center0,max(np.median(np.diff(x))*2,(x[-1]-x[0])/20),baseline]
    popt,_=curve_fit(fn,x,y,p0=p0,maxfev=20000)
    pred=fn(x,*popt); resid=y-pred; width=2.354820045*abs(popt[2]) if model.lower()=="gaussian" else 2*abs(popt[2])
    out={"center_nm":float(popt[1]),"amplitude":float(popt[0]),"width_fwhm_nm":float(width),"baseline":float(popt[3]),"r_squared":float(1-np.sum(resid**2)/np.sum((y-np.mean(y))**2)) if np.sum((y-np.mean(y))**2)>0 else float("nan"),"model":model}
    if rest_wavelength_nm is not None:
        rest=float(rest_wavelength_nm)
        if not np.isfinite(rest) or rest<=0: raise ValueError("Rest wavelength must be positive")
        out["doppler_shift_nm"]=out["center_nm"]-rest; out["nonrelativistic_velocity_m_s"]=C_M_S*(out["center_nm"]/rest-1)
    out["caveat"]="Input wavelengths must be instrument-calibrated; line-of-sight velocity is non-relativistic and depends on correct rest wavelength and line identification."
    return out
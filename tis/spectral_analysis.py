"""Spectral line fitting and fit-quality metrics."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import optimize
from scipy.special import voigt_profile

def _is_finite_real_scalar(value: Any) -> bool:
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return bool(np.isfinite(value))
    except (TypeError, ValueError, OverflowError):
        return False


def fit_spectral_line(x: Any, y: Any, xmin: float | None = None, xmax: float | None = None, model: str = 'Gaussian', sigma_y: Any | None = None) -> dict[str, float | str]:
    xx, yy = np.asarray(x,float).ravel(), np.asarray(y,float).ravel()
    if len(xx) != len(yy):
        raise ValueError("Spectral coordinates and intensity values must have equal lengths.")
    if ((xmin is not None and not _is_finite_real_scalar(xmin))
            or (xmax is not None and not _is_finite_real_scalar(xmax))):
        raise ValueError("Fit limits must be finite real numeric scalars when supplied.")
    if xmin is not None and xmax is not None and xmin >= xmax:
        raise ValueError("xmin must be smaller than xmax.")
    sigma_all=None
    if sigma_y is not None:
        sigma_all=np.asarray(sigma_y,float).ravel()
        if len(sigma_all)!=len(xx) or not np.all(np.isfinite(sigma_all)) or np.any(sigma_all<=0):
            raise ValueError("sigma_y must be finite, positive and match the original spectrum length.")
    mask = np.isfinite(xx)&np.isfinite(yy)
    if sigma_all is not None: mask &= np.isfinite(sigma_all)&(sigma_all>0)
    if xmin is not None: mask &= xx>=xmin
    if xmax is not None: mask &= xx<=xmax
    xx,yy=xx[mask],yy[mask]
    sigma_fit=None if sigma_all is None else sigma_all[mask]
    if len(xx)<7: raise ValueError('At least 7 valid points are required in the fit range.')
    sort=np.argsort(xx); xx,yy=xx[sort],yy[sort]
    if sigma_fit is not None: sigma_fit=sigma_fit[sort]
    if len(np.unique(xx)) != len(xx):
        raise ValueError("Spectral coordinates in the selected fit range must be unique.")
    span=float(np.ptp(xx))
    if span<=0: raise ValueError('Spectral coordinate must span a nonzero interval.')
    low=float(np.percentile(yy,10)); high=float(np.percentile(yy,90))
    emission_strength=float(np.max(yy)-low); absorption_strength=float(high-np.min(yy))
    if absorption_strength > emission_strength:
        baseline0=high; amp0=float(np.min(yy)-high); center0=float(xx[np.argmin(yy)]); polarity='absorption'
    else:
        baseline0=low; amp0=float(np.max(yy)-low); center0=float(xx[np.argmax(yy)]); polarity='emission'
    if abs(amp0)<=np.finfo(float).eps:
        raise ValueError("No resolvable emission or absorption feature is present in the selected fit range.")
    step=float(np.median(np.diff(xx)))
    if not isinstance(model, str):
        raise ValueError('Model must be a string: Gaussian, Lorentzian or Voigt.')
    model=model.lower()
    if model not in {"gaussian","lorentzian","voigt"}:
        raise ValueError('Model must be Gaussian, Lorentzian or Voigt.')
    if model=="voigt" and len(xx)<15:
        raise ValueError("Voigt fitting has five free parameters; at least 15 valid spectral points are required.")
    if model=='gaussian':
        def fn(z,b,a,c,s): return b+a*np.exp(-0.5*((z-c)/s)**2)
        p0=[baseline0,amp0,center0,max(step,span/20)]
        lo=[-np.inf,-np.inf,float(xx[0]),max(step/10,1e-15)]; hi=[np.inf,np.inf,float(xx[-1]),span*2]
        try:
            p,cov=optimize.curve_fit(fn,xx,yy,p0=p0,bounds=(lo,hi),sigma=sigma_fit,absolute_sigma=sigma_fit is not None,maxfev=30000)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            raise ValueError(f"Gaussian spectral fit failed to converge: {exc}") from exc
        if cov.shape!=(4,4) or not np.all(np.isfinite(cov)):
            raise ValueError("Gaussian fit covariance is singular or non-finite; inspect fit range and signal-to-noise.")
        b,a,c,s=map(float,p); pred=fn(xx,*p); width=2*np.sqrt(2*np.log(2))*s
        se=np.sqrt(np.maximum(np.diag(cov),0.0))
        return {'model':'Gaussian','baseline':b,'amplitude':a,'center':c,'sigma':s,'FWHM':float(width),'area_above_baseline':float(a*s*np.sqrt(2*np.pi)),'r_squared':_r2(yy,pred),'line_polarity':polarity,'uncertainty_weighted_fit':sigma_fit is not None,
                'standard_error_baseline':float(se[0]),'standard_error_amplitude':float(se[1]),'standard_error_center':float(se[2]),'standard_error_sigma':float(se[3]),
                'standard_error_FWHM':float(2*np.sqrt(2*np.log(2))*se[3])}
    if model=='lorentzian':
        def fn(z,b,a,c,g): return b+a/(1+((z-c)/g)**2)
        p0=[baseline0,amp0,center0,max(step,span/20)]
        lo=[-np.inf,-np.inf,float(xx[0]),max(step/10,1e-15)]; hi=[np.inf,np.inf,float(xx[-1]),span*2]
        try:
            p,cov=optimize.curve_fit(fn,xx,yy,p0=p0,bounds=(lo,hi),sigma=sigma_fit,absolute_sigma=sigma_fit is not None,maxfev=30000)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            raise ValueError(f"Lorentzian spectral fit failed to converge: {exc}") from exc
        if cov.shape!=(4,4) or not np.all(np.isfinite(cov)):
            raise ValueError("Lorentzian fit covariance is singular or non-finite; inspect fit range and signal-to-noise.")
        b,a,c,g=map(float,p); pred=fn(xx,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
        return {'model':'Lorentzian','baseline':b,'amplitude':a,'center':c,'half_width_at_half_max':g,'FWHM':2*g,'area_above_baseline':float(np.pi*a*g),'r_squared':_r2(yy,pred),'line_polarity':polarity,'uncertainty_weighted_fit':sigma_fit is not None,
                'standard_error_baseline':float(se[0]),'standard_error_amplitude':float(se[1]),'standard_error_center':float(se[2]),'standard_error_HWHM':float(se[3]),'standard_error_FWHM':float(2*se[3])}
    if model=='voigt':
        def fn(z,b,a,c,s,g): return b+a*voigt_profile(z-c,s,g)/max(voigt_profile(0,s,g),np.finfo(float).eps)
        p0=[baseline0,amp0,center0,max(step,span/30),max(step,span/30)]
        lo=[-np.inf,-np.inf,float(xx[0]),max(step/10,1e-15),max(step/10,1e-15)]; hi=[np.inf,np.inf,float(xx[-1]),span*2,span*2]
        try:
            p,cov=optimize.curve_fit(fn,xx,yy,p0=p0,bounds=(lo,hi),sigma=sigma_fit,absolute_sigma=sigma_fit is not None,maxfev=40000)
        except (RuntimeError, ValueError, FloatingPointError) as exc:
            raise ValueError(f"Voigt spectral fit failed to converge: {exc}") from exc
        if cov.shape!=(5,5) or not np.all(np.isfinite(cov)):
            raise ValueError("Voigt fit covariance is singular or non-finite; inspect fit range and signal-to-noise.")
        b,a,c,s,g=map(float,p); pred=fn(xx,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
        gaussian_fwhm=2*np.sqrt(2*np.log(2))*s
        lorentzian_fwhm=2*g
        fwhm=float(0.5346*lorentzian_fwhm+np.sqrt(0.2166*lorentzian_fwhm**2+gaussian_fwhm**2))
        area=float(a/max(voigt_profile(0,s,g),np.finfo(float).eps))
        return {'model':'Voigt','baseline':b,'amplitude_peak':a,'center':c,'gaussian_sigma':s,'lorentzian_HWHM':g,
                'FWHM':fwhm,'area_above_baseline':area,'r_squared':_r2(yy,pred),
                'standard_error_baseline':float(se[0]),'standard_error_amplitude_peak':float(se[1]),'standard_error_center':float(se[2]),
                'standard_error_gaussian_sigma':float(se[3]),'standard_error_lorentzian_HWHM':float(se[4]),
                'line_polarity':polarity,'uncertainty_weighted_fit':sigma_fit is not None}
    raise ValueError('Model must be Gaussian, Lorentzian or Voigt.')


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    den=float(np.sum((y-np.mean(y))**2))
    return float(1-np.sum((y-pred)**2)/den) if den>0 else float('nan')

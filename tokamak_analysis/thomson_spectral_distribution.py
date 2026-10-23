"""Distribution-level statistics for calibrated Thomson-scattering spectra."""
from __future__ import annotations
import numpy as np

def analyze_thomson_spectral_distribution(wavelength_nm, signal, *, background=None, signal_uncertainty=None, minimum_positive_bins=3):
    """Compute positive-net-signal area, centroid, RMS width, skewness and excess kurtosis.

    Negative background-subtracted bins remain in net_signal and are excluded
    from positive-weight moments. This descriptive routine does not infer Te/ne.
    """
    x=np.asarray(wavelength_nm,dtype=float); y=np.asarray(signal,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or x.size!=y.size or x.size<5:
        raise ValueError("wavelength_nm and signal must be matching 1-D arrays with at least 5 samples")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("wavelength_nm and signal must contain only finite values")
    if np.any(np.diff(x)<=0): raise ValueError("wavelength_nm must be strictly increasing")
    if isinstance(minimum_positive_bins,bool) or not isinstance(minimum_positive_bins,(int,np.integer)) or minimum_positive_bins<3:
        raise ValueError("minimum_positive_bins must be an integer >= 3")
    if background is None: bg=np.zeros_like(y)
    else:
        b=np.asarray(background,dtype=float)
        if b.ndim==0: bg=np.full_like(y,float(b))
        elif b.ndim==1 and b.size==y.size: bg=b
        else: raise ValueError("background must be a finite scalar or array matching signal")
        if not np.all(np.isfinite(bg)): raise ValueError("background must contain only finite values")
    unc=None
    if signal_uncertainty is not None:
        unc=np.asarray(signal_uncertainty,dtype=float)
        if unc.ndim!=1 or unc.size!=y.size or not np.all(np.isfinite(unc)) or np.any(unc<=0):
            raise ValueError("signal_uncertainty must contain positive finite values and match signal")
    net=y-bg; positive=np.maximum(net,0.0); count=int(np.count_nonzero(net>0))
    if count<minimum_positive_bins: raise ValueError(f"at least {minimum_positive_bins} positive net-signal bins are required")
    integrate=getattr(np,"trapezoid",np.trapz)
    area=float(integrate(positive,x))
    if not np.isfinite(area) or area<=0: raise ValueError("positive net spectral area must be finite and greater than zero")
    dx=np.diff(x); iw=np.empty_like(x); iw[0]=dx[0]/2; iw[-1]=dx[-1]/2; iw[1:-1]=(dx[:-1]+dx[1:])/2
    w=positive*iw; norm=float(np.sum(w)); centroid=float(np.sum(w*x)/norm); z=x-centroid
    var=float(np.sum(w*z**2)/norm)
    if not np.isfinite(var) or var<=0: raise ValueError("spectral distribution has zero or invalid variance")
    sigma=float(np.sqrt(var)); skew=float(np.sum(w*z**3)/norm/sigma**3); kurt=float(np.sum(w*z**4)/norm/var**2-3)
    area_unc=centroid_unc=None
    if unc is not None:
        area_unc=float(np.sqrt(np.sum((iw[net>0]*unc[net>0])**2)))
        centroid_unc=float(np.sqrt(np.sum((iw*(x-centroid)/norm*unc)**2)))
    return {"wavelength_nm":x.copy(),"signal":y.copy(),"background":bg.copy(),"net_signal":net.copy(),
      "positive_net_signal":positive.copy(),"integrated_positive_area":area,
      "integrated_positive_area_uncertainty":area_unc,"centroid_nm":centroid,
      "centroid_uncertainty_nm":centroid_unc,"rms_width_nm":sigma,"skewness":skew,
      "excess_kurtosis":kurt,"positive_bin_count":count,"negative_net_bin_count":int(np.count_nonzero(net<0)),
      "sample_count":int(x.size),"background_subtracted":background is not None,
      "uncertainty_model":"First-order independent-sample propagation for area and centroid only; excludes background uncertainty and positive-bin selection." if unc is not None else "No uncertainty supplied.",
      "limitations":["Descriptive moments do not establish a Maxwellian distribution.","Negative net bins are retained but excluded from positive-weight moments.",
      "No instrument-response deconvolution, stray-light correction, collective/relativistic model, or density inference.",
      "Centroid and width depend on background choice and spectral-window coverage."]}

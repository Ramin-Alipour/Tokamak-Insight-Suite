"""Numerical routines for Tokamak Insight Suite. Inputs and assumptions are explicit."""
from __future__ import annotations
import hashlib
import io
import json
import math
from typing import Any
import numpy as np
import pandas as pd
from scipy import signal, optimize, integrate, linalg
from scipy.special import voigt_profile, erfcx

E_CHARGE = 1.602176634e-19
K_B = 1.380649e-23
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6
M_E = 9.1093837139e-31
C_LIGHT = 299792458.0
AMU = 1.66053906660e-27


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_table(data: bytes, filename: str, sheet_name: str | int = 0) -> pd.DataFrame:
    name = filename.lower()
    if name.endswith(('.csv', '.txt')):
        # CSV first; fall back to common tab/semicolon separators.
        try:
            frame = pd.read_csv(io.BytesIO(data), sep=None, engine='python', comment='#')
        except Exception:
            frame = pd.read_csv(io.BytesIO(data), comment='#')
    elif name.endswith(('.xlsx', '.xls')):
        frame = pd.read_excel(io.BytesIO(data), sheet_name=sheet_name)
    elif name.endswith(('.parquet', '.pq')):
        frame = pd.read_parquet(io.BytesIO(data))
    else:
        raise ValueError('Supported formats: CSV, TXT, XLSX/XLS and Parquet (Parquet requires a compatible engine).')
    if frame.empty:
        raise ValueError('The imported table is empty.')
    frame.columns = [str(c).strip() or f'column_{i}' for i, c in enumerate(frame.columns)]
    return frame


def signal_quality(values: Any) -> dict[str, Any]:
    x = np.asarray(values, dtype=float).ravel()
    finite = np.isfinite(x)
    v = x[finite]
    if not len(v):
        return {'n_total': int(len(x)), 'n_finite': 0, 'n_missing_or_nonfinite': int(len(x)), 'error': 'No finite numeric samples.'}
    return {
        'n_total': int(len(x)), 'n_finite': int(len(v)), 'n_missing_or_nonfinite': int(len(x)-len(v)),
        'mean': float(np.mean(v)), 'std_sample': float(np.std(v, ddof=1)) if len(v)>1 else 0.0,
        'median': float(np.median(v)), 'min': float(np.min(v)), 'max': float(np.max(v)),
        'rms_including_dc': float(np.sqrt(np.mean(v*v))), 'peak_to_peak': float(np.ptp(v)),
        'relative_rms_fluctuation': float(np.std(v)/abs(np.mean(v))) if abs(np.mean(v))>np.finfo(float).eps else None,
        'duplicate_fraction': float((len(v)-len(np.unique(v)))/len(v)),
    }


def linear_calibration(values: Any, slope: float, intercept: float = 0.0) -> np.ndarray:
    if not np.isfinite(slope) or not np.isfinite(intercept):
        raise ValueError('Calibration slope and intercept must be finite.')
    return np.asarray(values, dtype=float) * slope + intercept


def polynomial_calibration(values: Any, coefficients: list[float]) -> np.ndarray:
    if not coefficients or not np.all(np.isfinite(coefficients)):
        raise ValueError('Provide finite polynomial coefficients, highest order first.')
    return np.polyval(np.asarray(coefficients, dtype=float), np.asarray(values, dtype=float))



def integrate_time_series(
    time_s: Any,
    values: Any,
    scale: float = 1.0,
    initial_value: float = 0.0,
    baseline_start_s: float | None = None,
    baseline_end_s: float | None = None,
) -> dict[str, Any]:
    """Integrate a calibrated time trace by the trapezoidal rule, optionally subtracting a measured baseline.

    The scale converts input units to the derivative/integrand units required by the user.
    This utility does not infer a sensor transfer function or physical calibration.
    """
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y) or len(t) < 2:
        raise ValueError('Time and signal must have equal lengths and at least two samples.')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError('Time and signal must contain only finite values.')
    if np.any(np.diff(t) <= 0):
        raise ValueError('Time values must be strictly increasing with no duplicates.')
    if not np.isfinite(scale) or not np.isfinite(initial_value):
        raise ValueError('Scale and initial value must be finite.')
    if (baseline_start_s is None) != (baseline_end_s is None):
        raise ValueError('Specify both baseline start and end, or neither.')
    calibrated = y * float(scale)
    baseline = 0.0
    if baseline_start_s is not None:
        if not np.isfinite(baseline_start_s) or not np.isfinite(baseline_end_s) or baseline_start_s >= baseline_end_s:
            raise ValueError('Baseline interval must be finite with start < end.')
        mask = (t >= baseline_start_s) & (t <= baseline_end_s)
        if np.count_nonzero(mask) < 2:
            raise ValueError('Baseline interval must include at least two samples.')
        baseline = float(np.mean(calibrated[mask]))
        calibrated = calibrated - baseline
    integrated = float(initial_value) + integrate.cumulative_trapezoid(calibrated, t, initial=0.0)
    return {
        'time_s': t,
        'calibrated_signal': calibrated,
        'integrated_signal': integrated,
        'scale': float(scale),
        'baseline_offset_in_calibrated_units': baseline,
        'initial_value': float(initial_value),
        'n_samples': int(len(t)),
        'method': 'cumulative trapezoidal integration',
        'note': 'The result is physically meaningful only when the input signal, scale, baseline interval, sensor response and initial condition are appropriate for the diagnostic.',
    }


def prepare_time_series(time: Any, values: Any, sample_rate: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(values, dtype=float).ravel()
    if time is None:
        if sample_rate is None or not np.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError('A positive sample rate is required when no time column is supplied.')
        t = np.arange(len(y), dtype=float) / sample_rate
    else:
        t = np.asarray(time, dtype=float).ravel()
        if len(t) != len(y):
            raise ValueError('Time and signal columns must have the same length.')
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if len(t) < 8:
        raise ValueError('At least 8 finite samples are required for time-series analysis.')
    order = np.argsort(t, kind='stable')
    t, y = t[order], y[order]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError('Time values must be strictly increasing after sorting; duplicate timestamps need handling.')
    med_dt = float(np.median(dt))
    if not np.isfinite(med_dt) or med_dt <= 0:
        raise ValueError('Could not infer a valid sampling interval.')
    # Detect isolated dropped samples as well as globally jittery clocks.
    jitter = float(np.std(dt) / np.mean(dt)) if len(dt)>1 else 0.0
    max_gap_ratio = float(np.max(dt) / med_dt)
    outlier_fraction = float(np.mean(np.abs(dt - med_dt) > 0.05 * med_dt))
    if max_gap_ratio > 1.05 or outlier_fraction > 0.01 or jitter > 0.02:
        raise ValueError(
            "Irregular sampling or a timestamp gap was detected "
            f"(std/mean={jitter:.3g}, max_dt/median_dt={max_gap_ratio:.3g}, "
            f"intervals outside ±5%={outlier_fraction:.2%}). "
            "Repair/resample explicitly before FFT/filtering."
        )
    return t, y, 1.0/med_dt


def butterworth_filter(values: Any, fs: float, highpass: float | None = None, lowpass: float | None = None, order: int = 3) -> np.ndarray:
    x = np.asarray(values, dtype=float).ravel()
    if isinstance(order, bool) or int(order) != order or order < 1:
        raise ValueError('Filter order must be a positive integer.')
    order = int(order)
    if len(x) < max(12, 3*order+1):
        raise ValueError('Signal is too short for the selected filter.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling frequency must be positive and finite.')
    if not np.all(np.isfinite(x)):
        raise ValueError('Signal must contain only finite values; handle missing samples before filtering.')
    nyq = fs/2
    if highpass is not None and (not np.isfinite(highpass) or highpass <= 0):
        raise ValueError('High-pass cutoff must be positive and finite when supplied.')
    if lowpass is not None and (not np.isfinite(lowpass) or lowpass <= 0):
        raise ValueError('Low-pass cutoff must be positive and finite when supplied.')
    if highpass is None and lowpass is None:
        return x.copy()
    if highpass is not None and (highpass >= nyq or not np.isfinite(highpass)):
        raise ValueError('High-pass cutoff must be between 0 and Nyquist.')
    if lowpass is not None and (lowpass >= nyq or not np.isfinite(lowpass)):
        raise ValueError('Low-pass cutoff must be between 0 and Nyquist.')
    if highpass is not None and lowpass is not None:
        if highpass >= lowpass: raise ValueError('High-pass cutoff must be below low-pass cutoff.')
        wn, btype = [highpass, lowpass], 'bandpass'
    elif highpass is not None: wn, btype = highpass, 'highpass'
    else: wn, btype = lowpass, 'lowpass'
    sos = signal.butter(int(order), wn, btype=btype, fs=fs, output='sos')
    return signal.sosfiltfilt(sos, x)


def fft_spectrum(values: Any, fs: float, window: str = 'hann', detrend: bool = True) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(values, dtype=float).ravel()
    if len(x) < 2 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Need at least two samples and a positive finite sampling rate.')
    if not np.all(np.isfinite(x)):
        raise ValueError('Signal must contain only finite values; clean missing samples before FFT.')
    # Preserve the DC component when detrend=False; otherwise remove the mean.
    if detrend:
        x = signal.detrend(x)
    w = signal.get_window(window, len(x), fftbins=True)
    z = np.fft.rfft(x*w)
    f = np.fft.rfftfreq(len(x), 1/fs)
    amp = np.abs(z) / max(np.sum(w)/2, np.finfo(float).eps)
    if len(amp): amp[0] *= 0.5
    if len(x)%2==0 and len(amp)>1: amp[-1]*=0.5
    return f, amp


def welch_psd(values: Any, fs: float, nperseg: int | None = None,
              noverlap: int | None = None, window: str = 'hann') -> tuple[np.ndarray, np.ndarray]:
    """Estimate PSD with explicit segment length, overlap, and window choices."""
    x = np.asarray(values, dtype=float).ravel()
    if len(x) < 8 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('At least 8 samples and a positive finite sampling rate are required.')
    if not np.all(np.isfinite(x)):
        raise ValueError('Signal must contain only finite values; clean missing samples before PSD analysis.')
    if nperseg is not None and (isinstance(nperseg, bool) or int(nperseg) != nperseg or nperseg < 8):
        raise ValueError('nperseg must be an integer of at least 8 when supplied.')
    nperseg = min(len(x), int(nperseg) if nperseg is not None else min(4096, max(8, len(x)//4)))
    if noverlap is not None and (isinstance(noverlap, bool) or int(noverlap) != noverlap or noverlap < 0 or noverlap >= nperseg):
        raise ValueError('noverlap must be an integer in [0, nperseg).')
    if not isinstance(window, str) or not window.strip():
        raise ValueError('window must be a valid SciPy window name.')
    overlap = int(noverlap) if noverlap is not None else nperseg // 2
    try:
        return signal.welch(x, fs=fs, window=window, nperseg=nperseg, noverlap=overlap,
                            detrend='constant', scaling='density')
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Invalid Welch window or configuration: {exc}') from exc




def autocorrelation_analysis(values: Any, fs: float, max_lag_s: float | None = None) -> dict[str, Any]:
    """Compute normalized autocorrelation and its first positive 1/e decay time."""
    x = np.asarray(values, dtype=float).ravel()
    if len(x) < 8 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('At least 8 samples and a positive finite sampling rate are required.')
    if not np.all(np.isfinite(x)):
        raise ValueError('Signal must contain only finite values.')
    if max_lag_s is not None and (not np.isfinite(max_lag_s) or max_lag_s < 0):
        raise ValueError('Maximum lag must be finite and nonnegative.')
    x = x - np.mean(x)
    energy = float(np.dot(x, x))
    if energy <= np.finfo(float).tiny:
        raise ValueError('Autocorrelation is undefined for a constant or numerically flat signal.')
    corr = signal.correlate(x, x, mode='full', method='auto')[len(x)-1:] / energy
    lags = np.arange(len(x), dtype=float) / fs
    if max_lag_s is not None:
        keep = lags <= max_lag_s
        lags, corr = lags[keep], corr[keep]
    crossing = np.flatnonzero(corr <= np.exp(-1))
    tau = float(lags[crossing[0]]) if len(crossing) else None
    return {
        'lag_s': lags,
        'autocorrelation': corr,
        'e_folding_time_s': tau,
        'n_samples': int(len(x)),
        'note': 'The first 1/e crossing is a descriptive correlation-time estimate; periodic, noisy, nonstationary or short records require careful interpretation.',
    }


def time_frequency_spectrogram(
    values: Any,
    fs: float,
    nperseg: int = 256,
    noverlap: int | None = None,
    window: str = 'hann',
) -> dict[str, np.ndarray]:
    """Return a Welch-style short-time power spectrum for transient/frequency-evolving signals."""
    x = np.asarray(values, dtype=float).ravel()
    if len(x) < 8 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('At least 8 samples and a positive finite sampling rate are required.')
    if not np.all(np.isfinite(x)):
        raise ValueError('Signal must contain only finite values; clean missing samples before spectrogram analysis.')
    if isinstance(nperseg, (bool, np.bool_)) or int(nperseg) != nperseg or nperseg < 8:
        raise ValueError('nperseg must be an integer of at least 8.')
    nperseg = min(int(nperseg), len(x))
    if noverlap is not None:
        if isinstance(noverlap, (bool, np.bool_)) or int(noverlap) != noverlap or noverlap < 0 or noverlap >= nperseg:
            raise ValueError('noverlap must be an integer in [0, nperseg).')
        noverlap = int(noverlap)
    f, t, pxx = signal.spectrogram(
        x, fs=fs, window=window, nperseg=nperseg, noverlap=noverlap,
        detrend='constant', scaling='density', mode='psd'
    )
    return {'frequency_Hz': f, 'time_s': t, 'psd': pxx}


def cross_analysis(x: Any, y: Any, fs: float, nperseg: int | None = None) -> dict[str, np.ndarray]:
    a, b = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
    if len(a) != len(b) or len(a) < 8:
        raise ValueError('Signals must have equal length and at least 8 samples.')
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Sampling frequency must be positive and finite.')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Both signals must contain only finite values.')
    if nperseg is not None and (isinstance(nperseg, bool) or int(nperseg) != nperseg or nperseg < 1):
        raise ValueError('nperseg must be a positive integer when supplied.')
    n = min(len(a), int(nperseg) if nperseg is not None else min(1024, max(8, len(a)//4)))
    f, coh = signal.coherence(a,b,fs=fs,nperseg=n)
    fc, pxy = signal.csd(a,b,fs=fs,nperseg=n)
    return {'frequency_Hz':f, 'coherence':coh, 'cross_phase_rad':np.angle(pxy), 'cross_spectrum_real':pxy.real, 'cross_spectrum_imag':pxy.imag}


def cross_correlation_lag(x: Any, y: Any, fs: float, max_lag_s: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    a, b = np.asarray(x,float).ravel(), np.asarray(y,float).ravel()
    if len(a) != len(b) or len(a) < 3 or not np.isfinite(fs) or fs <= 0:
        raise ValueError('Equal-length signals and a positive finite sampling rate are required.')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Both signals must contain only finite values.')
    if max_lag_s is not None and (not np.isfinite(max_lag_s) or max_lag_s < 0):
        raise ValueError('Maximum lag must be finite and nonnegative when supplied.')
    a, b = signal.detrend(a), signal.detrend(b)
    corr = signal.correlate(a,b,mode='full',method='auto')
    corr /= max(np.linalg.norm(a)*np.linalg.norm(b), np.finfo(float).eps)
    lags = signal.correlation_lags(len(a),len(b),mode='full') / fs
    if max_lag_s is not None:
        keep = np.abs(lags)<=max_lag_s
        lags,corr=lags[keep],corr[keep]
    idx=int(np.argmax(np.abs(corr)))
    return lags,corr,float(lags[idx])


def fit_spectral_line(x: Any, y: Any, xmin: float | None = None, xmax: float | None = None, model: str = 'Gaussian', sigma_y: Any | None = None) -> dict[str, float | str]:
    xx, yy = np.asarray(x,float).ravel(), np.asarray(y,float).ravel()
    if (xmin is not None and not np.isfinite(xmin)) or (xmax is not None and not np.isfinite(xmax)):
        raise ValueError("Fit limits must be finite when supplied.")
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
    model=model.lower()
    if model=='gaussian':
        def fn(z,b,a,c,s): return b+a*np.exp(-0.5*((z-c)/s)**2)
        p0=[baseline0,amp0,center0,max(step,span/20)]
        lo=[-np.inf,-np.inf,float(xx[0]),max(step/10,1e-15)]; hi=[np.inf,np.inf,float(xx[-1]),span*2]
        p,cov=optimize.curve_fit(fn,xx,yy,p0=p0,bounds=(lo,hi),sigma=sigma_fit,absolute_sigma=sigma_fit is not None,maxfev=30000)
        b,a,c,s=map(float,p); pred=fn(xx,*p); width=2*np.sqrt(2*np.log(2))*s
        se=np.sqrt(np.maximum(np.diag(cov),0.0))
        return {'model':'Gaussian','baseline':b,'amplitude':a,'center':c,'sigma':s,'FWHM':float(width),'area_above_baseline':float(a*s*np.sqrt(2*np.pi)),'r_squared':_r2(yy,pred),'line_polarity':polarity,'uncertainty_weighted_fit':sigma_fit is not None,
                'standard_error_baseline':float(se[0]),'standard_error_amplitude':float(se[1]),'standard_error_center':float(se[2]),'standard_error_sigma':float(se[3]),
                'standard_error_FWHM':float(2*np.sqrt(2*np.log(2))*se[3])}
    if model=='lorentzian':
        def fn(z,b,a,c,g): return b+a/(1+((z-c)/g)**2)
        p0=[baseline0,amp0,center0,max(step,span/20)]
        lo=[-np.inf,-np.inf,float(xx[0]),max(step/10,1e-15)]; hi=[np.inf,np.inf,float(xx[-1]),span*2]
        p,cov=optimize.curve_fit(fn,xx,yy,p0=p0,bounds=(lo,hi),sigma=sigma_fit,absolute_sigma=sigma_fit is not None,maxfev=30000)
        b,a,c,g=map(float,p); pred=fn(xx,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
        return {'model':'Lorentzian','baseline':b,'amplitude':a,'center':c,'half_width_at_half_max':g,'FWHM':2*g,'area_above_baseline':float(np.pi*a*g),'r_squared':_r2(yy,pred),'line_polarity':polarity,'uncertainty_weighted_fit':sigma_fit is not None,
                'standard_error_baseline':float(se[0]),'standard_error_amplitude':float(se[1]),'standard_error_center':float(se[2]),'standard_error_HWHM':float(se[3]),'standard_error_FWHM':float(2*se[3])}
    if model=='voigt':
        def fn(z,b,a,c,s,g): return b+a*voigt_profile(z-c,s,g)/max(voigt_profile(0,s,g),np.finfo(float).eps)
        p0=[baseline0,amp0,center0,max(step,span/30),max(step,span/30)]
        lo=[-np.inf,-np.inf,float(xx[0]),max(step/10,1e-15),max(step/10,1e-15)]; hi=[np.inf,np.inf,float(xx[-1]),span*2,span*2]
        p,cov=optimize.curve_fit(fn,xx,yy,p0=p0,bounds=(lo,hi),sigma=sigma_fit,absolute_sigma=sigma_fit is not None,maxfev=40000)
        b,a,c,s,g=map(float,p); pred=fn(xx,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
        return {'model':'Voigt','baseline':b,'amplitude_peak':a,'center':c,'gaussian_sigma':s,'lorentzian_HWHM':g,'r_squared':_r2(yy,pred),
                'standard_error_baseline':float(se[0]),'standard_error_amplitude_peak':float(se[1]),'standard_error_center':float(se[2]),
                'standard_error_gaussian_sigma':float(se[3]),'standard_error_lorentzian_HWHM':float(se[4]),'line_polarity':polarity,'uncertainty_weighted_fit':sigma_fit is not None}
    raise ValueError('Model must be Gaussian, Lorentzian or Voigt.')


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    den=float(np.sum((y-np.mean(y))**2))
    return float(1-np.sum((y-pred)**2)/den) if den>0 else float('nan')


def doppler_temperature_eV(sigma_lambda: float, lambda0: float, mass_amu: float) -> float:
    if not np.all(np.isfinite([sigma_lambda, lambda0, mass_amu])) or sigma_lambda < 0 or lambda0 <= 0 or mass_amu <= 0:
        raise ValueError('Width must be nonnegative; wavelength and mass must be positive; all inputs must be finite.')
    return float((mass_amu*AMU*C_LIGHT**2/E_CHARGE)*(sigma_lambda/lambda0)**2)


def doppler_temperature_with_uncertainty_eV(
    sigma_lambda: float, lambda0: float, mass_amu: float,
    sigma_sigma_lambda: float = 0.0, sigma_lambda0: float = 0.0,
    sigma_mass_amu: float = 0.0,
) -> dict[str, float | str]:
    """Estimate Doppler temperature and first-order 1-sigma uncertainty.

    Inputs and their uncertainties must use the same wavelength units. Independent
    uncertainties are propagated in quadrature; model/systematic errors are not included.
    """
    vals=[sigma_lambda,lambda0,mass_amu,sigma_sigma_lambda,sigma_lambda0,sigma_mass_amu]
    if not np.all(np.isfinite(vals)) or sigma_lambda < 0 or lambda0 <= 0 or mass_amu <= 0:
        raise ValueError("Width must be nonnegative; wavelength and mass positive; inputs finite.")
    if min(sigma_sigma_lambda,sigma_lambda0,sigma_mass_amu) < 0:
        raise ValueError("Input uncertainties must be finite and nonnegative.")
    temperature=doppler_temperature_eV(sigma_lambda,lambda0,mass_amu)
    variance_terms=[]
    if sigma_lambda > 0:
        variance_terms.append((2.0*sigma_sigma_lambda/sigma_lambda)**2)
    elif sigma_sigma_lambda > 0:
        raise ValueError("First-order uncertainty propagation is undefined at zero Doppler width; use a profile-likelihood or Monte Carlo interval.")
    variance_terms.append((2.0*sigma_lambda0/lambda0)**2)
    variance_terms.append((sigma_mass_amu/mass_amu)**2)
    relative=float(np.sqrt(sum(variance_terms)))
    return {"temperature_eV":temperature,"standard_uncertainty_eV":float(temperature*relative),
            "relative_standard_uncertainty":relative,
            "uncertainty_method":"First-order independent-error propagation; excludes model/systematic uncertainty."}


def doppler_velocity_with_uncertainty_m_s(
    observed_lambda: float, lambda0: float,
    sigma_observed_lambda: float = 0.0, sigma_lambda0: float = 0.0,
) -> dict[str, float | str]:
    """Calculate nonrelativistic line-of-sight velocity and propagated 1-sigma uncertainty.

    Wavelengths and their uncertainties must use the same units. Independent measurement
    uncertainties are propagated in quadrature; systematic wavelength-calibration errors
    are not included unless explicitly represented in the supplied uncertainties.
    """
    vals=[observed_lambda,lambda0,sigma_observed_lambda,sigma_lambda0]
    if not np.all(np.isfinite(vals)) or observed_lambda <= 0 or lambda0 <= 0:
        raise ValueError("Observed and rest wavelengths must be positive and all inputs finite.")
    if sigma_observed_lambda < 0 or sigma_lambda0 < 0:
        raise ValueError("Wavelength uncertainties must be nonnegative.")
    velocity=float(C_LIGHT*(observed_lambda/lambda0-1.0))
    uncertainty=float(C_LIGHT*np.sqrt((sigma_observed_lambda/lambda0)**2+
                                     (observed_lambda*sigma_lambda0/lambda0**2)**2))
    return {"line_of_sight_velocity_m_s":velocity,
            "standard_uncertainty_m_s":uncertainty,
            "uncertainty_method":"First-order independent-error propagation; nonrelativistic Doppler approximation."}


def doppler_velocity_m_s(delta_lambda: float, lambda0: float) -> float:
    if not np.all(np.isfinite([delta_lambda, lambda0])) or lambda0 <= 0:
        raise ValueError('Wavelength shift must be finite and rest wavelength positive.')
    return float(C_LIGHT*delta_lambda/lambda0)


def linear_phase_density(phase_rad: Any, frequency_hz: float, sign_convention: int = -1) -> np.ndarray:
    """Return line-integrated electron density ∫n_e dl in m⁻², not local density.

    sign_convention must be +1 or -1 and reflects the instrument's phase convention:
    Δφ = sign_convention * r_e * wavelength * ∫n_e dl.
    """
    phase=np.asarray(phase_rad,float)
    if not np.isfinite(frequency_hz) or frequency_hz<=0 or not np.all(np.isfinite(phase)):
        raise ValueError('Frequency must be positive and finite; phase values must be finite.')
    if sign_convention not in (-1,1):
        raise ValueError('sign_convention must be either -1 or +1.')
    r_e=2.8179403262e-15
    wavelength=C_LIGHT/frequency_hz
    return phase/(sign_convention*r_e*wavelength)


def derive_parameters(ne_m3: float, te_eV: float, ti_eV: float, ip_A: float, vloop_V: float,
                      major_radius_m: float, minor_radius_m: float, bt_T: float,
                      stored_energy_J: float | None = None, loss_power_W: float | None = None,
                      ni_m3: float | None = None, ion_mass_amu: float = 2.0) -> dict[str, float | None]:
    vals=[ne_m3,te_eV,ti_eV,ip_A,vloop_V,major_radius_m,minor_radius_m,bt_T,ion_mass_amu]
    if not np.all(np.isfinite(vals)) or min(ne_m3,te_eV,ti_eV,major_radius_m,minor_radius_m)<0 or major_radius_m==0 or minor_radius_m==0 or ion_mass_amu<=0:
        raise ValueError('Inputs must be finite; densities/temperatures nonnegative, radii and ion mass positive. Ip and Bt are signed quantities.')
    if ni_m3 is not None and (not np.isfinite(ni_m3) or ni_m3 < 0):
        raise ValueError('Ion density must be finite and nonnegative.')
    if stored_energy_J is not None and (not np.isfinite(stored_energy_J) or stored_energy_J < 0):
        raise ValueError('Stored energy must be finite and nonnegative when supplied.')
    if loss_power_W is not None and (not np.isfinite(loss_power_W) or loss_power_W < 0):
        raise ValueError('Loss power must be finite and nonnegative when supplied.')
    ni=ne_m3 if ni_m3 is None else ni_m3
    pe=ne_m3*te_eV*E_CHARGE; pi=ni*ti_eV*E_CHARGE; p=pe+pi
    fpe=float(np.sqrt(ne_m3*E_CHARGE**2/(EPS0*M_E))/(2*np.pi)) if ne_m3>0 else 0.0
    debye=float(np.sqrt(EPS0*te_eV*E_CHARGE/(ne_m3*E_CHARGE**2))) if ne_m3>0 and te_eV>0 else None
    ng=abs(ip_A)/1e6/(np.pi*minor_radius_m**2)
    p_ohm=vloop_V*ip_A
    beta_t=p/(bt_T**2/(2*MU0)) if abs(bt_T)>0 else None
    tau=stored_energy_J/loss_power_W if stored_energy_J is not None and loss_power_W is not None and loss_power_W>0 else None
    cs=float(np.sqrt(E_CHARGE*(te_eV+ti_eV)/ (ion_mass_amu*AMU))) if te_eV+ti_eV>0 else 0.0
    return {'electron_pressure_Pa':float(pe),'ion_pressure_Pa':float(pi),'total_thermal_pressure_Pa':float(p),
            'electron_plasma_frequency_Hz':fpe,'debye_length_m':debye,'greenwald_density_1e20_m-3':float(ng),
            'ohmic_power_approx_W':float(p_ohm),'toroidal_beta_fraction':float(beta_t) if beta_t is not None else None,
            'energy_confinement_time_s':float(tau) if tau is not None else None,'ion_sound_speed_approx_m_s':cs,
            'assumption_ion_density_equals_electron_density':bool(ni_m3 is None)}


def uncertainty_linear(y: Any, slope: float, intercept: float, sigma_y: float = 0.0,
                       sigma_slope: float = 0.0, sigma_intercept: float = 0.0) -> tuple[np.ndarray,np.ndarray]:
    if not np.all(np.isfinite([slope, intercept, sigma_y, sigma_slope, sigma_intercept])) or min(sigma_y, sigma_slope, sigma_intercept) < 0:
        raise ValueError('Calibration coefficients and uncertainties must be finite; uncertainties must be nonnegative.')
    x=np.asarray(y,float)
    if not np.all(np.isfinite(x)):
        raise ValueError('Input values must be finite; clean missing/nonfinite samples before calibration.')
    cal=slope*x+intercept
    var=(slope*sigma_y)**2+(x*sigma_slope)**2+sigma_intercept**2
    return cal,np.sqrt(np.maximum(var,0))


def regularized_tomography(A: Any, y: Any, lam: float = 1e-2, nonnegative: bool = True) -> dict[str, Any]:
    """Solve min ||Ax-y||² + lam ||x||². Optional nonnegative constrained least squares."""
    A=np.asarray(A,float); y=np.asarray(y,float).ravel()
    if A.ndim!=2 or A.shape[0]!=len(y) or A.size==0: raise ValueError('A must be a nonempty 2D matrix with one row per measurement.')
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(y)): raise ValueError('Response matrix and measurements must contain only finite values.')
    if lam<0 or not np.isfinite(lam): raise ValueError('Regularization must be finite and nonnegative.')
    if nonnegative:
        augA=np.vstack([A,np.sqrt(lam)*np.eye(A.shape[1])]) if lam else A
        augy=np.concatenate([y,np.zeros(A.shape[1])]) if lam else y
        res=optimize.lsq_linear(augA,augy,bounds=(0,np.inf),method='trf')
        x=res.x; success=bool(res.success); message=str(res.message)
    else:
        if lam > 0:
            augA=np.vstack([A,np.sqrt(lam)*np.eye(A.shape[1])])
            augy=np.concatenate([y,np.zeros(A.shape[1])])
        else:
            augA,augy=A,y
        x=np.linalg.lstsq(augA,augy,rcond=None)[0]
        success=True; message='least-squares / Tikhonov solution'
    residual=A@x-y
    return {'solution':x,'residual':residual,'residual_norm':float(np.linalg.norm(residual)),
            'regularization':float(lam),'success':success,'message':message,'nonnegative':bool(nonnegative)}


def spectral_calibration(raw_pixel: Any, reference_wavelength: Any, degree: int = 1) -> dict[str, Any]:
    if isinstance(degree, (bool, np.bool_)) or not isinstance(degree, (int, np.integer)) or degree < 1:
        raise ValueError('Polynomial degree must be a positive integer.')
    x=np.asarray(raw_pixel,float).ravel(); y=np.asarray(reference_wavelength,float).ravel()
    if len(x) != len(y): raise ValueError('Pixel and reference-wavelength arrays must have equal length.')
    good=np.isfinite(x)&np.isfinite(y); x,y=x[good],y[good]
    if len(x)<degree+2: raise ValueError('Need at least degree+2 finite calibration points.')
    if len(np.unique(x))<degree+1: raise ValueError('Pixel coordinates do not span enough unique points for this polynomial degree.')
    coeff=np.polyfit(x,y,degree); pred=np.polyval(coeff,x); resid=y-pred
    return {'coefficients_high_to_low':coeff.tolist(),'rms_residual':float(np.sqrt(np.mean(resid**2))),
            'max_abs_residual':float(np.max(np.abs(resid))),'n_lines':int(len(x)),'degree':int(degree),
            'calibrated_values':np.polyval(coeff,x).tolist(),'residuals':resid.tolist()}


def probe_iv_summary(voltage: Any, current: Any) -> dict[str, Any]:
    """Descriptive I-V preprocessing only; does not silently infer Te/ne without a validated regime model."""
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    good=np.isfinite(v)&np.isfinite(i); v,i=v[good],i[good]
    if len(v)<5: raise ValueError('At least 5 finite I-V points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    if len(np.unique(v))<5: raise ValueError('Voltage must contain at least 5 unique values.')
    dIdV=np.gradient(i,v)
    return {'n_points':int(len(v)),'voltage_min_V':float(v.min()),'voltage_max_V':float(v.max()),
            'current_min_A':float(i.min()),'current_max_A':float(i.max()),
            'floating_potential_interpolated_V':float(np.interp(0, i[np.argsort(i)], v[np.argsort(i)])) if (np.all(np.diff(i)>=0) or np.all(np.diff(i)<=0)) and (i.min()<=0<=i.max()) else None,
            'max_dIdV_A_per_V':float(np.max(dIdV)),'min_dIdV_A_per_V':float(np.min(dIdV)),
            'warning':'Descriptive summary only. Plasma potential, Te and density require a justified probe regime/sheath model, collection geometry and uncertainty.'}

# Expanded diagnostic workflows (v0.3): calculations operate on user-supplied,
# calibrated data. They do not invent device calibration or geometry.
def langmuir_iv_analysis(voltage: Any, current: Any, electron_fit_range: tuple[float,float] | None = None,
                         ion_fit_range: tuple[float,float] | None = None, area_m2: float | None = None,
                         ion_mass_amu: float | None = None, ti_eV: float = 0.0,
                         sheath_factor: float = 0.61, gamma_i: float = 1.0) -> dict[str, Any]:
    """Analyze a single Langmuir I-V sweep with explicit model-input validation."""
    for name, fit_range in (("electron_fit_range", electron_fit_range), ("ion_fit_range", ion_fit_range)):
        if fit_range is not None:
            if len(fit_range) != 2 or not np.all(np.isfinite(np.asarray(fit_range, dtype=float))):
                raise ValueError(f"{name} must contain exactly two finite voltage limits.")
    if (area_m2 is None) != (ion_mass_amu is None):
        raise ValueError("area_m2 and ion_mass_amu must be supplied together for density estimation.")
    if area_m2 is not None and (not np.isfinite(area_m2) or area_m2 <= 0):
        raise ValueError("Collection area must be positive and finite.")
    if ion_mass_amu is not None and (not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0):
        raise ValueError("Ion mass must be positive and finite.")
    if not np.all(np.isfinite([ti_eV, sheath_factor, gamma_i])) or ti_eV < 0 or sheath_factor <= 0 or gamma_i < 0:
        raise ValueError("ti_eV and gamma_i must be finite and nonnegative; sheath_factor must be positive and finite.")
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    good=np.isfinite(v)&np.isfinite(i); v,i=v[good],i[good]
    if len(v)<8 or len(np.unique(v))<8: raise ValueError('At least 8 distinct valid voltage points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    out=probe_iv_summary(v,i)
    # Handle exact zero-current samples explicitly; avoid 0/0 for repeated zero values.
    exact_zero=np.flatnonzero(i == 0)
    cross=np.where(i[:-1]*i[1:]<0)[0]
    if len(exact_zero):
        out['floating_potential_V']=float(v[exact_zero[0]])
    elif len(cross):
        k=int(cross[0])
        out['floating_potential_V']=float(v[k]-i[k]*(v[k+1]-v[k])/(i[k+1]-i[k]))
    else:
        out['floating_potential_V']=None
    didv=np.gradient(i,v); out['max_dIdV_A_per_V']=float(np.max(didv)); out['plasma_potential_proxy_V']=float(v[np.argmax(didv)])
    if ion_fit_range is not None:
        m=(v>=min(ion_fit_range))&(v<=max(ion_fit_range))
        if m.sum()>=3:
            p=np.polyfit(v[m],i[m],1); out['ion_saturation_linear_slope_A_per_V']=float(p[0]); out['ion_saturation_intercept_A']=float(p[1]); out['ion_saturation_fit_R2']=_r2(i[m],np.polyval(p,v[m]))
            out['ion_saturation_current_A']=float(np.median(i[m]))
        else:
            raise ValueError("ion_fit_range must contain at least 3 valid samples.")
    if electron_fit_range is not None:
        ion_baseline=float(out.get('ion_saturation_current_A',0.0))
        electron_current=i-ion_baseline
        m=(v>=min(electron_fit_range))&(v<=max(electron_fit_range))&(electron_current>0)
        if m.sum()<4: raise ValueError('Electron-retarding fit range needs at least 4 points with positive current after ion-current baseline subtraction.')
        p=np.polyfit(v[m],np.log(electron_current[m]),1); pred=np.polyval(p,v[m]); slope=float(p[0])
        out['electron_retarding_fit_R2']=_r2(np.log(electron_current[m]),pred)
        out['ion_current_baseline_subtracted_A']=ion_baseline
        if slope>0:
            out['electron_temperature_eV']=1.0/slope
            out['electron_retarding_log_slope_per_V']=slope
            out['electron_retarding_fit_intercept']=float(p[1])
        else: out['electron_temperature_eV']=None
    if area_m2 is not None and 'ion_saturation_current_A' in out:
        te_value = out.get('electron_temperature_eV')
        if te_value is None or not np.isfinite(te_value) or te_value <= 0:
            raise ValueError("Ion-density estimation requires a valid positive electron temperature from the electron-retarding fit; refusing to substitute Te=0.")
        te=float(te_value)
        cs=math.sqrt(E_CHARGE*(ti_eV*gamma_i + te)/(ion_mass_amu*AMU))
        denom=sheath_factor*E_CHARGE*area_m2*cs
        out['ion_sound_speed_m_s']=cs
        out['ion_density_estimate_m-3']=abs(out['ion_saturation_current_A'])/denom if denom>0 else None
        out['ion_density_model']='n_i≈|I_isat|/(sheath_factor·e·A·c_s); depends on sheath model, geometry and ion temperature.'
    out['interpretation_warning']='Single-probe estimates depend on sweep direction, magnetic field, collisions, secondary emission, sheath expansion, surface condition and fit-window choice. Check fit R² and device-specific probe theory.'
    return out

def floating_potential(voltage: Any, current: Any) -> float | None:
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel(); m=np.isfinite(v)&np.isfinite(i); v,i=v[m],i[m]
    if len(v)<2: return None
    o=np.argsort(v); v,i=v[o],i[o]; ix=np.where(i[:-1]*i[1:]<=0)[0]
    if not len(ix): return None
    k=ix[0]
    if i[k+1]==i[k]: return float((v[k]+v[k+1])/2)
    return float(v[k]-i[k]*(v[k+1]-v[k])/(i[k+1]-i[k]))


def mode_number_from_phase(theta_rad: Any, phase_rad: Any, weights: Any | None = None,
                           max_mode_number: int | None = None) -> dict[str, float | str | int]:
    """Fit spatial phase after sorting sensors by angle; report the sampling alias limit.

    Phase-only data cannot distinguish aliased modes (for example n=7 and n=1
    on six equally spaced sensors). Supply a physically justified max_mode_number
    to reject unsupported modes; a mode at the spatial Nyquist limit is ambiguous.
    Optional weights should be coherence or inverse-variance weights.
    """
    th=np.asarray(theta_rad,float).ravel(); ph=np.asarray(phase_rad,float).ravel()
    if len(th)!=len(ph) or len(th)<3 or not np.all(np.isfinite(th)) or not np.all(np.isfinite(ph)):
        raise ValueError('Provide at least 3 paired finite sensor angles and phases.')
    th=np.mod(th,2*np.pi)
    order=np.argsort(th, kind='stable'); th,ph=th[order],ph[order]
    if np.unique(np.round(th,12)).size < 3:
        raise ValueError('At least 3 distinct sensor angles modulo 2π are required.')
    gaps=np.diff(np.r_[th, th[0]+2*np.pi])
    if np.any(gaps <= 1e-10):
        raise ValueError('Sensor angles must be unique modulo 2π.')
    alias_limit=int(np.floor(np.pi/np.max(gaps)))
    if alias_limit < 1:
        raise ValueError('Sensor geometry is too sparse for an unambiguous nonzero mode estimate.')
    if max_mode_number is not None and (isinstance(max_mode_number,bool) or int(max_mode_number)!=max_mode_number or max_mode_number < 0):
        raise ValueError('max_mode_number must be a nonnegative integer when supplied.')
    limit=min(alias_limit,int(max_mode_number)) if max_mode_number is not None else alias_limit
    # Unwrap only after spatial sorting; fit against angles shifted to start at zero.
    ph=np.unwrap(ph)
    th0=th-th[0]
    X=np.column_stack([th0,np.ones_like(th0)])
    if weights is None:
        w=np.ones(len(th))
    else:
        w0=np.asarray(weights,float).ravel()
        if len(w0)!=len(th) or not np.all(np.isfinite(w0)) or np.any(w0<0) or not np.any(w0>0):
            raise ValueError('Weights must be finite, nonnegative, match sensor count, and include a positive value.')
        w=w0[order]
    coef=np.linalg.lstsq(X*np.sqrt(w[:,None]),ph*np.sqrt(w),rcond=None)[0]
    pred=X@coef
    n=float(coef[0])
    if abs(n) >= limit-1e-8 and limit > 0:
        raise ValueError(f'Estimated |n|={abs(n):.4g} is at/above the unambiguous spatial limit {limit}; mode direction/aliasing is ambiguous. Add sensors or supply a justified lower max_mode_number.')
    residual=ph-pred
    return {'mode_number_fit':n,'nearest_integer_mode_number':float(np.rint(n)),'phase_offset_rad':float(coef[1]),
            'phase_fit_R2':_r2(ph,pred),'phase_rms_residual_rad':float(np.sqrt(np.average(residual**2,weights=w))),
            'spatial_alias_limit_mode_number':limit,
            'aliasing_note':'Modes above the spatial Nyquist limit alias to lower modes; phase-only fitting cannot identify the true aliased mode without additional geometry or a prior bound.'}


def mode_number_from_channels(theta_rad: Any, signals: Any, fs: float,
                              frequency_range_hz: tuple[float, float] | None = None,
                              min_coherence: float = 0.2) -> dict[str, Any]:
    """Extract relative cross-spectral phases from synchronized sensor traces and fit a mode number.

    The first sensor is the phase reference. The dominant shared spectral peak is selected
    unless a frequency band is supplied; coherence with the reference weights the phase fit.
    Sensor geometry must be a single toroidal/poloidal ring with a consistent angle convention.
    """
    theta = np.asarray(theta_rad, dtype=float).ravel()
    data = np.asarray(signals, dtype=float)
    if data.ndim != 2 or data.shape[0] != len(theta) or data.shape[0] < 3 or data.shape[1] < 64:
        raise ValueError("signals must have shape (at least 3 sensors, at least 64 time samples) matching theta.")
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(data)) or not np.isfinite(fs) or fs <= 0:
        raise ValueError("Sensor angles and synchronized signals must be finite; fs must be positive.")
    if not np.isfinite(min_coherence) or not 0 <= min_coherence <= 1:
        raise ValueError("min_coherence must be in [0, 1].")
    if frequency_range_hz is not None:
        if len(frequency_range_hz) != 2 or not np.all(np.isfinite(frequency_range_hz)) or frequency_range_hz[0] < 0 or frequency_range_hz[0] >= frequency_range_hz[1] or frequency_range_hz[1] > fs/2:
            raise ValueError("frequency_range_hz must be a finite increasing interval within [0, Nyquist].")
    data = data - np.mean(data, axis=1, keepdims=True)
    nperseg = min(data.shape[1], max(64, min(4096, data.shape[1]//2)))
    f, p_ref = signal.welch(data[0], fs=fs, nperseg=nperseg, noverlap=nperseg//2)
    power = p_ref.copy()
    for row in data[1:]:
        _, p = signal.welch(row, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        power += p
    allowed = f > 0
    if frequency_range_hz is not None:
        allowed &= (f >= frequency_range_hz[0]) & (f <= frequency_range_hz[1])
    if not np.any(allowed):
        raise ValueError("No positive-frequency bins fall inside the selected frequency band.")
    candidates = np.flatnonzero(allowed)
    peak_index = int(candidates[np.argmax(power[candidates])])
    peak_frequency = float(f[peak_index])
    phases = [0.0]
    coherences = [1.0]
    for row in data[1:]:
        fc, pxy = signal.csd(data[0], row, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        _, coh = signal.coherence(data[0], row, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        idx = int(np.argmin(np.abs(fc - peak_frequency)))
        phases.append(float(np.angle(pxy[idx])))
        coherences.append(float(coh[idx]))
    weights = np.asarray(coherences)
    if np.count_nonzero(weights >= min_coherence) < 3:
        raise ValueError("Fewer than three sensors have adequate coherence with the reference at the selected frequency.")
    weights = np.where(weights >= min_coherence, weights, 0.0)
    fit = mode_number_from_phase(theta, phases, weights=weights)
    fit.update({
        "selected_frequency_Hz": peak_frequency,
        "sensor_phase_rad_relative_to_first": [float(v) for v in phases],
        "coherence_with_reference": [float(v) for v in coherences],
        "minimum_coherence_threshold": float(min_coherence),
        "phase_extraction_method": "cross-spectral phase relative to first sensor; coherence-weighted spatial fit",
    })
    return fit


def fluctuation_flux(density: Any, radial_velocity: Any, temperature_eV: Any | None = None) -> dict[str, float]:
    """Compute covariance fluxes from aligned physical-unit time series."""
    n=np.asarray(density,float).ravel(); vr=np.asarray(radial_velocity,float).ravel()
    if len(n)!=len(vr) or len(n)<3: raise ValueError('Density and radial velocity must be aligned and have at least 3 samples.')
    te=None
    if temperature_eV is not None:
        te=np.asarray(temperature_eV,float).ravel()
        if len(te)!=len(n): raise ValueError('Temperature series must have same original length as density.')
    good=np.isfinite(n)&np.isfinite(vr)
    if te is not None: good &= np.isfinite(te)
    n,vr=n[good],vr[good]
    if te is not None: te=te[good]
    if len(n)<3: raise ValueError('Not enough finite aligned samples after applying the common validity mask.')
    nf=n-np.mean(n); vf=vr-np.mean(vr)
    out={'particle_flux_covariance_m-2_s-1':float(np.mean(nf*vf)),'mean_density_m-3':float(np.mean(n)),'mean_radial_velocity_m_s':float(np.mean(vr))}
    if te is not None:
        tf=te-np.mean(te)
        out['density_temperature_velocity_covariance_eV_m-2_s-1']=float(np.mean(nf*tf*vf))
        out['electron_heat_flux_proxy_W_m-2']=float(1.5*E_CHARGE*np.mean(n*te*vf))
        out['heat_flux_note']='Proxy using 3/2 n e T_e v_r; use a validated transport convention and include conductive/convective terms as appropriate.'
    out['note']='Covariance estimates require synchronized measurements, calibrated units and justified radial-velocity reconstruction.'
    return out


def exponential_heat_flux_fit(distance_m: Any, heat_flux_W_m2: Any) -> dict[str, float]:
    """Fit q(s)=q0 exp(-(s-s0)/lambda_q)+background using user-selected target interval."""
    x=np.asarray(distance_m,float).ravel(); y=np.asarray(heat_flux_W_m2,float).ravel(); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<6 or len(np.unique(x))<6: raise ValueError('At least 6 distinct valid target-coordinate points are required.')
    o=np.argsort(x); x,y=x[o],y[o]
    span=float(np.ptp(x)); base0=float(np.percentile(y,10)); q0=max(float(np.max(y)-base0),1e-12)
    def fn(z,q,l,b): return q*np.exp(-(z-x[0])/l)+b
    p,_=optimize.curve_fit(fn,x,y,p0=[q0,max(span/5,1e-9),base0],bounds=([0,span/10000,-np.inf],[np.inf,span*100,np.inf]),maxfev=30000)
    pred=fn(x,*p)
    return {'q0_W_m-2':float(p[0]),'lambda_q_m':float(p[1]),'background_W_m-2':float(p[2]),'r_squared':_r2(y,pred),'n_points':int(len(x)),'note':'Fit quality and mapping from camera pixels to target coordinate must be checked; this is not a heat-transport model.'}


def eich_heat_flux_fit(distance_m: Any, heat_flux_W_m2: Any) -> dict[str, Any]:
    """Fit the standard Eich-like divertor heat-flux profile including Gaussian spreading.

    q(s) = q0/2 * exp(-((s-s0)/S)^2) * erfcx(S/(2*lambda_q) - (s-s0)/S) + q_bg.
    The fit is descriptive and assumes the supplied coordinate is already mapped to target
    distance; it does not infer equilibrium mapping or account for diagnostic response.
    """
    x=np.asarray(distance_m,float).ravel(); y=np.asarray(heat_flux_W_m2,float).ravel()
    valid=np.isfinite(x)&np.isfinite(y); x,y=x[valid],y[valid]
    if len(x)<10 or len(np.unique(x))<10:
        raise ValueError("Eich fit requires at least 10 distinct finite target-coordinate points.")
    order=np.argsort(x); x,y=x[order],y[order]
    span=float(np.ptp(x))
    if span<=0: raise ValueError("Target coordinate must span a nonzero interval.")
    baseline=float(np.percentile(y,10)); peak=float(np.max(y))
    if peak<=baseline: raise ValueError("Heat-flux profile must contain a positive signal above its baseline.")
    def model(s,q0,lam,S,s0,bg):
        u=(s-s0)/S
        arg=S/(2*lam)-u
        # erfcx form avoids overflow in exp(a^2)*erfc(a) for the far SOL tail.
        return 0.5*q0*np.exp(-u*u)*erfcx(arg)+bg
    p0=[max(peak-baseline,1e-12),max(span/30,span/10000),max(span/50,span/10000),float(x[np.argmax(y)]),baseline]
    lower=[0.0,span/10000,span/10000,float(x[0]-span),-np.inf]
    upper=[np.inf,span*100,span*10,float(x[-1]+span),np.inf]
    p,cov=optimize.curve_fit(model,x,y,p0=p0,bounds=(lower,upper),maxfev=100000)
    pred=model(x,*p); se=np.sqrt(np.maximum(np.diag(cov),0.0))
    return {
        "q0_W_m-2":float(p[0]),"lambda_q_m":float(p[1]),"spreading_S_m":float(p[2]),
        "strike_point_s0_m":float(p[3]),"background_W_m-2":float(p[4]),
        "standard_error_q0_W_m-2":float(se[0]),"standard_error_lambda_q_m":float(se[1]),
        "standard_error_spreading_S_m":float(se[2]),"standard_error_strike_point_s0_m":float(se[3]),
        "standard_error_background_W_m-2":float(se[4]),"r_squared":_r2(y,pred),"n_points":int(len(x)),
        "model":"Eich-like exponential-convolved-with-Gaussian profile",
        "note":"Parameter identifiability depends on target coverage, noise, spatial mapping and instrument response; inspect covariance and fit residuals before physical interpretation."
    }


def integrate_profile(coordinate: Any, profile: Any) -> dict[str, float]:
    x=np.asarray(coordinate,float).ravel(); y=np.asarray(profile,float).ravel(); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<2: raise ValueError('At least two finite coordinate-profile points are required.')
    o=np.argsort(x); x,y=x[o],y[o]
    if np.any(np.diff(x)<=0): raise ValueError('Coordinate values must be unique after sorting.')
    return {'integral_trapezoid':float(integrate.trapezoid(y,x)),'coordinate_min':float(x[0]),'coordinate_max':float(x[-1]),'profile_max':float(np.max(y)),'profile_min':float(np.min(y))}


def spectrum_moments(energy_eV: Any, counts: Any, background: Any | None = None) -> dict[str, float]:
    """Calculate moments only for nonnegative background-corrected spectral weights.

    Negative bins are not silently clipped: use a justified background/noise model
    or an explicit nonnegative reconstruction before interpreting moments.
    """
    e=np.asarray(energy_eV,float).ravel(); c=np.asarray(counts,float).ravel()
    if len(e)!=len(c) or len(e)<3: raise ValueError('Energy and counts must have same length and at least 3 samples.')
    if not np.all(np.isfinite(e)):
        raise ValueError('Energy coordinates must be finite.')
    if background is not None:
        b=np.asarray(background,float).ravel()
        if len(b)!=len(c) or not np.all(np.isfinite(b)): raise ValueError('Background must be finite and match counts length.')
        c=c-b
    m=np.isfinite(e)&np.isfinite(c); e,c=e[m],c[m]
    o=np.argsort(e); e,c=e[o],c[o]
    if np.any(np.diff(e) <= 0):
        raise ValueError('Energy coordinates must be unique after sorting.')
    if np.any(c < 0):
        raise ValueError('Background-subtracted spectrum contains negative bins; moments are not computed by clipping noise. Improve the background/noise model or explicitly select a justified nonnegative signal.')
    area=float(integrate.trapezoid(c,e))
    if area<=0: raise ValueError('Integrated background-subtracted spectrum must be positive.')
    mean=float(integrate.trapezoid(e*c,e)/area); var=float(integrate.trapezoid((e-mean)**2*c,e)/area)
    return {'integrated_counts_eV':area,'mean_energy_eV':mean,'rms_energy_spread_eV':math.sqrt(max(var,0)),'peak_energy_eV':float(e[np.argmax(c)]),'n_points':int(len(e)),'note':'Moments are descriptive; detector response, efficiency, pile-up and background model are not corrected automatically.'}


def neutron_rate_from_counts(counts_per_s: float, efficiency: float, geometric_factor: float = 1.0, calibration_factor: float = 1.0) -> dict[str, float]:
    vals=[counts_per_s, efficiency, geometric_factor, calibration_factor]
    if not np.all(np.isfinite(vals)) or counts_per_s<0 or not 0<efficiency<=1 or geometric_factor<=0 or calibration_factor<=0:
        raise ValueError('Counts must be finite and nonnegative; efficiency in (0,1], geometric and calibration factors finite and positive.')
    rate=float(counts_per_s/(efficiency*geometric_factor)*calibration_factor)
    return {'measured_count_rate_s-1':float(counts_per_s),'estimated_source_rate_s-1':rate,'efficiency':float(efficiency),'geometric_factor':float(geometric_factor),'calibration_factor':float(calibration_factor),'note':'Only valid when these factors represent the detector response and geometry for the specified neutron energy and configuration.'}


def equilibrium_grid_derivatives(R_m: Any, Z_m: Any, psi: Any, Bphi_T: float) -> dict[str, np.ndarray]:
    """Compute derivatives and Bp from a supplied poloidal-flux grid; does not solve equilibrium."""
    R=np.asarray(R_m,float).ravel(); Z=np.asarray(Z_m,float).ravel(); p=np.asarray(psi,float)
    if p.shape!=(len(Z),len(R)) or len(R)<3 or len(Z)<3: raise ValueError('psi must have shape (len(Z), len(R)) with at least 3 points per axis.')
    if np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0) or np.any(R<=0): raise ValueError('R and Z must be strictly increasing and R positive.')
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)) or not np.all(np.isfinite(p)):
        raise ValueError('R, Z and psi grid must contain only finite values.')
    if not np.isfinite(Bphi_T):
        raise ValueError('Toroidal magnetic field must be finite.')
    dpsi_dZ,dpsi_dR=np.gradient(p,Z,R,edge_order=2)
    RR=np.broadcast_to(R[None,:],p.shape)
    BR=-dpsi_dZ/RR; BZ=dpsi_dR/RR; BP=np.sqrt(BR**2+BZ**2); BT=np.full_like(BP,float(Bphi_T))
    return {'dpsi_dR':dpsi_dR,'dpsi_dZ':dpsi_dZ,'BR_T':BR,'BZ_T':BZ,'Bp_T':BP,'Bphi_T':BT,'Btotal_T':np.sqrt(BP**2+BT**2),'note':'Uses the supplied psi convention; verify whether psi is poloidal flux or flux divided by 2π and match units/sign conventions.'}


def xray_spectrum_summary(energy_keV: Any, counts: Any, background: Any | None = None) -> dict[str, float]:
    r=spectrum_moments(energy_keV,counts,background)
    r['peak_energy_keV']=r.pop('peak_energy_eV')/1.0
    r['mean_energy_keV']=r.pop('mean_energy_eV')
    r['rms_energy_spread_keV']=r.pop('rms_energy_spread_eV')
    r['integrated_counts_keV']=r.pop('integrated_counts_eV')
    r['note']='Energy axis is interpreted as keV. Detector response and absolute calibration are not applied.'
    return r


def abel_reconstruction(chord_impact_m: Any, line_integrals: Any, radial_edges_m: Any,
                        lam: float = 1e-2, nonnegative: bool = True) -> dict[str, Any]:
    """Axisymmetric Abel inversion via a shell path-length matrix and Tikhonov regularization."""
    b=np.asarray(chord_impact_m,float).ravel(); y=np.asarray(line_integrals,float).ravel(); edges=np.asarray(radial_edges_m,float).ravel()
    if len(b)!=len(y) or len(b)<3: raise ValueError('At least 3 paired chord impact parameters and line integrals are required.')
    if len(edges)<3 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges)<=0) or edges[0]<0:
        raise ValueError('Radial edges must be finite, increasing, nonnegative and contain at least 3 values.')
    if not np.all(np.isfinite(b)) or not np.all(np.isfinite(y)) or np.any(b<0): raise ValueError('Chord coordinates and line integrals must be finite; impact parameters nonnegative.')
    rin=edges[:-1][None,:]; rout=edges[1:][None,:]; bb=b[:,None]
    A=2*(np.sqrt(np.maximum(rout*rout-bb*bb,0))-np.sqrt(np.maximum(rin*rin-bb*bb,0)))
    A=np.where(bb>=rout,0,A)
    result=regularized_tomography(A,y,lam,nonnegative)
    centers=.5*(edges[:-1]+edges[1:])
    return {'radius_centers_m':centers,'emissivity_or_density_profile':result['solution'],'response_matrix':A,'residual_norm':result['residual_norm'],'success':result['success'],'message':result['message'],'lambda':float(lam),'nonnegative':bool(nonnegative),'note':'Axisymmetry, straight chord geometry and the units of the supplied line integrals are assumed; validate edge definition and diagnostic response.'}


def axisymmetric_radiated_power(R_m: Any, Z_m: Any, emissivity_W_m3: Any) -> dict[str, float]:
    """Approximate total radiated power from a poloidal emissivity grid assuming toroidal symmetry."""
    R=np.asarray(R_m,float).ravel(); Z=np.asarray(Z_m,float).ravel(); eps=np.asarray(emissivity_W_m3,float)
    if eps.shape!=(len(Z),len(R)) or len(R)<2 or len(Z)<2: raise ValueError('Emissivity grid shape must be (len(Z),len(R)).')
    if np.any(np.diff(R)<=0) or np.any(np.diff(Z)<=0) or np.any(R<=0): raise ValueError('R and Z must be increasing and R positive.')
    if not np.all(np.isfinite(R)) or not np.all(np.isfinite(Z)):
        raise ValueError('R and Z coordinates must contain only finite values.')
    if not np.all(np.isfinite(eps)): raise ValueError('Emissivity grid must be finite.')
    if np.any(eps < 0):
        raise ValueError('Emissivity must be nonnegative for a physical radiated-power integral; inspect background subtraction or inversion artifacts first.')
    # Integrate toroidal volume element 2*pi*R dR dZ over the provided rectangular grid.
    radial=integrate.trapezoid(2*np.pi*R[None,:]*eps,R,axis=1)
    power=float(integrate.trapezoid(radial,Z))
    return {'total_radiated_power_W':power,'emissivity_max_W_m-3':float(np.max(eps)),'emissivity_min_W_m-3':float(np.min(eps)),'assumption':'Axisymmetric emissivity and supplied grid covers the intended integration domain.'}


def profile_gradient_analysis(coordinate: Any, profile: Any) -> dict[str, Any]:
    """Calculate profile gradients and basic pedestal/peak descriptors from calibrated profile data."""
    x=np.asarray(coordinate,float).ravel(); y=np.asarray(profile,float).ravel(); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<5: raise ValueError('At least 5 finite profile points are required.')
    o=np.argsort(x); x,y=x[o],y[o]
    if np.any(np.diff(x)<=0): raise ValueError('Profile coordinate must be unique and increasing.')
    grad=np.gradient(y,x,edge_order=2); k=int(np.argmax(np.abs(grad)))
    result={'peak_value':float(np.max(y)),'minimum_value':float(np.min(y)),'peak_coordinate':float(x[np.argmax(y)]),'profile_integral':float(integrate.trapezoid(y,x)),'max_abs_gradient':float(np.max(np.abs(grad))),'max_gradient_coordinate':float(x[k]),'gradient_at_max_abs':float(grad[k]),'n_points':int(len(x))}
    # Half-maximum width is meaningful only for nonnegative profiles with a positive peak.
    if np.max(y)>0 and np.min(y)>=0:
        inds=np.where(y>=.5*np.max(y))[0]
        if len(inds): result['FWHM_coordinate_units']=float(x[inds[-1]]-x[inds[0]])
    return result


def triple_probe_temperature_eV(v_difference_V: float, model_factor: float = 1.0) -> dict[str, float | str]:
    """First-order triple-probe Te estimate under a user-selected idealized exponential-current model."""
    if not np.isfinite(v_difference_V) or v_difference_V<=0 or not np.isfinite(model_factor) or model_factor<=0:
        raise ValueError('Voltage difference and model factor must be positive and finite.')
    te=float(v_difference_V/(np.log(2)*model_factor))
    return {'electron_temperature_estimate_eV':te,'voltage_difference_V':float(v_difference_V),'model_factor':float(model_factor),'assumption':'Idealized triple-probe relation; confirm the actual electrode configuration, current balance, sheath model and circuit definition before use.'}


def reflectometry_cutoff_density(frequency_hz: float, mode: str = 'O') -> dict[str, float | str]:
    """Cold-plasma cutoff density estimate for O/X mode at user-supplied probing frequency."""
    if not np.isfinite(frequency_hz) or frequency_hz<=0: raise ValueError('Frequency must be positive and finite.')
    mode=mode.upper()
    if mode not in {'O','X'}: raise ValueError("Mode must be 'O' or 'X'.")
    # O-mode: omega_pe = omega. X-mode requires B and has multiple cutoffs, so do not fake it.
    if mode=='X': raise ValueError('X-mode cutoff depends on magnetic field and branch; use the O-mode formula only or supply a validated X-mode model.')
    n=float(EPS0*M_E*(2*np.pi*frequency_hz)**2/E_CHARGE**2)
    return {'frequency_Hz':float(frequency_hz),'mode':'O','cutoff_density_m-3':n,'formula':'n_cut = ε0 m_e (2π f)^2 / e²','assumption':'Cold, collisionless O-mode cutoff; actual reflectometry inversion also requires propagation and cutoff-layer interpretation.'}


def electron_cyclotron_resonance_field(frequency_hz: float, harmonic: int = 1) -> dict[str, float | int | str]:
    if not np.isfinite(frequency_hz) or frequency_hz<=0 or not np.isfinite(harmonic) or float(harmonic) != int(harmonic) or int(harmonic)<1:
        raise ValueError('Frequency must be positive and harmonic a positive integer.')
    B=float(2*np.pi*frequency_hz*M_E/(E_CHARGE*int(harmonic)))
    return {'frequency_Hz':float(frequency_hz),'harmonic':int(harmonic),'resonance_field_T':B,'formula':'ω = h e B / m_e','assumption':'Relativistic and Doppler shifts are neglected; resonance geometry must be considered for actual ECE interpretation.'}


def corrected_line_ratio(intensity_1: float, intensity_2: float, response_1: float = 1.0, response_2: float = 1.0) -> dict[str, float | str]:
    vals=[intensity_1,intensity_2,response_1,response_2]
    if not all(np.isfinite(v) for v in vals) or intensity_1<0 or intensity_2<=0 or response_1<=0 or response_2<=0: raise ValueError('Intensities must be nonnegative (line 2 positive) and instrument responses positive and finite.')
    ratio=float((intensity_1/response_1)/(intensity_2/response_2))
    return {'raw_intensity_ratio':float(intensity_1/intensity_2),'instrument_response_corrected_ratio':ratio,'response_1':float(response_1),'response_2':float(response_2),'note':'A physical Te/ne/impurity inference requires validated atomic emissivity coefficients or a collisional-radiative model; this routine only corrects the measured ratio for supplied responses.'}

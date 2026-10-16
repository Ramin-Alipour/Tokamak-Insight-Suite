"""Signal-processing algorithms extracted from the legacy numerical core."""
from __future__ import annotations
from typing import Any
import numpy as np
from scipy import signal

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
    # Remove only the mean (constant detrend); preserve linear trends unless the
    # caller explicitly preprocesses them. This matches the documented FFT option.
    if detrend:
        x = signal.detrend(x, type="constant")
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
    """Return coherence and cross phase; phase convention is φ_y − φ_x (SciPy CSD)."""
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
    overlap=n//2
    f, coh = signal.coherence(a,b,fs=fs,nperseg=n,noverlap=overlap,window='hann')
    fc, pxy = signal.csd(a,b,fs=fs,nperseg=n,noverlap=overlap,window='hann')
    step=max(1,n-overlap)
    segments=1+max(0,(len(a)-n)//step)
    # Approximate 95% null threshold and phase standard error from nominal Welch
    # segment count. Overlap reduces the effective independent averages, so these
    # are diagnostic estimates, not exact confidence intervals.
    coh_threshold=float(1.0-0.05**(1.0/max(1,segments-1)))
    phase_se=np.full_like(coh,np.nan,dtype=float)
    positive=coh>np.finfo(float).eps
    phase_se[positive]=np.sqrt(np.maximum(1.0-coh[positive],0.0)/(2.0*coh[positive]*max(segments,1)))
    return {'frequency_Hz':f, 'coherence':coh, 'cross_phase_rad':np.angle(pxy),
            'cross_spectrum_real':pxy.real, 'cross_spectrum_imag':pxy.imag,
            'coherence_95pct_null_threshold':np.full_like(coh,coh_threshold,dtype=float),
            'phase_standard_error_approx_rad':phase_se,
            'nominal_welch_segment_count':np.full(len(f),segments,dtype=int)}


def cross_correlation_lag(x: Any, y: Any, fs: float, max_lag_s: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Return lag at maximum absolute correlation; y delayed relative to x gives a negative lag."""
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

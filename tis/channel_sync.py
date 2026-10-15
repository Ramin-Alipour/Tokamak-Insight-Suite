from __future__ import annotations
import numpy as np
from .core import cross_correlation_lag


def analyze_channel_synchronization(time_s, channel_signals, *, reference_channel: int = 0, max_lag_s: float | None = None) -> dict:
    """Estimate relative channel delays by cross-correlation against a reference channel."""
    t = np.asarray(time_s, dtype=float).ravel()
    data = np.asarray(channel_signals, dtype=float)
    if data.ndim != 2 or data.shape[0] != t.size or t.size < 8 or data.shape[1] < 2:
        raise ValueError('channel_signals must be time-by-channel with at least eight samples and two channels.')
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(data)) or np.any(np.diff(t) <= 0):
        raise ValueError('Time and signals must be finite and time strictly increasing.')
    if isinstance(reference_channel, bool) or not isinstance(reference_channel, int) or not 0 <= reference_channel < data.shape[1]:
        raise ValueError('reference_channel index is out of range.')
    dt = np.diff(t)
    dt_med = float(np.median(dt))
    if not np.allclose(dt, dt_med, rtol=1e-3, atol=max(1e-12, dt_med*1e-6)):
        raise ValueError('Time base is not sufficiently uniform; resample channels before lag estimation.')
    fs = 1/dt_med
    ref = data[:, reference_channel]
    lags, correlations = [], []
    for idx in range(data.shape[1]):
        if idx == reference_channel:
            lags.append(0.0); correlations.append(1.0); continue
        lag_axis, corr, lag = cross_correlation_lag(ref, data[:, idx], fs=fs, max_lag_s=max_lag_s)
        lags.append(float(lag))
        correlations.append(float(corr[int(np.argmin(np.abs(lag_axis-lag)))]))
    return {'diagnostic': 'Channel synchronization', 'time_s': t, 'sampling_frequency_Hz': fs,
        'reference_channel': reference_channel, 'relative_delay_s': np.asarray(lags),
        'correlation_at_selected_lag': np.asarray(correlations),
        'sample_interval_jitter_s_rms': float(np.std(dt-dt_med, ddof=0)),
        'assumptions_and_limits': ['Relative delay is estimated from signal correlation and may be ambiguous for periodic or weakly correlated channels.',
            'Common timing trigger and calibration pulses are preferred for absolute synchronization.',
            'Sign convention follows cross_correlation_lag: positive/negative lag must be interpreted with that function’s convention.']}

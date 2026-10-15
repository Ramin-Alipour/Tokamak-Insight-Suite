from __future__ import annotations
from typing import Any
import numpy as np
from .core import prepare_time_series, signal_quality, welch_psd, butterworth_filter


def fast_bolometry_timeseries(time_s: Any, channel_signal: Any, *, lowpass_hz: float | None = None, psd_segment_samples: int | None = None) -> dict[str, Any]:
    """Quality-check a fast-bolometer trace and summarize its fluctuation spectrum."""
    t, y, fs = prepare_time_series(time_s, channel_signal)
    if lowpass_hz is not None:
        if not np.isfinite(lowpass_hz) or lowpass_hz <= 0 or lowpass_hz >= fs / 2:
            raise ValueError('lowpass_hz must be positive and below Nyquist.')
        processed = butterworth_filter(y, fs, lowpass=float(lowpass_hz))
    else:
        processed = y.copy()
    freq, psd = welch_psd(processed, fs, nperseg=psd_segment_samples)
    peak = float(freq[1 + np.argmax(psd[1:])]) if len(psd) > 1 else None
    return {'time_s': t, 'signal_input_units': y, 'signal_processed': processed,
            'sample_rate_Hz': float(fs), 'nyquist_frequency_Hz': float(fs / 2),
            'quality': signal_quality(y), 'psd_frequency_Hz': freq,
            'psd_input_units_squared_per_Hz': psd,
            'dominant_nonzero_psd_frequency_Hz': peak, 'lowpass_hz': lowpass_hz,
            'n_samples': int(len(y)),
            'interpretation': 'Time-series analysis only; absolute radiated power requires detector response, calibration and geometry.'}

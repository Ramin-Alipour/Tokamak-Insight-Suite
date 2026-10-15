from __future__ import annotations
from typing import Any
import numpy as np
from .core import prepare_time_series, cross_correlation_lag


def estimate_channel_time_offset(time_s: Any, reference_signal: Any, comparison_signal: Any, *, max_lag_s: float | None = None) -> dict[str, Any]:
    """Estimate relative offset between already synchronized, equally sampled channels.

    The core correlation convention is retained: a comparison signal delayed
    relative to the reference produces a negative lag. Add the reported lag to
    the comparison channel's time coordinate as the first-order alignment correction.
    """
    t = np.asarray(time_s, dtype=float).ravel()
    a = np.asarray(reference_signal, dtype=float).ravel()
    b = np.asarray(comparison_signal, dtype=float).ravel()
    if len(t) != len(a) or len(t) != len(b):
        raise ValueError('Time and both channel arrays must have equal lengths.')
    if len(t) < 8 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError('At least eight finite samples with strictly increasing time are required.')
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError('Channel signals must be finite; repair missing samples explicitly.')
    _, _, fs = prepare_time_series(t, a)
    _, corr, lag = cross_correlation_lag(a, b, fs, max_lag_s=max_lag_s)
    peak = float(np.max(np.abs(corr))) if len(corr) else 0.0
    return {'estimated_lag_s': float(lag), 'correction_to_comparison_time_s': float(lag),
            'sample_rate_Hz': float(fs), 'sample_period_s': float(1 / fs),
            'lag_in_samples': float(lag * fs), 'absolute_peak_correlation': peak,
            'interpretation': 'Correlation-based relative timing estimate, not a hardware trigger calibration. Periodicity, weak coupling, channel response and clock drift can make the lag ambiguous.'}

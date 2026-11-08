import numpy as np
import pytest

from tis.events import interval_statistics


def test_interval_statistics_rejects_nonchronological_event_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        interval_statistics([1.0, 0.5, 2.0])


def test_interval_statistics_rejects_nonfinite_event_times():
    with pytest.raises(ValueError, match="finite"):
        interval_statistics([1.0, np.nan])

def test_event_clean_accepts_quantized_microsecond_intervals_on_epoch_timebase():
    from tis.events import _clean

    intervals = np.tile([2e-6, 3e-6], 40)
    time_s = 1_700_000_000.0 + np.r_[0.0, np.cumsum(intervals)]
    signal = np.zeros(time_s.size)
    cleaned_time, cleaned_signal, dt = _clean(time_s, signal, min_n=16)
    assert cleaned_time.size == time_s.size
    assert cleaned_signal.size == signal.size
    assert dt == pytest.approx(np.median(np.diff(time_s)))


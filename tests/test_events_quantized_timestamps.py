import numpy as np
import pytest

from tis.events import _clean


def test_event_clean_accepts_microsecond_quantized_400khz_timestamps():
    true_time = np.arange(4096, dtype=float) / 400_000
    rounded_time = np.round(true_time * 1e6) / 1e6
    values = np.sin(2 * np.pi * 20_000 * true_time)

    time, signal, dt = _clean(rounded_time, values)

    assert len(time) == len(signal) == len(values)
    assert dt == pytest.approx(2.5e-6, rel=0.2)


def test_event_clean_still_rejects_a_dropped_sample_gap():
    time = np.arange(1000, dtype=float) * 2.5e-6
    time[501:] += 10e-6
    values = np.sin(2 * np.pi * 20_000 * time)

    with pytest.raises(ValueError, match="approximately uniform sampling"):
        _clean(time, values)

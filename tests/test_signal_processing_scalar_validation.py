"""Regression tests for robust validation of public signal-processing scalars."""
import numpy as np
import pytest

from tis.signal_processing import (
    autocorrelation_analysis,
    butterworth_filter,
    cross_analysis,
    cross_correlation_lag,
    fft_spectrum,
    time_frequency_spectrogram,
    welch_psd,
)

_BAD_SCALARS = [True, np.bool_(True), "10", np.nan, np.inf, 10**1000]


@pytest.mark.parametrize("bad", _BAD_SCALARS)
@pytest.mark.parametrize(
    "call",
    [
        lambda bad: fft_spectrum(np.arange(32.0), bad),
        lambda bad: welch_psd(np.arange(32.0), bad),
        lambda bad: autocorrelation_analysis(np.sin(np.arange(32.0)), bad),
        lambda bad: time_frequency_spectrogram(np.sin(np.arange(32.0)), bad),
        lambda bad: cross_analysis(np.sin(np.arange(32.0)), np.cos(np.arange(32.0)), bad),
        lambda bad: cross_correlation_lag(np.arange(32.0), np.arange(32.0)[::-1], bad),
        lambda bad: butterworth_filter(np.sin(np.arange(32.0)), bad, lowpass=1.0),
    ],
)
def test_signal_apis_reject_invalid_sampling_rates(call, bad):
    with pytest.raises(ValueError):
        call(bad)


@pytest.mark.parametrize("bad", _BAD_SCALARS)
def test_filter_rejects_invalid_order_without_overflow(bad):
    with pytest.raises(ValueError):
        butterworth_filter(np.sin(np.arange(64.0)), 100.0, lowpass=10.0, order=bad)


@pytest.mark.parametrize("bad", _BAD_SCALARS)
def test_welch_rejects_invalid_segment_length_without_overflow(bad):
    with pytest.raises(ValueError):
        welch_psd(np.sin(np.arange(64.0)), 100.0, nperseg=bad)


@pytest.mark.parametrize("bad", _BAD_SCALARS)
def test_spectrogram_rejects_invalid_segment_length_without_overflow(bad):
    with pytest.raises(ValueError):
        time_frequency_spectrogram(np.sin(np.arange(64.0)), 100.0, nperseg=bad)


@pytest.mark.parametrize("bad", _BAD_SCALARS)
def test_cross_analysis_rejects_invalid_segment_length_without_overflow(bad):
    x = np.sin(np.arange(64.0))
    with pytest.raises(ValueError):
        cross_analysis(x, np.cos(np.arange(64.0)), 100.0, nperseg=bad)

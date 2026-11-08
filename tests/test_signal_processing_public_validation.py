"""Regression tests for public scalar validation in signal-processing APIs."""
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


BAD_REAL_VALUES = [True, np.bool_(True), "1000", np.nan, np.inf, -np.inf, -1]


@pytest.mark.parametrize("bad_fs", BAD_REAL_VALUES)
@pytest.mark.parametrize(
    "call",
    [
        lambda x, fs: butterworth_filter(x, fs, lowpass=100.0),
        lambda x, fs: fft_spectrum(x, fs),
        lambda x, fs: welch_psd(x, fs),
        lambda x, fs: autocorrelation_analysis(x, fs),
        lambda x, fs: time_frequency_spectrogram(x, fs),
        lambda x, fs: cross_analysis(x, x, fs),
        lambda x, fs: cross_correlation_lag(x, x, fs),
    ],
)
def test_signal_apis_reject_invalid_sampling_frequency(bad_fs, call):
    x = np.sin(np.arange(256, dtype=float) * 0.1)
    with pytest.raises(ValueError, match="sampling|Sampling"):
        call(x, bad_fs)


@pytest.mark.parametrize("parameter", ["highpass", "lowpass"])
@pytest.mark.parametrize("bad_value", [True, np.bool_(True), "10", np.nan, np.inf, -np.inf])
def test_butterworth_rejects_non_real_cutoff_values(parameter, bad_value):
    x = np.sin(np.arange(256, dtype=float) * 0.1)
    kwargs = {parameter: bad_value}
    with pytest.raises(ValueError, match="cutoff"):
        butterworth_filter(x, 1000.0, **kwargs)


@pytest.mark.parametrize("bad_lag", [True, np.bool_(True), "0.1", np.nan, np.inf, -np.inf, -1.0])
@pytest.mark.parametrize("call", [autocorrelation_analysis, cross_correlation_lag])
def test_correlation_apis_reject_invalid_max_lag(bad_lag, call):
    x = np.sin(np.arange(256, dtype=float) * 0.1)
    with pytest.raises(ValueError, match="Maximum lag"):
        if call is autocorrelation_analysis:
            call(x, 1000.0, max_lag_s=bad_lag)
        else:
            call(x, x, 1000.0, max_lag_s=bad_lag)

"""Regression tests for malformed numeric parameters in signal processing."""
import numpy as np
import pytest

from tis.signal_processing import (
    butterworth_filter,
    time_frequency_spectrogram,
    welch_psd,
)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 3.5])
def test_filter_order_invalid_values_raise_value_error(value):
    with pytest.raises(ValueError, match="Filter order"):
        butterworth_filter(np.arange(64.0), fs=100.0, lowpass=20.0, order=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 7.5, "8"])
def test_welch_nperseg_invalid_values_raise_value_error(value):
    with pytest.raises(ValueError, match="nperseg"):
        welch_psd(np.arange(128.0), fs=1000.0, nperseg=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 2.5, "8"])
def test_welch_noverlap_invalid_values_raise_value_error(value):
    with pytest.raises(ValueError, match="noverlap"):
        welch_psd(np.arange(128.0), fs=1000.0, nperseg=32, noverlap=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 7.5, "8"])
def test_spectrogram_nperseg_invalid_values_raise_value_error(value):
    with pytest.raises(ValueError, match="nperseg"):
        time_frequency_spectrogram(np.arange(128.0), fs=1000.0, nperseg=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 2.5, "8"])
def test_spectrogram_noverlap_invalid_values_raise_value_error(value):
    with pytest.raises(ValueError, match="noverlap"):
        time_frequency_spectrogram(
            np.arange(128.0), fs=1000.0, nperseg=32, noverlap=value
        )

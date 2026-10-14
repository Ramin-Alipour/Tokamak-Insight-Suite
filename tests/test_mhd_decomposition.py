"""Regression tests for publication-relevant MHD decompositions."""
import numpy as np
import pytest

from tis.core import morlet_wavelet_analysis, svd_mode_decomposition


def test_svd_recovers_rank_one_multichannel_structure():
    t = np.linspace(0, 1, 512, endpoint=False)
    common = np.sin(2 * np.pi * 40 * t)
    matrix = np.column_stack([common, 2 * common, -0.5 * common])
    result = svd_mode_decomposition(matrix, center=True)
    assert result["energy_fraction"][0] == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(result["reconstruction"], matrix, atol=1e-12)
    assert result["principal_components"].shape == (512, 3)
    assert result["channel_modes"].shape == (3, 3)


def test_svd_standardization_is_explicit_and_rejects_constant_channels():
    t = np.linspace(0, 1, 128, endpoint=False)
    matrix = np.column_stack([np.sin(2 * np.pi * 10 * t), 3 * np.cos(2 * np.pi * 10 * t)])
    result = svd_mode_decomposition(matrix, standardize=True)
    assert result["standardized"] is True
    with pytest.raises(ValueError, match="constant"):
        svd_mode_decomposition(np.column_stack([matrix, np.ones(len(t))]), standardize=True)


def test_svd_rejects_nonfinite_matrix():
    with pytest.raises(ValueError, match="finite"):
        svd_mode_decomposition([[1, 2], [3, np.nan]])


def test_morlet_wavelet_localizes_known_tone_frequency():
    fs = 1024.0
    t = np.arange(2048) / fs
    frequency = 80.0
    x = np.sin(2 * np.pi * frequency * t)
    frequencies = np.linspace(20.0, 200.0, 181)
    result = morlet_wavelet_analysis(t, x, frequencies)
    central = (result["time_s"] >= 0.4) & (result["time_s"] <= 1.6)
    average_power = np.mean(result["power"][:, central], axis=1)
    peak_frequency = result["frequencies_Hz"][np.argmax(average_power)]
    assert peak_frequency == pytest.approx(frequency, abs=4.0)
    assert result["power"].shape == (len(frequencies), len(t))
    assert np.all(np.isfinite(result["power"]))


def test_morlet_wavelet_rejects_nyquist_and_unsorted_frequencies():
    t = np.arange(128) / 128.0
    x = np.sin(2 * np.pi * 10 * t)
    with pytest.raises(ValueError, match="Nyquist"):
        morlet_wavelet_analysis(t, x, [10, 64])
    with pytest.raises(ValueError, match="strictly increasing"):
        morlet_wavelet_analysis(t, x, [20, 10])


def test_morlet_wavelet_rejects_frequency_support_longer_than_record():
    fs = 10000.0
    t = np.arange(256) / fs
    x = np.sin(2 * np.pi * 100 * t)
    with pytest.raises(ValueError, match="too long for this record"):
        morlet_wavelet_analysis(t, x, [100.0])

def test_ir_t1_published_mhd_frequency_band_is_resolved_by_fft_and_welch():
    """Synthetic numerical stress test at the ~44 kHz band reported for IR-T1.

    This checks sampling, frequency-axis and PSD behavior at the publication's
    frequency scale; it is not a reproduction of the paper's missing raw trace.
    """
    from tis.core import fft_spectrum, welch_psd

    fs = 200_000.0
    n = 10_000
    t = np.arange(n) / fs
    x = 0.7 * np.sin(2 * np.pi * 44_000.0 * t)

    freq_fft, amplitude = fft_spectrum(x, fs, window="hann")
    freq_psd, psd = welch_psd(x, fs, nperseg=2_000, noverlap=1_000)

    fft_peak = freq_fft[np.argmax(amplitude)]
    psd_peak = freq_psd[np.argmax(psd)]
    integrated_variance = float(np.sum(psd) * (freq_psd[1] - freq_psd[0]))

    assert fft_peak == pytest.approx(44_000.0, abs=20.0)
    assert psd_peak == pytest.approx(44_000.0, abs=100.0)
    assert integrated_variance == pytest.approx(np.var(x), rel=0.03)

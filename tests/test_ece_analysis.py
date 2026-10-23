import numpy as np
import pytest

from tokamak_analysis.ece_analysis import (
    analyze_ece, brightness_temperature_K, brightness_temperature_eV,
    resonance_field_T, resonance_radius_m,
)


def test_fundamental_resonance_field():
    assert resonance_field_T(28e9, 1) == pytest.approx(1.000, rel=0.01)


def test_harmonic_resonance_field_scales_inversely():
    assert resonance_field_T(56e9, 2) == pytest.approx(resonance_field_T(28e9, 1))


def test_resonance_mapping_inverse_circular_field():
    r = resonance_radius_m(np.array([56e9]), 1.0, 2.0, harmonic=2)
    assert r.shape == (1,)
    assert r[0] == pytest.approx(1.0, rel=0.01)


def test_brightness_temperature_zero_radiance_is_zero():
    assert brightness_temperature_K(0.0, 100e9) == 0
    assert brightness_temperature_eV(0.0, 100e9) == 0


def test_temperature_requires_calibrated_radiance():
    with pytest.raises(ValueError, match="spectral_radiance"):
        analyze_ece([80e9, 90e9], mode="brightness_temperature")


def test_channel_fluctuations_return_sampling_and_psd():
    t = np.arange(256) * 1e-6
    y = np.column_stack([np.sin(2*np.pi*10e3*t), 2*np.sin(2*np.pi*20e3*t)])
    result = analyze_ece([80e9, 90e9], signal=y, time_s=t, mode="channel_fluctuations")
    assert result["fluctuation_psd"].shape == (129, 2)
    assert result["fluctuation_rms"][1] == pytest.approx(2 / np.sqrt(2), rel=0.03)
    assert result["nyquist_frequency_Hz"] == pytest.approx(500e3)


def test_invalid_frequency_and_sampling_are_rejected():
    with pytest.raises(ValueError):
        analyze_ece([90e9, 80e9], mode="resonance_mapping")
    with pytest.raises(ValueError):
        analyze_ece([80e9, 90e9], signal=np.ones((3, 2)), time_s=[0, 1, 3], mode="channel_fluctuations")

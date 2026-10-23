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
    assert r[0] == pytest.approx(2.0, rel=0.01)


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


def test_one_sided_psd_obeys_parseval_for_even_and_odd_sample_counts():
    for n in (255, 256):
        dt = 2.5e-6
        t = np.arange(n) * dt
        y = np.column_stack([
            1.7 * np.sin(2 * np.pi * 17 / (n * dt) * t),
            0.4 * np.cos(2 * np.pi * 31 / (n * dt) * t),
        ])
        result = analyze_ece([80e9, 90e9], signal=y, time_s=t,
                             mode="channel_fluctuations")
        df = result["fluctuation_psd_frequency_Hz"][1]
        integrated = np.sum(result["fluctuation_psd"], axis=0) * df
        expected = np.mean((y - y.mean(axis=0)) ** 2, axis=0)
        np.testing.assert_allclose(integrated, expected, rtol=1e-12, atol=1e-12)


def test_noise_and_optical_depth_shape_validation():
    with pytest.raises(ValueError, match="noise_std"):
        analyze_ece([80e9, 90e9], signal=np.ones((8, 2)), time_s=np.arange(8),
                    noise_std=[1, 2, 3], mode="channel_fluctuations")
    with pytest.raises(ValueError, match="optical_depth"):
        analyze_ece([80e9, 90e9], spectral_radiance_W_m2_sr_Hz=[1e-10, 2e-10],
                    optical_depth=[1, 2, 3], mode="brightness_temperature")


def test_psd_dc_and_nyquist_are_not_doubled():
    n = 256
    dt = 1e-6
    t = np.arange(n) * dt
    # A Nyquist-frequency alternating signal has all its power in the final bin.
    y = ((-1.0) ** np.arange(n))[:, None]
    result = analyze_ece([80e9], signal=y, time_s=t,
                         mode="channel_fluctuations")
    df = result["fluctuation_psd_frequency_Hz"][1]
    assert result["fluctuation_psd"][-1, 0] * df == pytest.approx(1.0, rel=1e-12)



@pytest.mark.parametrize("harmonic", [0, -1, 1.5, True, "2"])
def test_invalid_harmonic_is_rejected_by_pipeline(harmonic):
    with pytest.raises(ValueError, match="harmonic"):
        analyze_ece([80e9, 90e9], harmonic=harmonic, mode="resonance_mapping")


@pytest.mark.parametrize("radiance", [
    [1e-10, np.nan],
    [1e-10, np.inf],
    [1e-10, -1e-10],
    1e-10,
])
def test_invalid_calibrated_radiance_is_rejected(radiance):
    with pytest.raises(ValueError, match="spectral_radiance"):
        analyze_ece([80e9, 90e9], spectral_radiance_W_m2_sr_Hz=radiance)


def test_analysis_modes_reject_conflicting_inputs():
    with pytest.raises(ValueError, match="does not consume signal"):
        analyze_ece([80e9, 90e9], spectral_radiance_W_m2_sr_Hz=[1e-10, 2e-10],
                    signal=np.ones((8, 2)), time_s=np.arange(8), mode="brightness_temperature")
    with pytest.raises(ValueError, match="does not consume spectral radiance"):
        analyze_ece([80e9, 90e9], spectral_radiance_W_m2_sr_Hz=[1e-10, 2e-10],
                    signal=np.ones((8, 2)), time_s=np.arange(8), mode="channel_fluctuations")
    with pytest.raises(ValueError, match="accepts resonance inputs only"):
        analyze_ece([80e9, 90e9], spectral_radiance_W_m2_sr_Hz=[1e-10, 2e-10],
                    mode="resonance_mapping")


def test_multitime_multichannel_radiance_shape_is_preserved():
    radiance = np.array([[1e-10, 2e-10], [2e-10, 4e-10]])
    result = analyze_ece([80e9, 90e9], spectral_radiance_W_m2_sr_Hz=radiance)
    assert result["brightness_temperature_K"].shape == (2, 2)
    assert np.all(np.isfinite(result["brightness_temperature_eV"]))

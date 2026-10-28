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


def test_resonance_mapping_does_not_fabricate_high_field_side_radius():
    with pytest.raises(ValueError, match="viewing geometry and equilibrium"):
        resonance_radius_m(56e9, 1.0, 2.0, harmonic=2, sign=-1)


def test_resonance_mapping_rejects_numpy_boolean_sign():
    with pytest.raises(ValueError, match="sign=-1"):
        resonance_radius_m(56e9, 1.0, 2.0, harmonic=2, sign=np.bool_(True))


def test_ece_rejects_unhashable_mode_cleanly():
    with pytest.raises(ValueError, match="Unsupported mode"):
        analyze_ece([80e9, 90e9], mode=[])


def test_scalar_noise_threshold_is_reported_per_channel():
    t = np.arange(64) * 1e-6
    y = np.column_stack([np.sin(2 * np.pi * 5e3 * t), 0.1 * np.sin(2 * np.pi * 8e3 * t)])
    result = analyze_ece([80e9, 90e9], signal=y, time_s=t,
                         noise_std=0.2, mode="channel_fluctuations")
    assert result["noise_std"].ndim == 0
    np.testing.assert_array_equal(result["above_noise_rms"], [True, False])


def test_fluctuation_input_rejects_nonfinite_samples_and_bad_noise_shape():
    t = np.arange(8, dtype=float)
    y = np.ones((8, 2), dtype=float)
    y[3, 0] = np.nan
    with pytest.raises(ValueError, match="finite samples"):
        analyze_ece([80e9, 90e9], signal=y, time_s=t,
                    mode="channel_fluctuations")
    with pytest.raises(ValueError, match="noise_std"):
        analyze_ece([80e9, 90e9], signal=np.ones((8, 2)), time_s=t,
                    noise_std=[0.1, 0.2, 0.3], mode="channel_fluctuations")


def test_resonance_mapping_requires_paired_field_geometry():
    with pytest.raises(ValueError, match="provided together"):
        analyze_ece([80e9, 90e9], major_radius_m=1.0,
                    mode="resonance_mapping")

def test_ece_profile_reconstruction_from_already_calibrated_multichannel_data():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    f = np.array([80e9, 90e9, 100e9])
    t = np.arange(16, dtype=float) * 1e-6
    calibrated_tb = np.tile([25.0, 45.0, 65.0], (t.size, 1))
    result = reconstruct_ece_temperature_profile(
        f, calibrated_tb, t,
        major_radius_m=1.0, B_axis_T=2.0, harmonic=2,
        assume_optically_thick=True,
    )
    np.testing.assert_allclose(result["brightness_temperature_eV"], [25, 45, 65])
    np.testing.assert_allclose(result["electron_temperature_channel_eV"], [25, 45, 65])
    assert result["profile_radius_m"].size == 100
    assert result["electron_temperature_profile_eV"].shape == (100,)
    assert result["profile_radius_m"][0] < result["profile_radius_m"][-1]
    assert "no extrapolation" in result["profile_method"]


def test_ece_profile_optical_depth_inversion_uses_single_slab_relation():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    f = np.array([80e9, 90e9, 100e9])
    t = np.arange(8, dtype=float) * 1e-6
    tau = np.array([1.0, 1.0, 1.0])
    te = np.array([100.0, 200.0, 300.0])
    tb = te * (1 - np.exp(-tau))
    result = reconstruct_ece_temperature_profile(
        f, np.tile(tb, (t.size, 1)), t,
        major_radius_m=1.0, B_axis_T=2.0, harmonic=2,
        optical_depth=tau,
    )
    np.testing.assert_allclose(result["electron_temperature_channel_eV"], te)


def test_ece_profile_requires_temperature_basis_and_rejects_negative_calibrated_input():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    f = np.array([80e9, 90e9, 100e9])
    t = np.arange(8, dtype=float) * 1e-6
    calibrated_tb = np.ones((t.size, f.size))
    with pytest.raises(ValueError, match="optical_depth"):
        reconstruct_ece_temperature_profile(
            f, calibrated_tb, t, major_radius_m=1.0, B_axis_T=2.0
        )
    invalid_tb = calibrated_tb.copy()
    invalid_tb[:, 1] = -1.0
    with pytest.raises(ValueError, match="brightness temperature"):
        reconstruct_ece_temperature_profile(
            f, invalid_tb, t, major_radius_m=1.0, B_axis_T=2.0,
            assume_optically_thick=True
        )




def test_ece_profile_rejects_negative_sample_even_when_channel_mean_is_positive():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequencies = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    calibrated_tb = np.ones((8, 3), dtype=float)
    calibrated_tb[0, 1] = -2.0  # Mean remains positive, but an input sample is invalid.
    with pytest.raises(ValueError, match="samples must be non-negative"):
        reconstruct_ece_temperature_profile(
            frequencies, calibrated_tb, time_s,
            major_radius_m=1.0, B_axis_T=2.0, harmonic=2,
            assume_optically_thick=True,
        )



def test_ece_profile_uses_explicit_time_window_instead_of_whole_shot_average():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequencies = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    early = np.tile([1.0, 2.0, 3.0], (4, 1))
    late = np.tile([100.0, 200.0, 300.0], (4, 1))
    signal = np.vstack([early, late])
    result = reconstruct_ece_temperature_profile(
        frequencies, signal, time_s,
        major_radius_m=1.0, B_axis_T=2.0, harmonic=2,
        assume_optically_thick=True, time_window_s=(0.0, 3.0),
    )
    np.testing.assert_allclose(result["mean_calibrated_brightness_temperature_eV"], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(result["electron_temperature_channel_eV"], [1.0, 2.0, 3.0])
    assert result["selected_sample_count"] == 4
    assert result["selected_time_window_s"] == [0.0, 3.0]


def test_ece_profile_rejects_invalid_or_out_of_range_time_window():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequencies = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    signal = np.ones((8, 3))
    kwargs = dict(
        major_radius_m=1.0, B_axis_T=2.0, harmonic=2,
        assume_optically_thick=True,
    )
    with pytest.raises(ValueError, match="time_window_s"):
        reconstruct_ece_temperature_profile(
            frequencies, signal, time_s,
            time_window_s=(5.0, 2.0), **kwargs,
        )
    with pytest.raises(ValueError, match="within the supplied time_s range"):
        reconstruct_ece_temperature_profile(
            frequencies, signal, time_s,
            time_window_s=(-1.0, 2.0), **kwargs,
        )


def test_ece_contract_matches_precalibrated_profile_pipeline():
    import json
    from pathlib import Path
    from tokamak_analysis.analysis_registry import ANALYSES
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    root = Path(__file__).resolve().parents[1]
    diagnostic_rows = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    diagnostic = next(row for row in diagnostic_rows if row["diagnostic"].casefold() == "electron cyclotron emission (ece)")
    analysis = next(row for row in ANALYSES if row["id"] == 3)
    contract = analysis["execution_contract"]

    assert callable(reconstruct_ece_temperature_profile)
    assert contract["completeness"] == "complete_with_declared_scope"
    assert "signal_to_brightness_temperature_eV_gain" not in {item["name"] for item in contract["input_specs"]}
    assert "signal_to_brightness_temperature_eV_offset" not in {item["name"] for item in contract["input_specs"]}
    assert any(item["name"] == "calibrated_brightness_temperature_eV" for item in contract["input_specs"])
    assert "must already be calibrated" in diagnostic["execution_contract"]["note"]
    assert diagnostic["implementation_status"].startswith("Complete within declared software scope")



def test_ece_fluctuation_analysis_exposes_welch_psd_and_resolves_tone():
    sample_rate = 20_000.0
    time_s = np.arange(2048, dtype=float) / sample_rate
    tone_hz = 1_250.0
    signal = np.column_stack([
        np.sin(2 * np.pi * tone_hz * time_s),
        0.5 * np.sin(2 * np.pi * 2_000.0 * time_s),
    ])
    result = analyze_ece(
        [80e9, 90e9], signal=signal, time_s=time_s,
        mode="channel_fluctuations",
    )

    assert result["welch_psd"].shape == (
        result["welch_psd_frequency_Hz"].size, 2
    )
    assert result["welch_psd_nperseg"] == 256
    assert result["welch_psd_overlap_samples"] == 128
    peak0 = result["welch_psd_frequency_Hz"][np.argmax(result["welch_psd"][:, 0])]
    peak1 = result["welch_psd_frequency_Hz"][np.argmax(result["welch_psd"][:, 1])]
    assert peak0 == pytest.approx(tone_hz, abs=sample_rate / 256)
    assert peak1 == pytest.approx(2_000.0, abs=sample_rate / 256)

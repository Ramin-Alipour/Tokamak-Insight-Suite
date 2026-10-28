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



def test_welch_psd_has_finite_approximate_confidence_bounds_and_sampling_audit():
    sample_rate = 20_000.0
    time_s = np.arange(2048, dtype=float) / sample_rate
    signal = np.column_stack([
        np.sin(2 * np.pi * 1_250.0 * time_s),
        0.3 * np.sin(2 * np.pi * 2_000.0 * time_s),
    ])
    result = analyze_ece(
        [80e9, 90e9], signal=signal, time_s=time_s,
        mode="channel_fluctuations", psd_confidence_level=0.90,
    )
    assert result["welch_psd_confidence_level"] == pytest.approx(0.90)
    assert result["welch_psd_degrees_of_freedom_approx"] > 2
    assert result["welch_psd_segment_count"] > 1
    assert result["welch_psd_confidence_lower"].shape == result["welch_psd"].shape
    assert result["welch_psd_confidence_upper"].shape == result["welch_psd"].shape
    assert np.all(np.isfinite(result["welch_psd_confidence_lower"]))
    assert np.all(result["welch_psd_confidence_lower"] <= result["welch_psd"])
    assert np.all(result["welch_psd"] <= result["welch_psd_confidence_upper"])
    assert result["sampling_interval_max_deviation_fraction"] < 1e-12
    assert result["sampling_uniformity_tolerance"] == pytest.approx(1e-3)


def test_welch_psd_reports_expected_white_noise_reference_without_subtraction():
    sample_rate = 10_000.0
    time_s = np.arange(1024, dtype=float) / sample_rate
    rng = np.random.default_rng(1234)
    signal = rng.normal(size=(time_s.size, 2))
    result = analyze_ece(
        [80e9, 90e9], signal=signal, time_s=time_s, noise_std=0.5,
        mode="channel_fluctuations",
    )
    assert result["white_noise_psd_expected"].shape == result["welch_psd"].shape
    assert np.all(result["white_noise_psd_expected"] >= 0)
    assert result["welch_psd_above_expected_white_noise"].shape == result["welch_psd"].shape
    # The observed PSD remains untouched; the expected noise PSD is diagnostic only.
    assert np.any(result["welch_psd"] != result["white_noise_psd_expected"])


def test_invalid_psd_confidence_level_is_rejected():
    with pytest.raises(ValueError, match="psd_confidence_level"):
        analyze_ece([80e9, 90e9], signal=np.ones((16, 2)),
                    time_s=np.arange(16), mode="channel_fluctuations",
                    psd_confidence_level=1.0)


def test_profile_uncertainties_bandwidth_envelope_and_time_resolved_output():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float) * 1e-6
    tb = np.tile([20.0, 40.0, 60.0], (8, 1))
    tb[4:, :] += np.array([10.0, 20.0, 30.0])
    result = reconstruct_ece_temperature_profile(
        frequency, tb, time_s, major_radius_m=1.0, B_axis_T=2.0,
        assume_optically_thick=True,
        channel_bandwidth_Hz=[1e9, 1e9, 1e9],
        brightness_temperature_uncertainty_eV=[2.0, 4.0, 6.0],
        frequency_uncertainty_Hz=[1e6, 1e6, 1e6],
        major_radius_uncertainty_m=0.001,
        B_axis_uncertainty_T=0.01,
        include_time_resolved_profile=True,
    )
    assert np.all(
        result["brightness_temperature_uncertainty_eV"] >= np.array([2.0, 4.0, 6.0]) / np.sqrt(8)
    )
    assert np.all(result["effective_sample_count_per_channel"] <= 8)
    assert np.all(result["effective_sample_count_per_channel"] >= 1)
    np.testing.assert_allclose(
        result["electron_temperature_channel_uncertainty_eV"],
        result["brightness_temperature_uncertainty_eV"],
    )
    assert np.all(result["resonance_radius_uncertainty_m"] > 0)
    assert np.all(result["resonance_radius_bandwidth_lower_m"] < result["resonance_radius_m"])
    assert np.all(result["resonance_radius_bandwidth_upper_m"] > result["resonance_radius_m"])
    assert result["electron_temperature_profile_uncertainty_eV"].shape == result["profile_radius_m"].shape
    assert result["electron_temperature_profile_time_eV"].shape == (8, 100)
    assert result["electron_temperature_profile_time_uncertainty_eV"].shape == (8, 100)
    assert result["electron_temperature_channel_time_uncertainty_eV"].shape == (8, 3)
    assert not np.allclose(result["electron_temperature_profile_time_eV"][0],
                           result["electron_temperature_profile_time_eV"][-1])


def test_profile_propagates_optical_depth_and_background_uncertainty():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tau = np.array([0.8, 1.0, 1.2])
    te = np.array([100.0, 200.0, 300.0])
    tb = te * (1.0 - np.exp(-tau))
    result = reconstruct_ece_temperature_profile(
        frequency, np.tile(tb, (8, 1)), time_s,
        major_radius_m=1.0, B_axis_T=2.0, optical_depth=tau,
        brightness_temperature_uncertainty_eV=1.0,
        optical_depth_uncertainty=0.02,
        background_temperature_eV=1.0,
        background_temperature_uncertainty_eV=0.1,
    )
    assert np.all(np.isfinite(result["electron_temperature_channel_uncertainty_eV"]))
    assert np.all(result["electron_temperature_channel_uncertainty_eV"] > 0)
    assert np.all(np.isfinite(result["electron_temperature_profile_uncertainty_eV"]))


def test_low_optical_depth_inversion_exposes_conditioning_warning():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tau = np.array([1e-5, 1.0, 1.2])
    te = np.array([100.0, 200.0, 300.0])
    tb = te * (1.0 - np.exp(-tau))
    result = reconstruct_ece_temperature_profile(
        frequency, np.tile(tb, (8, 1)), time_s,
        major_radius_m=1.0, B_axis_T=2.0, optical_depth=tau,
    )
    assert result["optical_depth_condition_warning"] is True
    assert result["optical_depth_inversion_sensitivity_dTe_dTb"][0] > 1e4


def test_profile_rejects_invalid_bandwidth_and_uncertainty_inputs():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.ones((8, 3))
    kwargs = dict(major_radius_m=1.0, B_axis_T=2.0, assume_optically_thick=True)
    with pytest.raises(ValueError, match="channel_bandwidth_Hz"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s, channel_bandwidth_Hz=[1e9, np.nan, 1e9], **kwargs
        )
    with pytest.raises(ValueError, match="frequency_uncertainty_Hz"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s, frequency_uncertainty_Hz=[1e9, 1e9, 1e11], **kwargs
        )
    with pytest.raises(ValueError, match="optical_depth_uncertainty requires"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s, optical_depth_uncertainty=0.1, **kwargs
        )


def test_ece_dashboard_wires_all_four_workflows_and_new_diagnostics():
    from pathlib import Path

    app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    for expected in (
        "Brightness temperature", "Resonance mapping", "Channel fluctuations / PSD",
        "Te profile reconstruction", "welch_psd_confidence_lower",
        "welch_psd_confidence_upper", "electron_temperature_profile_time_eV",
        "Channel bandwidths in Hz", "Data-quality audit",
    ):
        assert expected in app_source
    assert "st.radio('Temperature inference basis'" in app_source



def test_relativistic_resonance_correction_reduces_to_cold_limit():
    from tokamak_analysis.ece_analysis import resonance_field_relativistic_T

    cold = resonance_field_T(100e9, 2)
    corrected_zero = resonance_field_relativistic_T(
        100e9, 2, electron_energy_eV=0.0, parallel_beta=0.0
    )
    assert corrected_zero == pytest.approx(cold)
    assert resonance_field_relativistic_T(
        100e9, 2, electron_energy_eV=10000.0, parallel_beta=0.0
    ) > cold
    assert resonance_field_relativistic_T(
        100e9, 2, electron_energy_eV=0.0, parallel_beta=0.2
    ) < cold
    with pytest.raises(ValueError, match="parallel_beta"):
        resonance_field_relativistic_T(100e9, 2, parallel_beta=1.0)


def test_psd_confidence_is_validated_only_when_psd_is_requested():
    result = analyze_ece(
        [80e9, 90e9],
        spectral_radiance_W_m2_sr_Hz=[1e-10, 2e-10],
        mode="brightness_temperature",
        psd_confidence_level=None,
    )
    assert result["brightness_temperature_eV"].shape == (2,)
    with pytest.raises(ValueError, match="psd_confidence_level"):
        analyze_ece(
            [80e9, 90e9], signal=np.ones((32, 2)), time_s=np.arange(32),
            mode="channel_fluctuations", psd_confidence_level=1.0,
        )


def test_cross_channel_coherence_and_stationarity_audit_are_returned():
    fs = 10_000.0
    t = np.arange(1024) / fs
    a = np.sin(2 * np.pi * 1000 * t)
    y = np.column_stack([a, 0.7 * a, np.sin(2 * np.pi * 1800 * t)])
    result = analyze_ece(
        [80e9, 90e9, 100e9], signal=y, time_s=t,
        mode="channel_fluctuations", welch_nperseg=128,
    )
    assert len(result["cross_channel_spectra"]) == 3
    assert result["spectral_peak_candidates"]
    assert all(np.isfinite(item["prominence"]) for item in result["spectral_peak_candidates"])
    assert any(abs(item["frequency_Hz"] - 1000.0) < fs / 128
               for item in result["spectral_peak_candidates"] if item["channel_index"] == 0)
    pair = next(p for p in result["cross_channel_spectra"]
                if p["channel_i"] == 0 and p["channel_j"] == 1)
    assert np.nanmax(pair["coherence"]) > 0.95
    assert result["welch_psd_nperseg"] == 128
    assert result["stationarity_chunk_rms_relative_range"].shape == (3,)
    assert "anti-alias" in result["aliasing_assessment"]


def test_profile_uses_user_equilibrium_table_pchip_and_model_discrepancy_floor():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(16) * 1e-6
    tb = np.tile([20.0, 35.0, 60.0], (time_s.size, 1))
    result = reconstruct_ece_temperature_profile(
        frequency, tb, time_s, major_radius_m=1.5, B_axis_T=2.0,
        assume_optically_thick=True,
        equilibrium_radius_m=[1.0, 1.5, 2.0, 2.5],
        equilibrium_toroidal_field_T=[2.1, 1.8, 1.5, 1.2],
        profile_method="pchip",
        brightness_temperature_uncertainty_eV=1.0,
        model_discrepancy_fraction=0.1,
    )
    assert result["equilibrium_model"].startswith("user-supplied monotonic")
    assert result["profile_method"].startswith("pchip")
    assert result["profile_gradient_eV_per_m"].shape == result["profile_radius_m"].shape
    assert result["profile_spatial_quality"]["output_grid_is_not_measurement_resolution"]
    assert result["electron_temperature_profile_uncertainty_eV"] is not None
    assert result["model_discrepancy_fraction_user_supplied"] == pytest.approx(0.1)


def test_profile_rejects_nonmonotonic_equilibrium_and_bad_covariance():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.tile([10.0, 20.0, 30.0], (8, 1))
    common = dict(major_radius_m=1.5, B_axis_T=2.0, assume_optically_thick=True)
    with pytest.raises(ValueError, match="strictly monotonic"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s,
            equilibrium_radius_m=[1.0, 1.5, 2.0],
            equilibrium_toroidal_field_T=[1.0, 2.0, 1.5],
            **common,
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s,
            channel_covariance_eV2=[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            **common,
        )



def test_explicit_geometry_derived_channel_radii_override_circular_mapping():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float) * 1e-6
    tb = np.tile([10.0, 20.0, 15.0], (time_s.size, 1))
    result = reconstruct_ece_temperature_profile(
        frequency, tb, time_s, major_radius_m=1.5, B_axis_T=2.0,
        assume_optically_thick=True,
        channel_resonance_radius_m=[1.2, 1.6, 1.9],
        radius_mapping_uncertainty_m=[0.01, 0.02, 0.03],
    )
    np.testing.assert_allclose(result["resonance_radius_m"], [1.2, 1.6, 1.9])
    np.testing.assert_allclose(result["resonance_radius_uncertainty_m"], [0.01, 0.02, 0.03])
    assert "user-supplied channel resonance locations" in result["equilibrium_model"]
    assert result["resonance_radius_bandwidth_lower_m"] is None
    assert result["resonance_radius_bandwidth_upper_m"] is None


def test_explicit_geometry_mapping_rejects_conflicting_or_malformed_inputs():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.ones((8, 3))
    common = dict(major_radius_m=1.5, B_axis_T=2.0, assume_optically_thick=True)
    with pytest.raises(ValueError, match="one finite positive radius per frequency"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s, channel_resonance_radius_m=[1.0, np.nan, 2.0], **common
        )
    with pytest.raises(ValueError, match="not both"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s,
            channel_resonance_radius_m=[1.0, 1.5, 2.0],
            equilibrium_radius_m=[1.0, 1.5, 2.0],
            equilibrium_toroidal_field_T=[2.0, 1.5, 1.0],
            **common,
        )



def test_receiver_response_convolution_preserves_flat_spectrum_and_no_calibration():
    from tokamak_analysis.ece_analysis import convolve_ece_receiver_response

    frequency = np.linspace(80e9, 100e9, 101)
    radiance = np.full(frequency.size, 2.5e-10)
    response_frequency = np.linspace(85e9, 95e9, 21)
    response = np.vstack([
        np.exp(-0.5 * ((response_frequency - center) / 1e9) ** 2)
        for center in [88e9, 92e9]
    ])
    result = convolve_ece_receiver_response(
        frequency, radiance, response_frequency, response,
    )
    np.testing.assert_allclose(
        result["channel_spectral_radiance_W_m2_sr_Hz"], [2.5e-10, 2.5e-10],
        rtol=1e-12,
    )
    assert result["response_convolution_performed"] is True
    assert result["calibration_performed"] is False
    assert result["effective_center_frequency_Hz"].shape == (2,)
    assert np.all(result["effective_center_frequency_Hz"] > 85e9)
    assert np.all(result["effective_center_frequency_Hz"] < 95e9)


def test_receiver_response_rejects_uncovered_or_invalid_response():
    from tokamak_analysis.ece_analysis import convolve_ece_receiver_response

    frequency = np.linspace(80e9, 100e9, 10)
    radiance = np.ones(10)
    with pytest.raises(ValueError, match="within the measured radiance spectrum"):
        convolve_ece_receiver_response(
            frequency, radiance, [79e9, 90e9, 95e9], [[1.0, 1.0, 1.0]],
        )
    with pytest.raises(ValueError, match="positive integrated weight"):
        convolve_ece_receiver_response(
            frequency, radiance, [85e9, 90e9, 95e9], [[0.0, 0.0, 0.0]],
        )


def test_extreme_frequencies_fail_cleanly_instead_of_returning_infinite_fields():
    with pytest.raises(ValueError, match="finite positive"):
        resonance_field_T(np.inf, 1)
    with pytest.raises(ValueError, match="overflowed"):
        brightness_temperature_K(1e308, 1e9)



def test_profile_checks_user_supplied_physical_plasma_radius_bounds():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.tile([10.0, 20.0, 30.0], (8, 1))
    with pytest.raises(ValueError, match="outside the user-supplied plasma"):
        reconstruct_ece_temperature_profile(
            frequency, tb, time_s, major_radius_m=1.5, B_axis_T=2.0,
            assume_optically_thick=True,
            channel_resonance_radius_m=[1.1, 1.5, 2.1],
            plasma_radius_bounds_m=[1.2, 2.0],
        )


def test_full_channel_covariance_changes_profile_uncertainty():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequency = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.tile([10.0, 20.0, 40.0], (8, 1))
    common = dict(
        frequency_Hz=frequency, calibrated_brightness_temperature_eV=tb,
        time_s=time_s, major_radius_m=1.5, B_axis_T=2.0,
        assume_optically_thick=True, profile_method="linear",
    )
    diagonal = np.diag([1.0, 1.0, 1.0])
    correlated = np.full((3, 3), 0.8)
    np.fill_diagonal(correlated, 1.0)
    independent_result = reconstruct_ece_temperature_profile(
        **common, channel_covariance_eV2=diagonal,
    )
    correlated_result = reconstruct_ece_temperature_profile(
        **common, channel_covariance_eV2=correlated,
    )
    assert np.sum(correlated_result["electron_temperature_profile_uncertainty_eV"]**2) > np.sum(
        independent_result["electron_temperature_profile_uncertainty_eV"]**2
    )



def test_profile_rejects_duplicate_channel_resonance_radii():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    with pytest.raises(ValueError, match="must be distinct"):
        reconstruct_ece_temperature_profile(
            [80e9, 90e9, 100e9], np.tile([10.0, 20.0, 30.0], (8, 1)),
            np.arange(8, dtype=float), major_radius_m=1.5, B_axis_T=2.0,
            assume_optically_thick=True,
            channel_resonance_radius_m=[1.2, 1.5, 1.5],
        )



def test_ece_mapping_rejects_invalid_geometry_and_bandwidth():
    with pytest.raises(ValueError, match="major_radius_m"):
        analyze_ece([80e9, 90e9], major_radius_m=-1.0, B_axis_T=2.0,
                    mode="resonance_mapping")
    with pytest.raises(ValueError, match="B_axis_T"):
        analyze_ece([80e9, 90e9], major_radius_m=1.0, B_axis_T=np.inf,
                    mode="resonance_mapping")
    with pytest.raises(ValueError, match="bandwidth"):
        analyze_ece([80e9, 90e9], channel_bandwidth_Hz=[200e9, 1e9],
                    spectral_radiance_W_m2_sr_Hz=[1e-10, 1e-10])


def test_ece_relativistic_overflow_is_rejected_and_metadata_is_truthful():
    from tokamak_analysis.ece_analysis import resonance_field_relativistic_T, reconstruct_ece_temperature_profile

    with pytest.raises(ValueError, match="overflowed"):
        resonance_field_relativistic_T(100e9, 2, electron_energy_eV=1e308)
    result = reconstruct_ece_temperature_profile(
        [80e9, 90e9, 100e9], np.tile([20.0, 30.0, 40.0], (8, 1)),
        np.arange(8, dtype=float) * 1e-6,
        major_radius_m=1.0, B_axis_T=2.0, assume_optically_thick=True,
        electron_energy_eV=1000.0, parallel_beta=0.1,
    )
    assert "monoenergetic relativistic/Doppler" in result["resonance_model"]
    assert not any(item == "no relativistic or Doppler correction" for item in result["limitations"])


def test_ece_profile_accepts_singleton_optical_depth_and_rejects_array_scalars():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequencies = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.tile([10.0, 20.0, 30.0], (8, 1))
    result = reconstruct_ece_temperature_profile(
        frequencies, tb, time_s, major_radius_m=1.0, B_axis_T=2.0,
        optical_depth=[1.0],
    )
    np.testing.assert_allclose(
        result["electron_temperature_channel_eV"],
        np.array([10.0, 20.0, 30.0]) / (1.0 - np.exp(-1.0)),
    )
    with pytest.raises(ValueError, match="major_radius_m"):
        reconstruct_ece_temperature_profile(
            frequencies, tb, time_s, major_radius_m=[1.0, 1.1], B_axis_T=2.0,
            assume_optically_thick=True,
        )


def test_ece_covariance_psd_validation_is_scale_aware():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequencies = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.tile([10.0, 20.0, 30.0], (8, 1))
    covariance = np.diag([1e-20, 1e-20, -1e-14])
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_ece_temperature_profile(
            frequencies, tb, time_s, major_radius_m=1.0, B_axis_T=2.0,
            assume_optically_thick=True, channel_covariance_eV2=covariance,
        )


def test_mean_profile_uses_channel_covariance_but_time_resolved_uncertainty_does_not():
    from tokamak_analysis.ece_analysis import reconstruct_ece_temperature_profile

    frequencies = np.array([80e9, 90e9, 100e9])
    time_s = np.arange(8, dtype=float)
    tb = np.tile([10.0, 20.0, 30.0], (8, 1))
    diagonal = np.diag([1.0, 1.0, 1.0])
    correlated = np.array([[1.0, 0.8, 0.0], [0.8, 1.0, 0.0], [0.0, 0.0, 1.0]])
    kwargs = dict(
        frequency_Hz=frequencies, calibrated_brightness_temperature_eV=tb,
        time_s=time_s, major_radius_m=1.0, B_axis_T=2.0,
        assume_optically_thick=True, include_time_resolved_profile=True,
        brightness_temperature_uncertainty_eV=1.0,
    )
    independent = reconstruct_ece_temperature_profile(
        **kwargs, channel_covariance_eV2=diagonal
    )
    correlated_result = reconstruct_ece_temperature_profile(
        **kwargs, channel_covariance_eV2=correlated
    )
    assert not np.allclose(
        independent["electron_temperature_profile_uncertainty_eV"],
        correlated_result["electron_temperature_profile_uncertainty_eV"],
    )
    np.testing.assert_allclose(
        independent["electron_temperature_profile_time_uncertainty_eV"],
        correlated_result["electron_temperature_profile_time_uncertainty_eV"],
    )
    assert "covariance of the time-window mean" in correlated_result["time_resolved_uncertainty_note"]


def test_ece_fluctuation_psd_rejects_too_short_uniform_records():
    with pytest.raises(ValueError, match="at least 8 samples"):
        analyze_ece(
            [80e9], signal=np.ones((7, 1)), time_s=np.arange(7, dtype=float),
            mode="channel_fluctuations",
        )

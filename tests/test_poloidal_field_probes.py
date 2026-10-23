import numpy as np
import pytest

from tokamak_analysis.poloidal_field_probes import analyze_poloidal_field_probes


def test_calibrates_signal_and_derivative_for_uniform_ramp():
    t = np.linspace(0, 1, 101)
    signal = 2 * t
    result = analyze_poloidal_field_probes(
        t, signal, calibration_gain_T_per_unit=0.5,
        calibration_offset_T=0.1,
    )
    np.testing.assert_allclose(result["B_poloidal_T"], t + 0.1)
    np.testing.assert_allclose(result["dB_poloidal_dt_T_s"][1:-1], 1.0, atol=1e-12)
    assert result["sampling_rate_median_Hz"] == pytest.approx(100)


def test_baseline_and_multichannel_calibration():
    t = np.arange(5.0)
    y = np.column_stack([np.ones(5) * 3, np.ones(5) * 4])
    result = analyze_poloidal_field_probes(
        t, y, calibration_gain_T_per_unit=[2, 0.5],
        calibration_offset_T=[1, -1], baseline_interval_s=(0, 1),
        probe_names=["inner", "outer"],
    )
    np.testing.assert_allclose(result["B_poloidal_T"], 0)
    np.testing.assert_allclose(result["baseline_removed_T"], [7, 1])
    assert result["probe_names"] == ["inner", "outer"]


def test_uncertainty_propagation_and_nonuniform_time():
    t = np.array([0.0, 0.1, 0.21, 0.3, 0.4])
    y = np.arange(5.0)
    result = analyze_poloidal_field_probes(t, y, signal_uncertainty=0.2)
    assert np.all(result["B_poloidal_uncertainty_T"] == pytest.approx(0.2))
    assert np.all(np.isfinite(result["dB_poloidal_dt_uncertainty_T_s"]))
    assert "sampling_warning" in result


@pytest.mark.parametrize("time,signal", [
    ([0, 1], [1, 2]),
    ([0, 1, 1], [1, 2, 3]),
    ([0, 1, 2], [1, np.nan, 3]),
    ([0, 1, 2], [[1, 2], [3, 4]]),
])
def test_rejects_invalid_input(time, signal):
    with pytest.raises(ValueError):
        analyze_poloidal_field_probes(time, signal)


def test_rejects_invalid_uncertainty_and_gain():
    t = np.arange(4.0)
    with pytest.raises(ValueError):
        analyze_poloidal_field_probes(t, np.arange(4.0), calibration_gain_T_per_unit=0)
    with pytest.raises(ValueError):
        analyze_poloidal_field_probes(t, np.arange(4.0), signal_uncertainty=-1)



@pytest.mark.parametrize("interval", [1.0, (0.0,), (0.0, 1.0, 2.0), ("bad", 1.0)])
def test_rejects_malformed_baseline_interval(interval):
    t = np.arange(5.0)
    with pytest.raises(ValueError, match="baseline_interval_s"):
        analyze_poloidal_field_probes(t, np.arange(5.0), baseline_interval_s=interval)


def test_rejects_single_string_as_probe_names():
    t = np.arange(4.0)
    with pytest.raises(ValueError, match="probe_names"):
        analyze_poloidal_field_probes(t, np.arange(4.0), probe_names="inner")


def test_multichannel_uncertainty_and_nonuniform_derivative_shapes():
    t = np.array([0.0, 0.1, 0.23, 0.4])
    y = np.column_stack((2 * t + 1, -3 * t + 2))
    u = np.full_like(y, 0.05)
    result = analyze_poloidal_field_probes(
        t, y, calibration_gain_T_per_unit=[0.5, 2.0],
        signal_uncertainty=u, probe_names=("inner", "outer"),
    )
    assert result["B_poloidal_T"].shape == (t.size, 2)
    assert result["dB_poloidal_dt_T_s"].shape == (t.size, 2)
    assert result["B_poloidal_uncertainty_T"].shape == (t.size, 2)
    assert result["dB_poloidal_dt_uncertainty_T_s"].shape == (t.size, 2)
    np.testing.assert_allclose(result["dB_poloidal_dt_T_s"][1:-1, 0], 1.0, atol=1e-12)
    np.testing.assert_allclose(result["dB_poloidal_dt_T_s"][1:-1, 1], -6.0, atol=1e-12)

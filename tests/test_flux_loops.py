import numpy as np
import pytest

from tokamak_analysis.flux_loops import analyze_flux_loop, flux_from_voltage


def test_constant_voltage_faraday_sign_and_turns():
    t = np.linspace(0, 2, 5)
    result = analyze_flux_loop(t, np.full(5, 3.0), turns=3, initial_flux_Wb=0.4)
    np.testing.assert_allclose(result.flux_change_Wb, -t)
    np.testing.assert_allclose(result.flux_Wb, 0.4 - t)
    np.testing.assert_allclose(result.loop_voltage_integral_Vs, 3 * t)
    assert result.diagnostics["sample_count"] == 5


def test_polarity_and_offset_correction():
    result = analyze_flux_loop([0, 1, 2], [4, 4, 4], voltage_offset_V=1, polarity=-1)
    np.testing.assert_allclose(result.voltage_V, [-3, -3, -3])
    np.testing.assert_allclose(result.flux_change_Wb, [0, 1.5, 3])


def test_nonuniform_sampling_uses_trapezoid():
    result = analyze_flux_loop([0, 1, 3], [0, 2, 2])
    np.testing.assert_allclose(result.loop_voltage_integral_Vs, [0, 1, 5])
    np.testing.assert_allclose(result.flux_change_Wb, [0, -1, -5])


@pytest.mark.parametrize("time,voltage", [
    ([0, 1], [1]), ([0, 0], [1, 1]), ([0, np.nan], [1, 1]),
    ([0, 1], [1, np.inf]), ([0], [1]), ([[0, 1]], [[1, 1]]),
])
def test_invalid_signal_arrays_rejected(time, voltage):
    with pytest.raises(ValueError):
        analyze_flux_loop(time, voltage)


@pytest.mark.parametrize("kwargs", [
    {"turns": 0}, {"turns": 1.5}, {"turns": True},
    {"polarity": 0}, {"initial_flux_Wb": np.nan},
    {"voltage_offset_V": np.inf}, {"max_gap_s": 0},
])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        analyze_flux_loop([0, 1], [1, 1], **kwargs)


def test_maximum_gap_guard():
    with pytest.raises(ValueError, match="gaps"):
        analyze_flux_loop([0, 1, 4], [1, 1, 1], max_gap_s=2)


def test_zero_voltage_preserves_initial_flux_and_serializes():
    result = flux_from_voltage([0, 1, 2], [0, 0, 0], initial_flux_Wb=2.5)
    np.testing.assert_allclose(result.flux_Wb, 2.5)
    assert result.to_dict()["flux_Wb"] == [2.5, 2.5, 2.5]

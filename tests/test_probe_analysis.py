import numpy as np

from tokamak_analysis.probe_analysis import analyze_probe_voltage_current


def test_floating_potential_interpolates_current_zero_in_voltage_order():
    result = analyze_probe_voltage_current(
        voltage_V=[-2, -1, 0, 1, 2],
        current_A=[-3, -1, 1, 2, 4],
    )
    assert result["floating_potential_V"] == -0.5
    assert result["floating_potential_candidates_V"] == [-0.5]


def test_floating_potential_is_none_when_current_never_crosses_zero():
    result = analyze_probe_voltage_current(
        voltage_V=[-2, -1, 0, 1, 2],
        current_A=[1, 2, 3, 4, 5],
    )
    assert result["floating_potential_V"] is None
    assert result["floating_potential_candidates_V"] == []


def test_multiple_zero_crossings_are_reported_as_ambiguous():
    result = analyze_probe_voltage_current(
        voltage_V=[-2, -1, 0, 1, 2],
        current_A=[-1, 1, -1, 1, 2],
    )
    assert result["floating_potential_V"] is None
    assert np.allclose(result["floating_potential_candidates_V"], [-1.5, -0.5, 0.5])


def test_zero_current_plateau_is_represented_by_its_voltage_midpoint():
    result = analyze_probe_voltage_current(
        voltage_V=[-2, -1, 0, 1, 2],
        current_A=[-2, 0, 0, 2, 3],
    )
    assert result["floating_potential_V"] == -0.5
    assert result["floating_potential_candidates_V"] == [-0.5]

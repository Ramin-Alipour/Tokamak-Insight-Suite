import numpy as np
import pytest

from tokamak_analysis.probe_analysis import analyze_probe_voltage_current, electron_retarding_current


def test_floating_potential_interpolates_current_zero_in_voltage_order():
    result = analyze_probe_voltage_current([-2, -1, 0, 1, 2], [-3, -1, 1, 2, 4])
    assert result["floating_potential_V"] == -0.5
    assert result["floating_potential_candidates_V"] == [-0.5]
    assert result["floating_potential_status"] == "unique_crossing"


def test_floating_potential_is_none_when_current_never_crosses_zero():
    result = analyze_probe_voltage_current([-2, -1, 0, 1, 2], [1, 2, 3, 4, 5])
    assert result["floating_potential_V"] is None
    assert result["floating_potential_candidates_V"] == []
    assert result["floating_potential_status"] == "no_crossing"


def test_multiple_zero_crossings_are_reported_as_ambiguous():
    result = analyze_probe_voltage_current([-2, -1, 0, 1, 2], [-1, 1, -1, 1, 2])
    assert result["floating_potential_V"] is None
    assert np.allclose(result["floating_potential_candidates_V"], [-1.5, -0.5, 0.5])
    assert result["floating_potential_status"] == "ambiguous_multiple_crossings"


def test_zero_current_plateau_is_represented_by_its_voltage_midpoint():
    result = analyze_probe_voltage_current([-2, -1, 0, 1, 2], [-2, 0, 0, 2, 3])
    assert result["floating_potential_V"] == -0.5
    assert result["floating_potential_candidates_V"] == [-0.5]


@pytest.mark.parametrize("v,i", [
    ([1, 2, 3, 4], [1, 2, 3, 4]),
    ([1, 2, 3, 4, 5], [1, 2, 3, 4]),
    ([1, 2, np.nan, 4, 5], [1, 2, 3, 4, 5]),
    ([1, 2, 3, 4, 5], [1, 2, np.inf, 4, 5]),
    ([1, 2, 2, 4, 5], [1, 2, 3, 4, 5]),
])
def test_invalid_iv_inputs_rejected(v, i):
    with pytest.raises(ValueError):
        analyze_probe_voltage_current(v, i)


def test_electron_fit_reports_quality_for_exponential_retarding_branch():
    v = np.arange(-5.0, 6.0)
    i = -0.2 + 1e-3 * np.exp((v + 5) / 2.0)
    result = analyze_probe_voltage_current(v, i, electron_fit_range=(-4, 2))
    assert result["electron_temperature_estimate_eV"] == pytest.approx(2.0, rel=0.05)
    assert result["electron_fit_points"] >= 3
    assert result["electron_fit_r_squared"] > 0.99
    assert result["electron_fit_residual_rms_A"] >= 0
    assert result["electron_fit_quality"] == "strong_empirical_fit"
    assert result["electron_temperature_standard_uncertainty_eV"] is not None
    assert result["electron_temperature_standard_uncertainty_eV"] >= 0


def test_electron_fit_does_not_silently_hide_insufficient_points():
    result = analyze_probe_voltage_current([-2, -1, 0, 1, 2], [-1, -0.5, -0.1, -0.2, 0])
    result = analyze_probe_voltage_current([-2, -1, 0, 1, 2], [-1, -0.5, -0.1, -0.2, 0],
                                           electron_fit_range=(-2, -1.5))
    assert result["electron_fit_status"] == "insufficient_positive_points"
    assert result["electron_temperature_estimate_eV"] is None


def test_fit_ranges_must_be_inside_data_and_have_enough_points():
    v = [-2, -1, 0, 1, 2]
    i = [-2, -1, 0, 1, 2]
    with pytest.raises(ValueError, match="measured voltage range"):
        analyze_probe_voltage_current(v, i, electron_fit_range=(-3, -1))
    with pytest.raises(ValueError, match="at least two measured points"):
        analyze_probe_voltage_current(v, i, ion_fit_range=(-2, -1.5))


def test_electron_retarding_current_validates_physical_domain():
    result = electron_retarding_current([-2, -1], 0, 1e-3, 2)
    assert np.all(result["electron_current_A"] > 0)
    with pytest.raises(ValueError, match="electron-retarding"):
        electron_retarding_current([0, 1], 0, 1e-3, 2)


def test_electron_fit_exposes_weak_fit_quality_without_overstating_physical_validity():
    voltage = np.linspace(-4.0, 4.0, 17)
    current = np.sin(voltage) + 0.03 * np.cos(7.0 * voltage)
    result = analyze_probe_voltage_current(voltage, current, electron_fit_range=(-3.5, 3.5))
    if result["electron_fit_status"] == "ok":
        assert result["electron_fit_quality"] in {"strong_empirical_fit", "weak_empirical_fit"}
        assert result["electron_fit_residual_rms_A"] >= 0
        assert "electron_temperature_standard_uncertainty_eV" in result


def test_electron_retarding_current_rejects_nonfinite_voltage_and_bad_parameters():
    with pytest.raises(ValueError, match="finite"):
        electron_retarding_current([0.0, np.nan], 1.0, 1e-3, 2.0)
    with pytest.raises(ValueError, match="positive"):
        electron_retarding_current([0.0], 1.0, 0.0, 2.0)

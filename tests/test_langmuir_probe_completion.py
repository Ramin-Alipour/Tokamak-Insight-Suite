import numpy as np
import pytest

from tis.probe_analysis import langmuir_iv_analysis


def test_floating_potential_is_interpolated_and_reports_candidates():
    v = np.arange(-4.0, 5.0)
    i = v + 0.5
    result = langmuir_iv_analysis(v, i)
    assert result["floating_potential_V"] == pytest.approx(-0.5)
    assert result["floating_potential_candidates_V"] == pytest.approx([-0.5])
    assert result["floating_potential_status"] == "single_zero_crossing"


def test_zero_current_plateau_uses_midpoint():
    v = np.arange(-4.0, 5.0)
    i = np.array([-3., -2., -1., 0., 0., 1., 2., 3., 4.])
    result = langmuir_iv_analysis(v, i)
    assert result["floating_potential_V"] == pytest.approx(-0.5)
    assert result["floating_potential_candidates_V"] == pytest.approx([-0.5])


def test_multiple_zero_crossings_are_not_silently_selected():
    v = np.arange(-4.0, 5.0)
    i = np.array([-2., 1., -1., 1., -1., 1., 2., 3., 4.])
    result = langmuir_iv_analysis(v, i)
    assert result["floating_potential_V"] is None
    assert len(result["floating_potential_candidates_V"]) >= 3
    assert result["floating_potential_status"] == "ambiguous_multiple_zero_crossings"


@pytest.mark.parametrize("fit_range", [(2, 2), (3, 1), (np.nan, 2), (0, np.inf), (1,), (1, 2, 3)])
def test_invalid_electron_fit_ranges_are_rejected(fit_range):
    with pytest.raises(ValueError, match="strictly increasing"):
        langmuir_iv_analysis(np.arange(8.), np.arange(8.), electron_fit_range=fit_range)


@pytest.mark.parametrize("fit_range", [(2, 2), (3, 1), (np.nan, 2), (0, np.inf), (1,), (1, 2, 3)])
def test_invalid_ion_fit_ranges_are_rejected(fit_range):
    with pytest.raises(ValueError, match="strictly increasing"):
        langmuir_iv_analysis(np.arange(8.), np.arange(8.), ion_fit_range=fit_range)


def test_ion_positive_convention_is_equivalent_to_electron_positive():
    v = np.linspace(-5, 5, 21)
    current = np.sinh(v / 3.0)
    a = langmuir_iv_analysis(v, current, current_convention="electron_positive")
    b = langmuir_iv_analysis(v, -current, current_convention="ion_positive")
    assert a["floating_potential_V"] == pytest.approx(b["floating_potential_V"])
    np.testing.assert_allclose(a["current_min_A"], b["current_min_A"])
    np.testing.assert_allclose(a["current_max_A"], b["current_max_A"])


def test_duplicate_voltage_values_are_aggregated_after_pair_filtering():
    v = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 3, 4], dtype=float)
    i = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 5, 4], dtype=float)
    result = langmuir_iv_analysis(v, i)
    assert result["n_points"] == 9
    assert result["floating_potential_V"] == pytest.approx(0.0)


def test_nonfinite_samples_are_removed_but_reported_point_count_is_cleaned():
    v = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, np.nan])
    i = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 10])
    result = langmuir_iv_analysis(v, i)
    assert result["n_points"] == 9
    assert np.isfinite(result["max_dIdV_A_per_V"])


def _synthetic_iv_for_regression():
    voltage = np.linspace(-20.0, 20.0, 81)
    current = -1.0e-3 + 2.0e-6 * np.exp((voltage + 20.0) / 5.0)
    return voltage, current


def test_langmuir_iv_recovers_synthetic_electron_temperature():
    voltage, current = _synthetic_iv_for_regression()
    result = langmuir_iv_analysis(
        voltage, current, electron_fit_range=(-5.0, 10.0),
        ion_fit_range=(-20.0, -12.0),
    )
    assert result["electron_temperature_eV"] == pytest.approx(5.0, rel=0.08)
    assert result["electron_retarding_fit_R2"] > 0.99


@pytest.mark.parametrize("bad", [
    ["1", "2", "3", "4", "5", "6", "7", "8"],
    [True, False, True, False, True, False, True, False],
])
def test_langmuir_iv_rejects_non_numeric_or_boolean_voltage(bad):
    with pytest.raises(ValueError, match="real numeric"):
        langmuir_iv_analysis(bad, np.arange(8, dtype=float))


def test_langmuir_iv_rejects_electron_window_without_positive_excess():
    voltage = np.linspace(-10.0, 10.0, 41)
    current = np.full_like(voltage, -1.0e-3)
    with pytest.raises(ValueError, match="above the estimated ion-current baseline"):
        langmuir_iv_analysis(
            voltage, current, electron_fit_range=(-5.0, 5.0)
        )


def test_langmuir_iv_does_not_return_geometry_dependent_density():
    voltage, current = _synthetic_iv_for_regression()
    result = langmuir_iv_analysis(voltage, current)
    assert "ion_density_estimate_m-3" not in result
    assert "ion_density_status" not in result
    assert "ion_density_model" not in result

def test_one_shot_fit_range_iterables_are_normalized_and_reusable():
    voltage, current = _synthetic_iv_for_regression()
    expected = langmuir_iv_analysis(
        voltage, current,
        electron_fit_range=(-5.0, 10.0),
        ion_fit_range=(-20.0, -12.0),
    )
    actual = langmuir_iv_analysis(
        voltage, current,
        electron_fit_range=iter((-5.0, 10.0)),
        ion_fit_range=iter((-20.0, -12.0)),
    )
    assert actual["electron_temperature_eV"] == pytest.approx(
        expected["electron_temperature_eV"], rel=1e-12
    )
    assert actual["ion_saturation_current_A"] == pytest.approx(
        expected["ion_saturation_current_A"], rel=1e-12
    )
    assert actual["electron_retarding_fit_R2"] == pytest.approx(
        expected["electron_retarding_fit_R2"], rel=1e-12
    )

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

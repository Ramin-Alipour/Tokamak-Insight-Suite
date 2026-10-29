import numpy as np
import pytest

from tis.probe_analysis import floating_potential, langmuir_iv_analysis, probe_iv_summary


def test_langmuir_analysis_reports_preprocessing_quality_and_duplicate_aggregation():
    voltage = np.r_[np.linspace(-5.0, 5.0, 11), 0.0, np.nan]
    current = np.r_[np.linspace(-2e-3, 2e-3, 11), 1e-3, 0.0]

    result = langmuir_iv_analysis(voltage, current)

    quality = result["input_quality"]
    assert quality["input_pair_count"] == 13
    assert quality["finite_pair_count"] == 12
    assert quality["discarded_nonfinite_pair_count"] == 1
    assert quality["duplicate_voltage_pair_count_aggregated"] == 1
    assert quality["processed_unique_voltage_count"] == 11
    assert quality["current_convention_normalized"] == "electron_positive"

    processed_v = np.asarray(result["processed_voltage_V"])
    assert np.all(np.diff(processed_v) > 0)
    assert len(processed_v) == quality["processed_unique_voltage_count"]
    assert len(result["processed_current_A"]) == len(processed_v)
    assert len(result["dIdV_A_per_V"]) == len(processed_v)
    assert processed_v.tolist() == sorted(set(voltage[np.isfinite(voltage)].tolist()))



def test_langmuir_analysis_exposes_electron_fit_residual_diagnostics():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -1e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)

    result = langmuir_iv_analysis(
        voltage,
        current,
        electron_fit_range=(-8.0, 0.0),
    )

    assert result["electron_temperature_eV"] is not None
    assert result["electron_retarding_fit_points"] >= 4
    assert np.isfinite(result["electron_retarding_fit_R2"])
    assert np.isfinite(result["electron_retarding_fit_residual_rms_A"])
    assert np.isfinite(result["electron_retarding_fit_normalized_rmse"])
    assert result["electron_retarding_fit_status"] == "fit_completed_review_R2_and_residuals"


def test_floating_potential_helper_uses_zero_plateau_midpoint():
    assert floating_potential([-2, -1, 0, 1, 2], [-2, 0, 0, 2, 3]) == -0.5
    summary = probe_iv_summary([-2, -1, 0, 1, 2], [-2, 0, 0, 2, 3])
    assert summary["floating_potential_interpolated_V"] == -0.5
    assert summary["floating_potential_candidates_V"] == [-0.5]
    assert summary["floating_potential_status"] == "single_zero_crossing"


def test_floating_potential_helper_does_not_hide_multiple_crossings():
    voltage = [-2, -1, 0, 1, 2]
    current = [-1, 1, -1, 1, 2]
    assert floating_potential(voltage, current) is None
    summary = probe_iv_summary(voltage, current)
    assert summary["floating_potential_interpolated_V"] is None
    assert summary["floating_potential_status"] == "ambiguous_multiple_zero_crossings"
    assert np.allclose(summary["floating_potential_candidates_V"], [-1.5, -0.5, 0.5])


def test_floating_potential_helper_validates_array_shape_and_averages_duplicate_voltage():
    with pytest.raises(ValueError, match="one-dimensional"):
        floating_potential([[0, 1]], [[-1, 1]])
    assert floating_potential([-1, 0, 0, 1], [-1, -1, 1, 1]) == 0.0


def test_langmuir_ion_branch_fit_reports_residual_quality_diagnostics():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2e-3 + 2e-5 * voltage
    result = langmuir_iv_analysis(
        voltage,
        current,
        ion_fit_range=(-10.0, -5.0),
    )

    assert result["ion_saturation_fit_points"] >= 3
    assert np.isfinite(result["ion_saturation_fit_R2"])
    assert np.isfinite(result["ion_saturation_fit_residual_rms_A"])
    assert result["ion_saturation_fit_normalized_rmse"] is not None
    assert result["ion_saturation_fit_status"] == "fit_completed_review_R2_and_residuals"
    assert result["ion_saturation_current_method"] == "median_user_selected_ion_fit_range"

"""Tests for baseline versus treatment comparison statistics."""
import numpy as np
import pytest

from tis.core import compare_condition_signals


def test_condition_comparison_recovers_known_mean_and_rms_changes():
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = 0.6 * baseline
    result = compare_condition_signals(baseline, treatment)
    assert result["mean_change_percent_relative_to_abs_baseline_mean"] == pytest.approx(-40.0)
    assert result["RMS_change_percent"] == pytest.approx(-40.0)
    assert result["fraction_of_paired_samples_treatment_below_baseline"] == 1.0


def test_condition_comparison_withholds_unstable_mean_percent_for_zero_mean_signal():
    baseline = np.array([-1.0, 1.0, -2.0, 2.0])
    treatment = 0.5 * baseline
    result = compare_condition_signals(baseline, treatment)
    assert result["mean_change_percent_relative_to_abs_baseline_mean"] is None
    assert result["baseline_mean_percent_change_status"] == "UNSTABLE_NEAR_ZERO_BASELINE"
    assert result["RMS_change_percent"] == pytest.approx(-50.0)


def test_condition_comparison_applies_paired_time_window_and_finite_mask():
    time = np.arange(6, dtype=float)
    baseline = np.array([1, 1, 2, 2, 3, np.nan], dtype=float)
    treatment = np.array([2, 2, 1, 1, 1, 5], dtype=float)
    result = compare_condition_signals(baseline, treatment, time_s=time, start_s=2, end_s=4)
    assert result["n_paired_samples"] == 3
    assert result["n_excluded_samples"] == 3
    assert result["time_window_s"] == [2.0, 4.0]
    assert result["mean_difference_treatment_minus_baseline"] == pytest.approx(-1.3333333333)


def test_condition_comparison_rejects_bad_window_or_mismatched_arrays():
    with pytest.raises(ValueError, match="equal lengths"):
        compare_condition_signals([1, 2], [1, 2, 3])
    with pytest.raises(ValueError, match="start_s < end_s"):
        compare_condition_signals([1, 2], [2, 3], start_s=2, end_s=1)

"""Curve-level reference comparison regression tests."""
import numpy as np
import pandas as pd
import pytest

from tis.validation import compare_reference_series


def test_curve_comparison_interpolates_only_overlapping_domain():
    frame = pd.DataFrame({
        "x_reference": [0, 1, 2, 3, 4],
        "reference_value": [0, 2, 4, 6, 8],
        "x_software": [-1, 0, 1.5, 3, 5],
        "software_value": [-2, 0, 3, 6, 10],
    })
    result = compare_reference_series(frame)
    assert result["summary"]["n_compared_points"] == 3
    assert result["summary"]["agreement_status"] == "UNASSESSED"
    assert result["summary"]["root_mean_square_error"] == pytest.approx(0.0)
    np.testing.assert_allclose(result["aligned_comparison"]["x"], [0, 1.5, 3])


def test_curve_comparison_uses_explicit_tolerance():
    frame = pd.DataFrame({
        "x_reference": [0, 1, 2, 3],
        "reference_value": [0, 1, 2, 3],
        "x_software": [0, 1, 2, 3],
        "software_value": [0, 1.1, 2.1, 3.1],
    })
    result = compare_reference_series(frame, absolute_tolerance=0.11)
    assert result["summary"]["agreement_status"] == "AGREES_WITHIN_CRITERION"
    assert result["summary"]["fraction_within_criterion"] == pytest.approx(1.0)


def test_curve_comparison_uses_combined_uncertainty():
    frame = pd.DataFrame({
        "x_reference": [0, 1, 2, 3],
        "reference_value": [0, 1, 2, 3],
        "x_software": [0, 1, 2, 3],
        "software_value": [0, 1.5, 2, 3],
        "reference_uncertainty": [0.1, 0.1, 0.1, 0.1],
        "software_uncertainty": [0.1, 0.1, 0.1, 0.1],
    })
    result = compare_reference_series(frame, coverage_factor=2)
    assert result["summary"]["agreement_status"] == "DISAGREES_WITH_CRITERION"
    assert result["summary"]["fraction_within_criterion"] == pytest.approx(0.75)


def test_curve_comparison_rejects_duplicate_coordinates():
    frame = pd.DataFrame({
        "x_reference": [0, 1, 1],
        "reference_value": [0, 1, 2],
        "x_software": [0, 1, 2],
        "software_value": [0, 1, 2],
    })
    with pytest.raises(ValueError, match="unique"):
        compare_reference_series(frame)


def test_curve_comparison_rejects_missing_overlap():
    frame = pd.DataFrame({
        "x_reference": [0, 1, 2],
        "reference_value": [0, 1, 2],
        "x_software": [5, 6, 7],
        "software_value": [5, 6, 7],
    })
    with pytest.raises(ValueError, match="overlap"):
        compare_reference_series(frame)

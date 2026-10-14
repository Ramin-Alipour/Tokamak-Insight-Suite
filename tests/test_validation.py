"""Tests for transparent comparison of reference and software results."""
import numpy as np
import pandas as pd
import pytest

from tis.validation import compare_reference_results


def test_reference_comparison_uses_combined_uncertainty():
    frame = pd.DataFrame({
        "parameter": ["Te"],
        "reference_value": [10.0],
        "software_value": [12.0],
        "reference_uncertainty": [1.0],
        "software_uncertainty": [1.0],
    })
    out = compare_reference_results(frame, coverage_factor=2.0)
    assert out.loc[0, "agreement_status"] == "AGREES_WITHIN_UNCERTAINTY"
    assert out.loc[0, "normalized_difference"] == pytest.approx(np.sqrt(2))


def test_reference_comparison_does_not_invent_tolerance():
    frame = pd.DataFrame({"reference_value": [10.0], "software_value": [10.01]})
    out = compare_reference_results(frame)
    assert out.loc[0, "agreement_status"] == "UNASSESSED"
    assert out.loc[0, "absolute_difference"] == pytest.approx(0.01)


def test_reference_comparison_respects_absolute_tolerance():
    frame = pd.DataFrame({"reference_value": [10.0, 10.0], "software_value": [10.1, 10.3], "absolute_tolerance": [0.2, 0.2]})
    out = compare_reference_results(frame)
    assert list(out["agreement_status"]) == ["AGREES_WITH_TOLERANCE", "DISAGREES_WITH_TOLERANCE"]


def test_reference_comparison_relative_tolerance_handles_zero_reference():
    frame = pd.DataFrame({"reference_value": [0.0], "software_value": [0.0], "relative_tolerance": [0.05]})
    out = compare_reference_results(frame)
    assert out.loc[0, "agreement_status"] == "AGREES_WITH_TOLERANCE"


def test_reference_comparison_rejects_invalid_uncertainty():
    frame = pd.DataFrame({"reference_value": [1.0], "software_value": [1.0], "reference_uncertainty": [-1.0], "software_uncertainty": [0.0]})
    with pytest.raises(ValueError, match="(?i)uncertainties"):
        compare_reference_results(frame)


def test_reference_comparison_requires_numeric_values():
    frame = pd.DataFrame({"reference_value": [np.nan], "software_value": [1.0]})
    with pytest.raises(ValueError, match="finite numeric"):
        compare_reference_results(frame)

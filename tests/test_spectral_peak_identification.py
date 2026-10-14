"""Regression tests for spectral peak detection and line-list matching."""
import numpy as np
import pandas as pd
import pytest

from tis.core import detect_spectral_peaks, match_spectral_lines


def test_peak_detection_recovers_subsample_gaussian_centers():
    x = np.linspace(400.0, 404.0, 801)
    y = 100.0 * np.exp(-0.5 * ((x - 401.23) / 0.08) ** 2)
    y += 60.0 * np.exp(-0.5 * ((x - 403.1) / 0.12) ** 2)
    result = detect_spectral_peaks(x, y, min_prominence=5.0, min_distance_coordinate=0.5)
    assert result["n_detected_peaks"] == 2
    np.testing.assert_allclose(result["peak_coordinate"], [401.23, 403.1], atol=0.005)


def test_peak_detection_rejects_duplicate_coordinates():
    with pytest.raises(ValueError, match="unique"):
        detect_spectral_peaks([1, 2, 2, 3, 4], [0, 1, 3, 1, 0])


def test_peak_matching_flags_ambiguity_and_no_match():
    refs = pd.DataFrame({
        "wavelength_nm": [401.20, 401.25, 403.10],
        "species": ["Fe I", "Mn I", "O II"],
        "transition": ["line A", "line B", "line C"],
    })
    result = match_spectral_lines([401.23, 403.11, 405.0], refs, tolerance=0.04)
    assert list(result["match_status"]) == ["AMBIGUOUS_MATCH", "MATCH", "NO_MATCH_WITHIN_TOLERANCE"]
    assert result.loc[0, "candidate_count_within_tolerance"] == 2
    assert result.loc[1, "species"] == "O II"


def test_peak_matching_rejects_invalid_reference_table_or_tolerance():
    with pytest.raises(ValueError, match="tolerance"):
        match_spectral_lines([1.0], pd.DataFrame({"wavelength_nm": [1.0]}), tolerance=0.0)
    with pytest.raises(ValueError, match="must contain"):
        match_spectral_lines([1.0], pd.DataFrame({"lambda": [1.0]}), tolerance=0.1)

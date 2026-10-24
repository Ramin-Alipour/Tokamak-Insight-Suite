"""Regression tests for Thomson profile scalar validation."""
import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import build_profile, profile_summary


@pytest.mark.parametrize("bad", [True, np.bool_(True), "1.0", np.nan, np.inf, 10**1000])
def test_profile_summary_rejects_invalid_major_radius_scalar(bad):
    profile = build_profile([0.0, 0.1], [1e19, 5e18], [100.0, 50.0])
    with pytest.raises(ValueError, match="major_radius_m"):
        profile_summary(profile, major_radius_m=bad)

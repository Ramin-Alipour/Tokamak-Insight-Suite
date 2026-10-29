"""Regression tests for the geometry-independent Thomson profile summary."""
import numpy as np

from tokamak_analysis.thomson_scattering import build_profile, profile_summary


def test_profile_summary_uses_only_measured_coordinate_interval():
    profile = build_profile([0.0, 0.1, 0.2], [1e19, 5e18, 2e18], [100.0, 50.0, 20.0])
    summary = profile_summary(profile)
    assert summary.radius_min_m == 0.0
    assert summary.radius_max_m == 0.2
    assert np.isfinite(summary.density_line_average_m3)
    assert np.isfinite(summary.temperature_line_average_ev)
    assert not hasattr(summary, "volume_average_density_m3")

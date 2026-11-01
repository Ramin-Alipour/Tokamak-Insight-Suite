"""Regression tests for the canonical measured-spectrum Thomson workflow."""

import pytest
import tokamak_analysis.thomson_scattering as scattering
import tokamak_analysis.thomson_workflow as workflow


def test_prederived_profile_only_entry_points_are_removed():
    for name in (
        "ThomsonProfile", "ThomsonSummary", "build_profile",
        "profile_summary", "interpolate_profile", "weighted_mean",
        "reconstruct_thomson_profile",
    ):
        assert not hasattr(scattering, name)
    assert not hasattr(workflow, "analyze_thomson_profile")


def test_registered_spatial_profile_workflow_is_spectrum_driven():
    assert callable(workflow.analyze_thomson_spectra)
    assert callable(workflow.reconstruct_thomson_profile_from_spectra)



def test_profile_reconstruction_rejects_unordered_channel_coordinates_without_reordering():
    with pytest.raises(ValueError, match="input order is preserved and never repaired"):
        workflow._reconstruct_inferred_parameter_profile(
            [0.2, 0.1], [2.0, 1.0], method="pchip"
        )

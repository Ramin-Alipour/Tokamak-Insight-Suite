"""Cross-check advanced Analysis-1 Dashboard choices against executable API routes."""

import pytest

from tis.ui.advanced_thomson_physics_page import MODEL_LABELS
from tokamak_analysis.cts_full_physics import (
    ADVANCED_THOMSON_PHYSICS_MODELS,
    run_advanced_thomson_physics,
)


def test_every_advanced_dashboard_choice_has_a_public_dispatch_route():
    assert set(MODEL_LABELS) <= ADVANCED_THOMSON_PHYSICS_MODELS


def test_dispatcher_error_lists_every_supported_route():
    with pytest.raises(ValueError) as exc_info:
        run_advanced_thomson_physics("not_a_real_analysis1_model")
    message = str(exc_info.value)
    for model in sorted(ADVANCED_THOMSON_PHYSICS_MODELS):
        assert model in message
    assert "no automatic model substitution" in message


def test_supported_route_set_contains_spatial_and_full_em_inverses():
    assert {
        "thomson_joint_spatial_spectra",
        "thomson_psf_profile_inverse",
        "landau_fdt_full_em_constitutive",
        "cts_tabulated_evdf_landau",
        "cts_electrostatic_measured_spectrum_inference",
    } <= ADVANCED_THOMSON_PHYSICS_MODELS

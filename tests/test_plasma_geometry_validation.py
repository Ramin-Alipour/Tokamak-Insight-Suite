"""Geometry-domain regression tests for empirical plasma scalings."""
import pytest

from tis.plasma import ipb98y2_confinement_time, shaping_surface_area


def test_surface_area_scaling_rejects_minor_radius_not_smaller_than_major_radius():
    with pytest.raises(ValueError, match="Inverse aspect ratio"):
        shaping_surface_area(r_m=1.0, a_m=1.0, kappa=1.5)


def test_ipb98_scaling_rejects_invalid_inverse_aspect_ratio():
    with pytest.raises(ValueError, match="Inverse aspect ratio"):
        ipb98y2_confinement_time(
            ip_MA=1.0, bt_T=2.0, p_loss_MW=1.0, ne_19=5.0,
            mass_amu=2.0, r_m=1.0, a_m=1.0, kappa_a=1.5,
        )


def test_ipb98_scaling_accepts_valid_tokamak_geometry():
    result = ipb98y2_confinement_time(
        ip_MA=1.0, bt_T=2.0, p_loss_MW=1.0, ne_19=5.0,
        mass_amu=2.0, r_m=1.0, a_m=0.3, kappa_a=1.5,
    )
    assert result["tau_E_IPB98y2_s"] > 0

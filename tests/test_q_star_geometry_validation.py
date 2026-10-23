import pytest

from tis.plasma import q_star_cylindrical


def test_q_star_cylindrical_rejects_minor_radius_not_smaller_than_major_radius():
    for major_radius, minor_radius in ((1.0, 1.0), (0.9, 1.0), (1.0, 1.2)):
        with pytest.raises(ValueError, match="Inverse aspect ratio"):
            q_star_cylindrical(
                ip_MA=1.0,
                bt_T=2.0,
                r_m=major_radius,
                a_m=minor_radius,
                kappa=1.5,
            )


def test_q_star_cylindrical_accepts_valid_tokamak_geometry():
    result = q_star_cylindrical(
        ip_MA=1.0,
        bt_T=2.0,
        r_m=1.5,
        a_m=0.5,
        kappa=1.0,
    )
    assert result == pytest.approx(5.0 * 0.5**2 * 2.0 * (1.0 + 1.0**2) / (2.0 * 1.5 * 1.0))

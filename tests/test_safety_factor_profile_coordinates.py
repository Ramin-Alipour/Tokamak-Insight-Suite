import numpy as np
import pytest

from tokamak_analysis.safety_factor import estimate_safety_factor


def test_q95_is_not_inferred_from_profile_sample_index():
    result = estimate_safety_factor(
        1.5, 0.5, 2.0, 0.1, q_profile=[1.0, 1.5, 2.5]
    )
    assert "q95_provided" not in result
    assert "q95_note" in result


def test_q95_uses_explicit_normalized_radius_coordinate():
    result = estimate_safety_factor(
        1.5,
        0.5,
        2.0,
        0.1,
        q_profile=[1.0, 2.0, 4.0],
        q_profile_rho=[0.0, 0.5, 1.0],
    )
    assert result["q95_provided"] == pytest.approx(3.8)
    assert result["q95_interpolation_method"].startswith("linear interpolation")


@pytest.mark.parametrize(
    "rho",
    [
        [0.0, 0.8],
        [0.0, 0.95, 0.9],
        [0.0, 0.5, 1.1],
        [0.0, np.nan, 1.0],
        [0.0, 0.5],
    ],
)
def test_q95_rejects_invalid_or_noncovering_radius_coordinates(rho):
    with pytest.raises(ValueError, match="q_profile_rho"):
        estimate_safety_factor(
            1.5, 0.5, 2.0, 0.1, q_profile=[1.0, 2.0, 4.0], q_profile_rho=rho
        )


def test_radius_coordinate_requires_profile():
    with pytest.raises(ValueError, match="requires q_profile"):
        estimate_safety_factor(1.5, 0.5, 2.0, 0.1, q_profile_rho=[0.0, 1.0])

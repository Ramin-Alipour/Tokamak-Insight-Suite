import numpy as np
import pytest

from tokamak_analysis.safety_factor import estimate_safety_factor


def test_q95_is_not_inferred_from_profile_sample_index():
    result = estimate_safety_factor(
        1.5, 0.5, 2.0, 0.1, q_profile=[1.0, 1.5, 2.5]
    )
    assert "q95_provided" not in result
    assert "q95_note" in result


def test_conventional_q95_uses_normalized_poloidal_flux_coordinate():
    result = estimate_safety_factor(
        1.5,
        0.5,
        2.0,
        0.1,
        q_profile=[1.0, 2.0, 4.0],
        q_profile_psi_n=[0.0, 0.5, 1.0],
    )
    assert result["q95_provided"] == pytest.approx(3.8)
    assert result["q95_coordinate"] == "normalized poloidal flux psi_N=0.95"
    assert result["q95_interpolation_method"].startswith("linear interpolation")


def test_rho_coordinate_is_not_mislabeled_as_conventional_q95():
    result = estimate_safety_factor(
        1.5, 0.5, 2.0, 0.1,
        q_profile=[1.0, 2.0, 4.0],
        q_profile_rho=[0.0, 0.5, 1.0],
    )
    assert result["q_at_rho_0_95"] == pytest.approx(3.8)
    assert "q95_provided" not in result
    assert "not necessarily conventional q95" in result["profile_coordinate_note"]


@pytest.mark.parametrize("coordinate_name", ["q_profile_rho", "q_profile_psi_n"])
@pytest.mark.parametrize(
    "coordinate",
    [
        [0.0, 0.8],
        [0.0, 0.95, 0.9],
        [0.0, 0.5, 1.1],
        [0.0, np.nan, 1.0],
        [0.0, 0.5],
    ],
)
def test_rejects_invalid_or_noncovering_coordinates(coordinate_name, coordinate):
    with pytest.raises(ValueError, match=coordinate_name):
        estimate_safety_factor(
            1.5, 0.5, 2.0, 0.1,
            q_profile=[1.0, 2.0, 4.0],
            **{coordinate_name: coordinate},
        )


def test_profile_coordinate_requires_profile():
    with pytest.raises(ValueError, match="requires q_profile"):
        estimate_safety_factor(1.5, 0.5, 2.0, 0.1, q_profile_psi_n=[0.0, 1.0])


def test_profile_coordinates_are_mutually_exclusive():
    with pytest.raises(ValueError, match="only one profile coordinate"):
        estimate_safety_factor(
            1.5, 0.5, 2.0, 0.1,
            q_profile=[1.0, 2.0],
            q_profile_rho=[0.0, 1.0],
            q_profile_psi_n=[0.0, 1.0],
        )

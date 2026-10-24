import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import build_profile


@pytest.mark.parametrize(
    ("radius", "density", "temperature"),
    [
        ([False, True], [1.0, 2.0], [3.0, 4.0]),
        ([0.0, 1.0], ["1", "2"], [3.0, 4.0]),
        ([0.0, 1.0], [1.0, 2.0], [True, False]),
        ([[0.0], [1.0]], [1.0, 2.0], [3.0, 4.0]),
    ],
)
def test_thomson_profile_rejects_non_numeric_or_non_1d_measurements(
    radius, density, temperature
):
    with pytest.raises(ValueError, match="one-dimensional|real numeric"):
        build_profile(radius, density, temperature)


@pytest.mark.parametrize(
    "uncertainty",
    [
        [True, False],
        ["0.1", "0.2"],
        [[0.1], [0.2]],
    ],
)
def test_thomson_profile_rejects_invalid_uncertainty_types_and_shapes(uncertainty):
    with pytest.raises(ValueError, match="one-dimensional|real numeric|same length"):
        build_profile([0.0, 1.0], [1e19, 2e19], [10.0, 20.0],
                      density_uncertainty_m3=uncertainty)


def test_thomson_profile_accepts_valid_numeric_measurements_and_uncertainties():
    profile = build_profile(
        [1.0, 0.0], [2e19, 1e19], [20.0, 10.0],
        density_uncertainty_m3=[2e18, 1e18],
    )
    np.testing.assert_allclose(profile.radius_m, [0.0, 1.0])
    np.testing.assert_allclose(profile.density_m3, [1e19, 2e19])
    np.testing.assert_allclose(profile.density_uncertainty_m3, [1e18, 2e18])

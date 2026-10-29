import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import build_profile, interpolate_profile


def test_interpolation_propagates_independent_uncertainties_in_quadrature():
    profile = build_profile(
        radius_m=[0.0, 1.0],
        density_m3=[1.0e19, 2.0e19],
        temperature_ev=[10.0, 20.0],
        density_uncertainty_m3=[3.0e17, 4.0e17],
        temperature_uncertainty_ev=[3.0, 4.0],
    )

    result = interpolate_profile(profile, [0.0, 0.5, 1.0])

    # At the midpoint, independent endpoint errors combine as
    # sqrt((0.5*sigma_0)^2 + (0.5*sigma_1)^2), not linear interpolation.
    np.testing.assert_allclose(
        result["density_uncertainty_m3"],
        [3.0e17, 2.5e17, 4.0e17],
    )
    np.testing.assert_allclose(
        result["temperature_uncertainty_ev"],
        [3.0, 2.5, 4.0],
    )


def test_interpolation_uncertainty_follows_sorted_profile_order():
    profile = build_profile(
        radius_m=[1.0, 0.0],
        density_m3=[2.0e19, 1.0e19],
        temperature_ev=[20.0, 10.0],
        density_uncertainty_m3=[4.0e17, 3.0e17],
    )

    result = interpolate_profile(profile, [0.5])

    assert result["density_uncertainty_m3"][0] == pytest.approx(2.5e17)

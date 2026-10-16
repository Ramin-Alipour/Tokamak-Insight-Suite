import numpy as np
import pytest

from tokamak_analysis.thomson_workflow import analyze_thomson_profile


def test_thomson_workflow_sorts_and_calculates_linear_gradients():
    r = np.array([0.3, 0.0, 0.2, 0.1])
    ne = 2.0e19 * (1.0 - r)
    te = 100.0 * (1.0 - 0.5 * r)
    out = analyze_thomson_profile(r, ne, te)
    np.testing.assert_allclose(out["radius_m"], np.sort(r))
    np.testing.assert_allclose(out["density_gradient_m-4"], -2.0e19, rtol=1e-12)
    np.testing.assert_allclose(out["temperature_gradient_eV_m-1"], -50.0, rtol=1e-12)
    assert out["summary"].point_count == 4
    assert np.all(np.isfinite(out["density_scale_length_m"]))


def test_thomson_workflow_rejects_extrapolation_not_applicable():
    with pytest.raises(ValueError, match="equal lengths"):
        analyze_thomson_profile([0, 1, 2], [1, 2], [10, 20, 30])


def test_thomson_workflow_preserves_sorted_uncertainties():
    out = analyze_thomson_profile(
        [1, 0], [2e19, 1e19], [20, 10],
        density_uncertainty_m3=[2e18, 1e18],
        temperature_uncertainty_ev=[2, 1],
    )
    np.testing.assert_allclose(out["density_uncertainty_m3"], [1e18, 2e18])
    np.testing.assert_allclose(out["temperature_uncertainty_eV"], [1, 2])


def test_thomson_differential_cross_section_limits_and_symmetry():
    from tokamak_analysis.thomson_scattering import (
        ELECTRON_CLASSICAL_RADIUS_M,
        thomson_differential_cross_section_m2_sr,
    )

    r_e = ELECTRON_CLASSICAL_RADIUS_M
    values = thomson_differential_cross_section_m2_sr([0.0, np.pi / 2, np.pi])
    np.testing.assert_allclose(values, [r_e**2, r_e**2 / 2, r_e**2], rtol=1e-14)
    np.testing.assert_allclose(
        thomson_differential_cross_section_m2_sr([0.2, np.pi - 0.2])[0],
        thomson_differential_cross_section_m2_sr([0.2, np.pi - 0.2])[1],
        rtol=1e-14,
    )


def test_thomson_density_power_relation_round_trip():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    ne = np.array([1.0e19, 2.0e19, 5.0e19])
    incident = np.full(3, 2.0)
    length = np.full(3, 0.15)
    angle = np.array([0.0, np.pi / 2, np.pi])
    solid_angle = np.full(3, 1.0e-3)
    efficiency = np.full(3, 0.7)
    from tokamak_analysis.thomson_scattering import thomson_differential_cross_section_m2_sr
    detected = incident * ne * length * thomson_differential_cross_section_m2_sr(angle) * solid_angle * efficiency
    out = density_from_thomson_scattered_power(
        detected, incident, length, angle, solid_angle, optical_efficiency=efficiency
    )
    np.testing.assert_allclose(out["electron_density_m3"], ne, rtol=1e-14)
    np.testing.assert_allclose(out["model_scattered_power_W"], detected, rtol=1e-14)


def test_thomson_density_rejects_invalid_scattering_angle():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    with pytest.raises(ValueError, match="angle"):
        density_from_thomson_scattered_power(1.0, 2.0, 0.1, 4.0, 1.0e-3)

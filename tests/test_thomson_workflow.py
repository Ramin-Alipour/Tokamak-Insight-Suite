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
        major_radius_m=1.0,
    )
    np.testing.assert_allclose(out["density_uncertainty_m3"], [1e18, 2e18])
    np.testing.assert_allclose(out["temperature_uncertainty_eV"], [1, 2])
    assert np.all(np.isfinite(out["density_gradient_uncertainty_m-4"]))
    assert np.all(np.isfinite(out["temperature_gradient_uncertainty_eV_m-1"]))
    assert np.all(np.isfinite(out["density_scale_length_uncertainty_m"]))
    assert out["summary"].density_line_average_uncertainty_m3 is not None
    assert out["summary"].temperature_line_average_uncertainty_ev is not None
    assert out["summary"].density_volume_average_uncertainty_m3 is not None
    assert out["summary"].temperature_volume_average_uncertainty_ev is not None


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


def test_thomson_volume_average_rejects_signed_radius_coordinate():
    from tokamak_analysis.thomson_scattering import build_profile, profile_summary

    profile = build_profile([-0.2, 0.0, 0.2], [1e19, 2e19, 1e19], [20.0, 40.0, 20.0])
    with pytest.raises(ValueError, match="non-negative minor-radius"):
        profile_summary(profile, major_radius_m=1.0)



def test_thomson_power_density_propagates_independent_input_uncertainties():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    out = density_from_thomson_scattered_power(
        scattered_power_W=1.0e-14,
        incident_power_W=2.0,
        interaction_length_m=0.15,
        scattering_angle_rad=np.pi / 2,
        collection_solid_angle_sr=1.0e-3,
        optical_efficiency=0.7,
        scattered_power_uncertainty_W=0.02e-14,
        incident_power_uncertainty_W=0.02,
        interaction_length_uncertainty_m=0.001,
        scattering_angle_uncertainty_rad=0.01,
        collection_solid_angle_uncertainty_sr=0.02e-3,
        optical_efficiency_uncertainty=0.01,
    )
    assert out["electron_density_m3"] > 0
    assert 0 < out["electron_density_uncertainty_m3"] < out["electron_density_m3"]
    assert "first-order propagation" in out["uncertainty_model"].lower()


def test_thomson_power_density_rejects_negative_input_uncertainty():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    with pytest.raises(ValueError, match="finite and non-negative"):
        density_from_thomson_scattered_power(
            1.0e-9, 2.0, 0.15, np.pi / 2, 1.0e-3,
            scattered_power_uncertainty_W=-1.0e-12,
        )

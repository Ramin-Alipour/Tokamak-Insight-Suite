"""Integration tests for the exact relativistic optical Thomson flow model."""

import math

import pytest

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def test_exact_relativistic_thomson_flow_is_exposed_through_physics_api():
    laser_nm = 532.0
    angle = 1.1
    beta = -0.18
    sine = math.sin(angle / 2.0)
    scattered_nm = laser_nm * (1.0 - beta * sine) / (1.0 + beta * sine)

    result = run_advanced_thomson_physics(
        "thomson_relativistic_exact_flow_projection",
        fitted_center_nm=scattered_nm,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
    )

    assert result["electron_velocity_projection_m_s"] == pytest.approx(
        beta * 299792458.0, rel=1e-13
    )
    assert result["public_physics_entrypoint"] == (
        "thomson_relativistic_exact_flow_projection"
    )
    assert result["automatic_model_substitution"] is False
    assert result["identifiability"] == "one_projected_velocity_only"


def test_exact_relativistic_thomson_flow_rejects_unknown_uncertainty_contract():
    with pytest.raises(ValueError, match="all three independent uncertainties"):
        run_advanced_thomson_physics(
            "thomson_relativistic_exact_flow_projection",
            fitted_center_nm=532.0,
            laser_wavelength_nm=532.0,
            scattering_angle_rad=math.pi / 2.0,
            sigma_fitted_center_nm=1e-4,
        )


def test_exact_relativistic_thomson_flow_propagates_full_correlated_covariance():
    laser_nm, angle, beta = 532.0, 1.1, 0.025
    scattered_nm = laser_nm * (1.0 - beta * math.sin(angle / 2.0)) / (1.0 + beta * math.sin(angle / 2.0))
    covariance = [
        [4.0e-10, 1.0e-10, 2.0e-12],
        [1.0e-10, 9.0e-10, -1.0e-12],
        [2.0e-12, -1.0e-12, 4.0e-8],
    ]
    result = run_advanced_thomson_physics(
        "thomson_relativistic_exact_flow_projection",
        fitted_center_nm=scattered_nm, laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle, input_covariance_nm2_rad2=covariance,
    )
    j = result["velocity_jacobian_m_s_per_input_unit"]
    expected = math.sqrt(sum(j[i] * covariance[i][k] * j[k]
                              for i in range(3) for k in range(3)))
    assert result["uncertainty_electron_velocity_projection_m_s"] == pytest.approx(expected)
    assert "full_joint_covariance" in result["uncertainty_model"]


def test_exact_relativistic_thomson_flow_rejects_non_psd_joint_covariance():
    with pytest.raises(ValueError, match="positive semidefinite"):
        run_advanced_thomson_physics(
            "thomson_relativistic_exact_flow_projection",
            fitted_center_nm=532.0, laser_wavelength_nm=532.0,
            scattering_angle_rad=math.pi / 2.0,
            input_covariance_nm2_rad2=[[1.0, 2.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        )


def test_exact_relativistic_thomson_flow_rejects_mixed_covariance_and_sigmas():
    with pytest.raises(ValueError, match="not both"):
        run_advanced_thomson_physics(
            "thomson_relativistic_exact_flow_projection",
            fitted_center_nm=532.0, laser_wavelength_nm=532.0,
            scattering_angle_rad=math.pi / 2.0,
            sigma_fitted_center_nm=1e-4, sigma_laser_wavelength_nm=1e-4,
            sigma_scattering_angle_rad=1e-4,
            input_covariance_nm2_rad2=[[1e-8, 0.0, 0.0], [0.0, 1e-8, 0.0], [0.0, 0.0, 1e-8]],
        )

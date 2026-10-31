"""Regression tests for Thomson profiles reconstructed from measured spectra."""

import numpy as np
import pytest

def test_collective_kappa_profile_uses_external_monte_carlo_covariance(monkeypatch):
    import tokamak_analysis.thomson_workflow as workflow

    radii = np.array([0.1, 0.2, 0.3])
    order = []
    for i in range(3):
        order.extend([
            f"channel_{i}_electron_density_m3",
            f"channel_{i}_electron_temperature_eV",
            f"channel_{i}_kappa_index",
            f"channel_{i}_electron_velocity_projection_m_s",
            f"channel_{i}_ion_0_velocity_projection_m_s",
        ])
    external_covariance = np.eye(15)
    kappa_indices = [2, 7, 12]
    kappa_covariance = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.015],
        [0.005, 0.015, 0.16],
    ])
    external_covariance[np.ix_(kappa_indices, kappa_indices)] = kappa_covariance
    density_indices = [0, 5, 10]
    density_covariance = external_covariance[np.ix_(density_indices, density_indices)]
    channels = []
    for i, kappa in enumerate([3.0, 4.0, 5.0]):
        channels.append({
            "electron_density_m3": 4.0e18 + i * 1.0e17,
            "electron_temperature_eV": 100.0 + i * 10.0,
            "electron_velocity_projection_m_s": float(i),
            "electron_density_uncertainty_m3": 1.0e17,
            "electron_temperature_uncertainty_eV": np.sqrt(5.0),
            "electron_velocity_projection_uncertainty_m_s": 1.0,
            "kappa_index": kappa,
            "kappa_index_uncertainty": np.sqrt(2.0) * (0.2 + i * 0.1),
            "fit_covariance": np.diag([1.0e34, 4.0, (0.2 + i * 0.1) ** 2]),
            "scattering_parameter_alpha": 1.5 + i * 0.1,
            "ion_velocity_projection_m_s": [0.0],
            "ion_velocity_projection_uncertainty_m_s": [None],
            "spectral_distribution": {},
        })
    inference_result = {
        "model": "collective_kappa",
        "radius_m": radii,
        "channels": channels,
        "electron_temperature_profile_eV": np.array([100.0, 110.0, 120.0]),
        "electron_temperature_uncertainty_profile_eV": np.full(3, 2.0),
        "electron_density_profile_m3": np.array([4.0e18, 4.1e18, 4.2e18]),
        "electron_density_uncertainty_profile_m3": np.full(3, 1.0e17),
        "electron_density_joint_covariance_m6": None,
        "electron_density_external_parameter_covariance_m6": density_covariance,
        "electron_velocity_projection_profile_m_s": np.array([0.0, 1.0, 2.0]),
        "electron_velocity_projection_uncertainty_profile_m_s": np.ones(3),
        "collective_external_parameter_covariance": external_covariance,
        "collective_external_parameter_covariance_order": order,
        "absolute_scattered_power_available": False,
    }
    monkeypatch.setattr(workflow, "analyze_thomson_spectra", lambda *args, **kwargs: inference_result)
    monkeypatch.setattr(
        workflow, "_reconstruct_inferred_parameter_profile",
        lambda radius, values, **kwargs: {
            "status": "captured",
            "covariance": kwargs.get("covariance"),
        },
    )

    result = workflow.reconstruct_thomson_profile_from_spectra(
        profile_reconstruction_method="pchip", profile_reconstruction_points=20,
    )
    profiles = result["spatial_profile_reconstruction"]["profiles"]
    actual = profiles["electron_kappa_index"]["covariance"]
    expected_kappa_covariance = kappa_covariance + np.diag([0.04, 0.09, 0.16])
    np.testing.assert_allclose(actual, expected_kappa_covariance)

    expected_temperature_covariance = np.eye(3) + np.diag([4.0, 4.0, 4.0])
    np.testing.assert_allclose(
        profiles["electron_temperature_eV"]["covariance"],
        expected_temperature_covariance,
    )
    expected_density_covariance = density_covariance + np.diag([1.0e34] * 3)
    np.testing.assert_allclose(
        profiles["electron_density_m3"]["covariance"],
        expected_density_covariance,
    )


def test_noncollective_kappa_profile_uses_joint_gls_covariance(monkeypatch):
    import tokamak_analysis.thomson_workflow as workflow

    radii = np.array([0.1, 0.2, 0.3])
    kappa_covariance = np.array([
        [0.04, 0.01, 0.005],
        [0.01, 0.09, 0.015],
        [0.005, 0.015, 0.16],
    ])
    channels = []
    for i, kappa in enumerate([3.0, 4.0, 5.0]):
        channels.append({
            "electron_density_m3": 4.0e18 + i * 1.0e17,
            "electron_temperature_eV": 100.0 + i * 10.0,
            "electron_velocity_projection_m_s": float(i),
            "electron_density_uncertainty_m3": 1.0e17,
            "electron_temperature_uncertainty_eV": 2.0,
            "electron_velocity_projection_uncertainty_m_s": 1.0,
            "kappa_index": kappa,
            "kappa_index_uncertainty": 0.2 + i * 0.1,
            "fit_covariance": np.diag([1.0, 1.0, 1.0, (0.2 + i * 0.1) ** 2, 1.0, 1.0]),
            "spectral_distribution": {},
        })
    inference_result = {
        "model": "non_collective_kappa",
        "radius_m": radii,
        "channels": channels,
        "electron_temperature_profile_eV": np.array([100.0, 110.0, 120.0]),
        "electron_temperature_uncertainty_profile_eV": np.full(3, 2.0),
        "electron_density_profile_m3": np.array([4.0e18, 4.1e18, 4.2e18]),
        "electron_density_uncertainty_profile_m3": np.full(3, 1.0e17),
        "electron_density_joint_covariance_m6": None,
        "electron_density_external_parameter_covariance_m6": None,
        "electron_velocity_projection_profile_m_s": np.array([0.0, 1.0, 2.0]),
        "electron_velocity_projection_uncertainty_profile_m_s": np.ones(3),
        "joint_inference": {
            "total_kappa_parameter_covariance": kappa_covariance,
        },
        "absolute_scattered_power_available": False,
    }
    monkeypatch.setattr(workflow, "analyze_thomson_spectra", lambda *args, **kwargs: inference_result)
    monkeypatch.setattr(
        workflow, "_reconstruct_inferred_parameter_profile",
        lambda radius, values, **kwargs: {
            "status": "captured",
            "covariance": kwargs.get("covariance"),
        },
    )

    result = workflow.reconstruct_thomson_profile_from_spectra(
        profile_reconstruction_method="pchip", profile_reconstruction_points=20,
    )
    actual = result["spatial_profile_reconstruction"]["profiles"]["electron_kappa_index"]["covariance"]
    np.testing.assert_allclose(actual, kappa_covariance)

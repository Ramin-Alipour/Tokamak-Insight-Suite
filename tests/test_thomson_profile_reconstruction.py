"""Regression tests for Thomson profiles reconstructed from measured spectra."""

import numpy as np
import pytest

def test_collective_kappa_profile_is_rejected_outside_optical_analysis_1_scope():
    from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra

    with pytest.raises(ValueError, match="only the three registered"):
        reconstruct_thomson_profile_from_spectra(
            wavelength_nm=np.linspace(531.0, 533.0, 21),
            spectra=np.ones((2, 21)),
            radius_m=[0.1, 0.2],
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2,
            instrument_response=np.eye(21),
            model="collective_kappa",
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
        model="non_collective_kappa",
        profile_reconstruction_method="pchip", profile_reconstruction_points=20,
    )
    actual = result["spatial_profile_reconstruction"]["profiles"]["electron_kappa_index"]["covariance"]
    np.testing.assert_allclose(actual, kappa_covariance)


def test_positive_thomson_profiles_remain_positive_under_gcv_smoothing_spline():
    from tokamak_analysis.thomson_workflow import _reconstruct_inferred_parameter_profile

    radius = np.linspace(0.0, 0.4, 7)
    temperature = np.array([8.0, 180.0, 12.0, 240.0, 9.0, 160.0, 11.0])
    result = _reconstruct_inferred_parameter_profile(
        radius,
        temperature,
        positive=True,
        method="smoothing_spline",
        points=301,
    )
    profile = result["methods"]["smoothing_spline"]
    assert profile["status"] == "completed"
    assert np.all(np.isfinite(profile["values"]))
    assert np.all(profile["values"] > 0.0)
    assert np.all(np.isfinite(profile["gradient_per_m"]))


def test_full_cross_channel_covariance_is_used_in_gls_smoothing_objective():
    from tokamak_analysis.thomson_workflow import _reconstruct_inferred_parameter_profile

    radius = np.linspace(0.0, 0.6, 7)
    values = np.array([12.0, 18.0, 31.0, 44.0, 48.0, 43.0, 39.0])
    sigma = np.array([0.8, 1.0, 1.1, 1.2, 1.1, 1.0, 0.9])
    correlation = np.fromfunction(
        lambda i, j: 0.35 ** np.abs(i - j), (radius.size, radius.size)
    )
    covariance = correlation * np.outer(sigma, sigma)
    result = _reconstruct_inferred_parameter_profile(
        radius, values, covariance=covariance, positive=True,
        method="smoothing_spline", points=101,
    )
    profile = result["methods"]["smoothing_spline"]
    assert profile["status"] == "completed"
    assert np.all(np.isfinite(profile["values"]))
    assert np.all(profile["values"] > 0.0)
    diagnostics = profile["smoothing_diagnostics"]
    assert diagnostics["selection_method"] == "generalized_cross_validation_with_full_GLS_covariance"
    assert diagnostics["covariance_model"] == "researcher_supplied_full_cross_channel_covariance"
    assert diagnostics["smoothing_parameter"] > 0.0
    assert diagnostics["covariance_regularization"] == "none; non-positive-definite covariance is rejected"
    assert np.isfinite(diagnostics["system_condition_number"])
    assert np.isfinite(diagnostics["effective_degrees_of_freedom"])


def test_profile_reconstruction_propagates_supplied_coordinate_covariance():
    from tokamak_analysis.thomson_workflow import _reconstruct_inferred_parameter_profile

    radius = np.linspace(0.0, 0.5, 7)
    values = np.array([10.0, 18.0, 27.0, 34.0, 31.0, 22.0, 12.0])
    coordinate_sigma = np.full(radius.size, 2.0e-3)
    correlation = np.fromfunction(
        lambda i, j: 0.4 ** np.abs(i - j), (radius.size, radius.size)
    )
    coordinate_covariance = correlation * np.outer(coordinate_sigma, coordinate_sigma)

    result = _reconstruct_inferred_parameter_profile(
        radius, values, coordinate_covariance=coordinate_covariance,
        positive=True, method="pchip", points=101, mc_samples=100,
        random_seed=91,
    )
    profile = result["methods"]["pchip"]
    assert result["coordinate_uncertainty_model"] == (
        "researcher_supplied_full_coordinate_covariance_independent_of_profile_values"
    )
    assert profile["uncertainty_available"] is True
    assert profile["uncertainty_mc_successful_samples"] >= 80
    assert np.all(np.isfinite(profile["standard_uncertainty"]))
    assert np.any(profile["standard_uncertainty"] > 0.0)


def test_profile_reconstruction_rejects_invalid_coordinate_covariance():
    from tokamak_analysis.thomson_workflow import _reconstruct_inferred_parameter_profile

    radius = np.linspace(0.0, 0.5, 5)
    values = np.linspace(10.0, 20.0, 5)
    with pytest.raises(ValueError, match="Coordinate covariance"):
        _reconstruct_inferred_parameter_profile(
            radius, values, coordinate_covariance=np.ones((4, 4)),
            method="pchip",
        )


def test_joint_spatial_spectral_rejects_unpropagated_coordinate_covariance():
    from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra

    with pytest.raises(ValueError, match="not supported by the raw joint"):
        reconstruct_thomson_profile_from_spectra(
            joint_spatial_spectral=True,
            spatial_mixing_matrix=np.eye(2),
            profile_coordinate_covariance_m2=np.eye(2) * 1.0e-6,
        )

import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import reconstruct_thomson_profile


def test_smoothing_spline_reconstructs_spatial_thomson_profiles():
    radius = np.linspace(0.0, 0.5, 11)
    density = 4.0e19 * (1.0 - 0.7 * radius**2)
    temperature = 1200.0 * (1.0 - radius**2) + 50.0
    result = reconstruct_thomson_profile(
        radius, density, temperature, method="smoothing_spline", smoothing_factor=0.0, points=101
    )
    assert result["radius_m"].shape == (101,)
    assert result["density_m3"].shape == (101,)
    assert result["temperature_eV"].shape == (101,)
    assert result["density_gradient_m-4"].shape == (101,)
    assert result["temperature_gradient_eV_m-1"].shape == (101,)
    assert result["radius_m"][0] == pytest.approx(radius[0])
    assert result["radius_m"][-1] == pytest.approx(radius[-1])
    assert result["density_m3"][0] == pytest.approx(density[0], rel=1e-8)
    assert result["temperature_eV"][-1] == pytest.approx(temperature[-1], rel=1e-8)
    assert result["input_point_count"] == len(radius)
    assert result["reconstruction_point_count"] == 101


def test_pchip_profile_reconstruction_is_shape_preserving_and_sorts_samples():
    radius = np.array([0.4, 0.0, 0.3, 0.1, 0.2])
    density = 1.0e19 * (1.0 - radius)
    temperature = 100.0 * (1.0 - 0.5 * radius)
    result = reconstruct_thomson_profile(radius, density, temperature, method="pchip", points=101)
    assert np.all(np.diff(result["radius_m"]) > 0)
    assert np.all(np.diff(result["density_m3"]) <= 1e5)
    assert np.all(np.diff(result["temperature_eV"]) <= 1e-10)
    assert result["radial_domain_m"] == pytest.approx((0.0, 0.4))


def test_reconstruction_uses_positive_measurement_uncertainties_for_weighting():
    radius = np.linspace(0.0, 0.4, 8)
    density = 2e19 * (1.0 - radius) + np.array([0, 1, -1, 0.5, -0.5, 1, -1, 0]) * 1e17
    temperature = 500.0 * (1.0 - radius) + 10.0
    result = reconstruct_thomson_profile(
        radius, density, temperature,
        density_uncertainty_m3=np.full(radius.size, 2e17),
        temperature_uncertainty_ev=np.full(radius.size, 10.0),
        smoothing_factor=1.0,
    )
    assert np.all(np.isfinite(result["density_m3"]))
    assert np.all(np.isfinite(result["temperature_eV"]))
    # The public metadata contract is channel-specific and machine-readable.
    # Do not depend on the former prose-only "inverse-sigma fit weights" label.
    assert result["uncertainty_model"]["density_channel"] == "inverse_sigma_weighted"
    assert result["uncertainty_model"]["temperature_channel"] == "inverse_sigma_weighted"
    assert result["uncertainty_model"]["fit_confidence_intervals_estimated"] is True


@pytest.mark.parametrize("kwargs", [
    {"method": "unknown"},
    {"smoothing_factor": -1.0},
    {"points": 10},
])
def test_reconstruction_rejects_invalid_options(kwargs):
    with pytest.raises(ValueError):
        reconstruct_thomson_profile([0, 1, 2], [1, 2, 1], [10, 20, 10], **kwargs)


def test_weighted_spline_rejects_zero_uncertainty():
    with pytest.raises(ValueError, match="strictly positive"):
        reconstruct_thomson_profile(
            [0, 1, 2], [1, 2, 1], [10, 20, 10],
            density_uncertainty_m3=[0, 1, 1],
        )



@pytest.mark.parametrize("method", ["smoothing_spline", "pchip"])
def test_reconstruction_independently_recovers_linear_profiles_on_nonuniform_input_grid(method):
    radius = np.array([0.0, 0.1, 0.45, 0.7, 1.0])
    density = 3.0e19 - 1.0e19 * radius
    temperature = 100.0 - 20.0 * radius
    result = reconstruct_thomson_profile(radius, density, temperature,
        method=method, smoothing_factor=0.0, points=101)
    grid = result["reconstructed_radius_m"]
    np.testing.assert_allclose(result["reconstructed_density_m3"], 3.0e19 - 1.0e19 * grid, rtol=2e-12, atol=1e5)
    np.testing.assert_allclose(result["reconstructed_temperature_eV"], 100.0 - 20.0 * grid, rtol=2e-12, atol=1e-10)
    np.testing.assert_allclose(result["reconstruction_gradients"]["density_gradient_m-4"], -1.0e19, rtol=2e-12, atol=1e5)
    np.testing.assert_allclose(result["reconstruction_gradients"]["temperature_gradient_eV_m-1"], -20.0, rtol=2e-12, atol=1e-10)
    assert result["uncertainty_model"]["reconstructed_profile_uncertainty_available"] is False


def test_reconstruction_metadata_distinguishes_fit_weights_from_output_uncertainty():
    result = reconstruct_thomson_profile([0.0, 0.25, 0.5, 0.75, 1.0],
        [1e19, 1.2e19, 1.4e19, 1.2e19, 1e19], [50.0, 70.0, 90.0, 70.0, 50.0],
        density_uncertainty_m3=[1e17] * 5, temperature_uncertainty_ev=[2.0] * 5,
        method="smoothing_spline", smoothing_factor=0.5, points=31)
    model = result["uncertainty_model"]
    assert model["fit_weights_are_not_output_uncertainties"] is True
    assert model["reconstructed_profile_uncertainty_available"] is True
    assert model["fit_confidence_intervals_estimated"] is True


def test_line_average_compatibility_aliases_warn_and_return_radial_average():
    from tokamak_analysis.thomson_scattering import profile_summary

    profile = profile_summary(
        np.array([0.0, 1.0, 2.0]),
        np.array([1.0, 2.0, 3.0]),
        np.array([10.0, 20.0, 30.0]),
    )
    with pytest.warns(DeprecationWarning, match="use density_radial_average_m3"):
        assert profile.density_line_average_m3 == profile.density_radial_average_m3
    with pytest.warns(DeprecationWarning, match="use temperature_radial_average_ev"):
        assert profile.temperature_line_average_ev == profile.temperature_radial_average_ev


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
            "electron_temperature_uncertainty_eV": 2.0,
            "electron_velocity_projection_uncertainty_m_s": 1.0,
            "kappa_index": kappa,
            "kappa_index_uncertainty": 0.2 + i * 0.1,
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
    actual = result["spatial_profile_reconstruction"]["profiles"]["electron_kappa_index"]["covariance"]
    np.testing.assert_allclose(actual, kappa_covariance)


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

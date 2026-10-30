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
    assert result["uncertainty_model"]["fit_confidence_intervals_estimated"] is False


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

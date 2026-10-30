import numpy as np
import pytest

from tokamak_analysis.thomson_workflow import analyze_thomson_profile
from tokamak_analysis.thomson_scattering import build_profile, interpolate_profile


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


def test_thomson_workflow_reports_minimum_and_maximum_values_and_locations():
    out = analyze_thomson_profile(
        [0.2, 0.0, 0.3, 0.1],
        [5.0e19, 2.0e19, 8.0e19, 3.0e19],
        [50.0, 20.0, 80.0, 30.0],
    )
    summary = out["summary"]
    assert summary.density_min_m3 == pytest.approx(2.0e19)
    assert summary.density_min_radius_m == pytest.approx(0.0)
    assert summary.density_peak_m3 == pytest.approx(8.0e19)
    assert summary.density_peak_radius_m == pytest.approx(0.3)
    assert summary.temperature_min_ev == pytest.approx(20.0)
    assert summary.temperature_min_radius_m == pytest.approx(0.0)
    assert summary.temperature_peak_ev == pytest.approx(80.0)
    assert summary.temperature_peak_radius_m == pytest.approx(0.3)


def test_thomson_workflow_rejects_mismatched_profile_lengths():
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
    assert np.all(np.isfinite(out["density_gradient_uncertainty_m-4"]))
    assert np.all(np.isfinite(out["temperature_gradient_uncertainty_eV_m-1"]))
    assert np.all(np.isfinite(out["density_scale_length_uncertainty_m"]))
    assert np.all(np.isfinite(out["density_normalized_gradient_uncertainty_m-1"]))
    assert np.all(np.isfinite(out["temperature_normalized_gradient_uncertainty_m-1"]))
    assert out["summary"].density_line_average_uncertainty_m3 is not None
    assert out["summary"].temperature_line_average_uncertainty_ev is not None


def test_scale_length_threshold_is_dimensionally_consistent_for_small_values():
    # Small but valid calibrated values must not be discarded merely because
    # their magnitudes are below one in SI units.
    out = analyze_thomson_profile(
        [0.0, 1.0, 2.0],
        [3.0e-20, 2.0e-20, 1.0e-20],
        [3.0e-12, 2.0e-12, 1.0e-12],
    )
    np.testing.assert_allclose(
        out["density_scale_length_m"],
        [3.0, 2.0, 1.0],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        out["temperature_scale_length_m"],
        [3.0, 2.0, 1.0],
        rtol=1e-12,
    )


def test_scale_length_is_undefined_for_constant_profiles():
    out = analyze_thomson_profile(
        [0.0, 1.0, 2.0],
        [2.0e19, 2.0e19, 2.0e19],
        [100.0, 100.0, 100.0],
    )
    assert np.all(np.isnan(out["density_scale_length_m"]))
    assert np.all(np.isnan(out["temperature_scale_length_m"]))

def test_scale_length_uncertainty_uses_gradient_units_for_small_profiles():
    out = analyze_thomson_profile(
        [0.0, 1.0, 2.0],
        [3.0e-20, 2.0e-20, 1.0e-20],
        [3.0e-12, 2.0e-12, 1.0e-12],
        density_uncertainty_m3=[1.0e-21, 1.0e-21, 1.0e-21],
        temperature_uncertainty_ev=[1.0e-13, 1.0e-13, 1.0e-13],
    )
    assert np.all(np.isfinite(out["density_scale_length_m"]))
    assert np.all(np.isfinite(out["density_scale_length_uncertainty_m"]))
    assert np.all(np.isfinite(out["temperature_scale_length_m"]))
    assert np.all(np.isfinite(out["temperature_scale_length_uncertainty_m"]))



from tokamak_analysis.thomson_scattering import reconstruct_thomson_profile


@pytest.mark.parametrize("method", ["smoothing_spline", "pchip"])
def test_thomson_reconstruction_matches_registry_output_contract(method):
    out = reconstruct_thomson_profile(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [1.0e19, 1.4e19, 1.8e19, 1.3e19, 0.8e19],
        [80.0, 120.0, 160.0, 110.0, 60.0],
        method=method,
        smoothing_factor=0.0,
        points=41,
    )
    required = {
        "reconstructed_radius_m",
        "reconstructed_density_m3",
        "reconstructed_temperature_eV",
        "reconstruction_method",
        "reconstruction_gradients",
        "reconstruction_domain_m",
    }
    assert required.issubset(out)
    assert out["reconstruction_method"] == method
    assert out["reconstructed_radius_m"].shape == (41,)
    assert out["reconstructed_density_m3"].shape == (41,)
    assert out["reconstructed_temperature_eV"].shape == (41,)
    assert set(out["reconstruction_gradients"]) == {
        "density_gradient_m-4", "temperature_gradient_eV_m-1"
    }
    assert out["reconstruction_gradients"]["density_gradient_m-4"].shape == (41,)
    assert out["reconstruction_gradients"]["temperature_gradient_eV_m-1"].shape == (41,)
    assert out["reconstruction_domain_m"] == (0.0, 1.0)
    assert out["radius_m"] is out["reconstructed_radius_m"]


@pytest.mark.parametrize(
    "density_sigma, temperature_sigma, expected_density, expected_temperature",
    [
        ([1e17] * 5, None, "inverse_sigma_weighted", "not_supplied"),
        (None, [2.0] * 5, "not_supplied", "inverse_sigma_weighted"),
    ],
)
def test_thomson_spline_uncertainty_metadata_is_channel_specific(
    density_sigma, temperature_sigma, expected_density, expected_temperature
):
    out = reconstruct_thomson_profile(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [1e19, 1.2e19, 1.4e19, 1.2e19, 1e19],
        [50.0, 70.0, 90.0, 70.0, 50.0],
        density_uncertainty_m3=density_sigma,
        temperature_uncertainty_ev=temperature_sigma,
        method="smoothing_spline",
        smoothing_factor=0.5,
        points=31,
    )
    model = out["uncertainty_model"]
    assert model["density_channel"] == expected_density
    assert model["temperature_channel"] == expected_temperature
    assert model["fit_confidence_intervals_estimated"] is False


def test_thomson_pchip_uncertainties_are_reported_but_not_used_as_fit_weights():
    out = reconstruct_thomson_profile(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [1e19, 1.2e19, 1.4e19, 1.2e19, 1e19],
        [50.0, 70.0, 90.0, 70.0, 50.0],
        density_uncertainty_m3=[1e17] * 5,
        method="pchip",
        points=31,
    )
    assert out["uncertainty_model"]["density_channel"] == "not_used_by_fit"
    assert out["uncertainty_model"]["temperature_channel"] == "not_supplied"


@pytest.mark.parametrize("method", ["smoothing_spline", "pchip"])
def test_thomson_reconstruction_clipping_keeps_derivative_consistent(method):
    out = reconstruct_thomson_profile(
        [0.0, 0.25, 0.5, 0.75, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 10.0, 0.0, 10.0, 0.0],
        method=method,
        smoothing_factor=100.0,
        points=101,
    )
    ne = out["reconstructed_density_m3"]
    te = out["reconstructed_temperature_eV"]
    dne = out["reconstruction_gradients"]["density_gradient_m-4"]
    dte = out["reconstruction_gradients"]["temperature_gradient_eV_m-1"]
    assert np.all(ne >= 0.0)
    assert np.all(te >= 0.0)
    assert np.all(np.isfinite(dne))
    assert np.all(np.isfinite(dte))
    if method == "smoothing_spline":
        # Any sample projected from a negative spline overshoot has a zero
        # derivative by construction. Do not impose that rule on native
        # non-negative PCHIP zeros, whose one-sided slope can be non-zero.
        np.testing.assert_array_equal(dne[ne == 0.0], np.zeros(np.count_nonzero(ne == 0.0)))
        np.testing.assert_array_equal(dte[te == 0.0], np.zeros(np.count_nonzero(te == 0.0)))



def test_nonuniform_grid_quadratic_gradients_normalized_gradients_and_scale_lengths():
    radius = np.array([0.0, 0.4, 1.3, 2.7, 4.0])
    density = 2.0e19 + 3.0e18 * radius + 2.0e18 * radius**2
    temperature = 100.0 + 20.0 * radius + 5.0 * radius**2
    density_gradient = 3.0e18 + 4.0e18 * radius
    temperature_gradient = 20.0 + 10.0 * radius

    out = analyze_thomson_profile(radius, density, temperature)

    np.testing.assert_allclose(out["density_gradient_m-4"], density_gradient, rtol=2e-14)
    np.testing.assert_allclose(out["temperature_gradient_eV_m-1"], temperature_gradient, rtol=2e-14)
    np.testing.assert_allclose(out["density_normalized_gradient_m-1"], density_gradient / density, rtol=2e-14)
    np.testing.assert_allclose(out["temperature_normalized_gradient_m-1"], temperature_gradient / temperature, rtol=2e-14)
    np.testing.assert_allclose(out["density_scale_length_m"], -density / density_gradient, rtol=2e-14)
    np.testing.assert_allclose(out["temperature_scale_length_m"], -temperature / temperature_gradient, rtol=2e-14)


def test_local_gradient_resolvability_is_not_controlled_by_largest_remote_gradient():
    radius = np.arange(5.0)
    profile = np.array([1.0, 1.0 + 1.0e-10, 1.0 + 2.0e-10, 1.0 + 1.0e10, 1.0 + 2.0e10])
    out = analyze_thomson_profile(radius, profile, profile * 2.0)
    assert np.isfinite(out["density_gradient_m-4"][0])
    assert out["density_gradient_m-4"][0] > 0.0
    assert np.isfinite(out["density_scale_length_m"][0])
    assert np.isfinite(out["temperature_scale_length_m"][0])
    assert out["density_scale_length_m"][0] == pytest.approx(-profile[0] / out["density_gradient_m-4"][0])


def test_gradient_and_scale_length_uncertainties_match_independent_matrix_formula():
    radius = np.array([0.0, 0.5, 1.7, 3.0])
    density = 5.0 + 2.0 * radius + 0.3 * radius**2
    sigma = np.array([0.1, 0.2, 0.15, 0.12])
    out = analyze_thomson_profile(radius, density, 2.0 * density,
        density_uncertainty_m3=sigma, temperature_uncertainty_ev=2.0 * sigma)
    derivative_matrix = np.gradient(np.eye(radius.size), radius, axis=0, edge_order=2)
    gradient = derivative_matrix @ density
    gradient_sigma = np.sqrt((derivative_matrix**2) @ (sigma**2))
    covariance = np.diag(derivative_matrix) * sigma**2
    expected_scale_sigma = np.sqrt(np.maximum(
        sigma**2 / gradient**2 + density**2 * gradient_sigma**2 / gradient**4
        - 2.0 * density * covariance / gradient**3, 0.0))
    np.testing.assert_allclose(out["density_gradient_uncertainty_m-4"], gradient_sigma, rtol=2e-14, atol=1e-15)
    np.testing.assert_allclose(out["density_scale_length_uncertainty_m"], expected_scale_sigma, rtol=2e-13, atol=1e-14)
    assert np.all(np.isfinite(out["density_gradient_uncertainty_m-4"][[0, 1, -1]]))


def test_nonuniform_trapezoidal_line_average_uncertainty_uses_correct_weights():
    radius = np.array([0.0, 0.5, 2.0, 3.0])
    density = np.array([2.0, 3.0, 5.0, 7.0])
    sigma = np.array([0.1, 0.2, 0.3, 0.4])
    out = analyze_thomson_profile(radius, density, density * 10.0, density_uncertainty_m3=sigma)
    weights = np.array([0.25, 1.0, 1.25, 0.5]) / 3.0
    assert out["summary"].density_line_average_m3 == pytest.approx(np.sum(weights * density))
    assert out["summary"].density_line_average_uncertainty_m3 == pytest.approx(np.sqrt(np.sum((weights * sigma)**2)))


@pytest.mark.parametrize("targets", [[], [True], ["0.5"], [np.nan], [np.inf]])
def test_interpolation_rejects_empty_boolean_string_or_nonfinite_targets(targets):
    profile = build_profile([0.0, 1.0], [1.0, 2.0], [10.0, 20.0])
    with pytest.raises(ValueError):
        interpolate_profile(profile, targets)


def test_interpolation_accepts_nonempty_finite_real_numeric_targets():
    profile = build_profile([0.0, 1.0], [1.0, 3.0], [10.0, 30.0])
    result = interpolate_profile(profile, np.array([0.25, 0.75]))
    np.testing.assert_allclose(result["density_m3"], [1.5, 2.5])
    np.testing.assert_allclose(result["temperature_ev"], [15.0, 25.0])

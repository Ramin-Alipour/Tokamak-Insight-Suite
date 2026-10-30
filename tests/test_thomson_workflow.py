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
    np.testing.assert_array_equal(dne[ne == 0.0], np.zeros(np.count_nonzero(ne == 0.0)))
    np.testing.assert_array_equal(dte[te == 0.0], np.zeros(np.count_nonzero(te == 0.0)))

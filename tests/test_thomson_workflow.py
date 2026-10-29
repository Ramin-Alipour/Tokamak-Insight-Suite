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


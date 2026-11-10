import numpy as np

from tis.products.thomson.physics.thomson_distribution import (
    _moment_statistics,
    analyze_thomson_spectral_distribution,
)


def test_projected_evdf_moment_closure_includes_measured_response():
    """Closure must compare response-broadened prediction with measured moments."""
    wavelength = np.linspace(1059.5, 1060.5, 81)
    laser = 1060.0
    angle = np.pi / 2.0
    n = wavelength.size

    # A normalized finite-resolution line-spread operator.  Its columns are
    # probabilities, so the forward operation conserves integrated power.
    pixel = np.arange(n, dtype=float)
    response = np.exp(-0.5 * ((pixel[:, None] - pixel[None, :]) / 1.2) ** 2)
    response /= response.sum(axis=0, keepdims=True)

    intrinsic = np.exp(-0.5 * ((wavelength - laser) / 0.075) ** 2)
    measured = response @ intrinsic
    background = np.zeros_like(measured)

    result = analyze_thomson_spectral_distribution(
        wavelength,
        measured,
        instrument_response_matrix=response,
        background_signal=background,
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        projected_evdf_options={
            "velocity_grid_m_s": np.linspace(-4.0e5, 4.0e5, 161),
            "regularization_strength": 1.0e-3,
            "max_iter": 3000,
        },
    )

    inverse = result["projected_evdf_inference"]
    closure = inverse["spectral_moment_closure"]
    expected_primary = (
        inverse.get("primary_result_status", "identified_from_measurement")
        == "identified_from_measurement"
    )
    assert closure["status"] == (
        "identified_response_aware_moment_closure"
        if expected_primary else "conditional_response_aware_moment_closure"
    )
    assert closure["closure_is_primary"] is expected_primary
    assert closure["primary_result_status"] == (
        "identified_from_measurement"
        if expected_primary else "not_fully_identifiable_from_measured_spectrum"
    )
    assert closure["instrument_response_in_closure"] is True
    assert closure["background_model_shared_with_measured_moments"] is True

    # Independently recompute the predicted moments from the actual fitted
    # measured-space spectrum returned by the inverse forward operator.
    fitted_moments = _moment_statistics(
        wavelength,
        np.asarray(inverse["fitted_spectrum"]) + background,
        None,
        None,
        background,
    )
    assert np.isclose(
        closure["response_forward_modelled_centroid_nm"],
        fitted_moments["center"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert np.isclose(
        closure["response_forward_modelled_sigma_nm"],
        fitted_moments["sigma_observed"],
        rtol=1e-12,
        atol=1e-12,
    )


def test_projected_evdf_accepts_discrepancy_and_systematic_covariance_without_parametric_model():
    wavelength = np.linspace(1059.5, 1060.5, 81)
    laser = 1060.0
    n = wavelength.size
    pixel = np.arange(n, dtype=float)
    response = np.exp(-0.5 * ((pixel[:, None] - pixel[None, :]) / 1.2) ** 2)
    response /= response.sum(axis=0, keepdims=True)
    intrinsic = np.exp(-0.5 * ((wavelength - laser) / 0.075) ** 2)
    measured = response @ intrinsic
    background = np.zeros_like(measured)

    result = analyze_thomson_spectral_distribution(
        wavelength,
        measured,
        instrument_response_matrix=response,
        background_signal=background,
        laser_wavelength_nm=laser,
        scattering_angle_rad=np.pi / 2.0,
        model_discrepancy_covariance=np.eye(n) * 1.0e-4,
        systematic_covariance=np.eye(n) * 2.0e-4,
        projected_evdf_options={
            "velocity_grid_m_s": np.linspace(-4.0e5, 4.0e5, 161),
            "regularization_strength": 1.0e-3,
            "max_iter": 3000,
        },
    )

    inverse = result["projected_evdf_inference"]
    assert inverse["model_discrepancy_covariance_included"] is True
    assert inverse["systematic_covariance_included"] is True
    assert inverse["additive_covariance_independence_assumption"]
    assert "physical_inference" not in result

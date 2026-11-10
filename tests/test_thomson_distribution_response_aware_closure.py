import numpy as np
import pytest

from tis.products.thomson.physics.thomson_distribution import (
    _moment_statistics,
    _projected_evdf_physical_moments,
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



def test_projected_evdf_physical_moments_recover_analytic_asymmetric_mixture():
    """A known two-Maxwellian 1-D mixture checks pressure and heat-flux moments.

    For weights (3/4, 1/4), component means (-s, 3s), and common
    component variance s^2, the mixture has mean 0, variance 4s^2,
    central third moment 6s^3 and central fourth moment 42s^4.
    """
    from scipy.constants import elementary_charge, electron_mass

    scale = 2.0e5
    velocity = np.linspace(-10.0 * scale, 12.0 * scale, 12001)
    gaussian = lambda mean: np.exp(
        -0.5 * ((velocity - mean) / scale) ** 2
    ) / (np.sqrt(2.0 * np.pi) * scale)
    pdf = 0.75 * gaussian(-scale) + 0.25 * gaussian(3.0 * scale)
    density = 4.0e18

    moments = _projected_evdf_physical_moments(
        velocity,
        pdf,
        electron_density_m3=density,
    )

    expected_variance = 4.0 * scale**2
    expected_mu3 = 6.0 * scale**3
    expected_mu4 = 42.0 * scale**4
    expected_temperature = electron_mass * expected_variance / elementary_charge
    expected_pressure = electron_mass * density * expected_variance
    expected_projected_heat_flux = 0.5 * electron_mass * density * expected_mu3

    assert moments["mean_velocity_parallel_m_s"] == pytest.approx(0.0, abs=2e-6 * scale)
    assert moments["velocity_parallel_variance_m2_s2"] == pytest.approx(expected_variance, rel=2e-5)
    assert moments["directional_equivalent_temperature_eV"] == pytest.approx(expected_temperature, rel=2e-5)
    assert moments["central_third_velocity_moment_m3_s3"] == pytest.approx(expected_mu3, rel=3e-5)
    assert moments["central_fourth_velocity_moment_m4_s4"] == pytest.approx(expected_mu4, rel=3e-5)
    assert moments["skewness"] == pytest.approx(6.0 / (4.0 ** 1.5), rel=3e-5)
    assert moments["excess_kurtosis"] == pytest.approx(42.0 / 16.0 - 3.0, rel=3e-5)
    assert moments["parallel_projected_pressure_Pa"] == pytest.approx(expected_pressure, rel=2e-5)
    assert moments["parallel_projected_heat_flux_W_m2"] == pytest.approx(expected_projected_heat_flux, rel=3e-5)
    assert "not_the_full_three_dimensional_electron_heat_flux" in moments["heat_flux_physical_scope"]

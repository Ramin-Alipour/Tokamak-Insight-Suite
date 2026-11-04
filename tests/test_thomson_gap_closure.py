import numpy as np
import pytest

from tokamak_analysis.thomson_gap_closure import (
    reconstruct_psf_aware_thomson_profile,
    reconstruct_relativistic_projected_thomson_evdf,
)


def test_psf_aware_profile_recovers_full_rank_latent_profile():
    mixing = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    truth = np.array([2.0, 1.0])
    result = reconstruct_psf_aware_thomson_profile(
        mixing @ truth, mixing, regularization_strength=1e-10,
        measurement_uncertainty=np.full(3, 1e-3),
    )
    assert result["status"] == "identified_from_measurement"
    np.testing.assert_allclose(result["latent_profile"], truth, rtol=2e-3, atol=2e-3)
    assert result["null_space_dimension"] == 0


def test_psf_aware_profile_does_not_report_null_space_as_primary():
    mixing = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    result = reconstruct_psf_aware_thomson_profile(
        np.array([2.0, 2.0]), mixing, regularization_strength=1e-2,
    )
    assert result["latent_profile"] is None
    assert result["status"] == "conditional_regularized_candidate"
    assert result["null_space_dimension"] >= 1


def test_relativistic_projected_evdf_normalizes_and_reports_model_scope():
    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    angle = np.pi / 2
    velocity = np.linspace(-3.0e6, 3.0e6, 121)
    beta = velocity / 299792458.0
    centres = laser * (1.0 - beta * np.sin(angle / 2)) / (1.0 + beta * np.sin(angle / 2))
    widths = np.diff(np.r_[wavelength[0]-(wavelength[1]-wavelength[0])/2,
                           0.5*(wavelength[:-1]+wavelength[1:]),
                           wavelength[-1]+(wavelength[-1]-wavelength[-2])/2])
    latent = np.zeros((wavelength.size, velocity.size))
    for j, centre in enumerate(centres):
        right = np.searchsorted(wavelength, centre, side="right")
        left = right - 1
        frac = (centre-wavelength[left])/(wavelength[right]-wavelength[left])
        latent[left,j] = (1-frac)/widths[left]
        latent[right,j] = frac/widths[right]
    dv_weights = np.full(velocity.size, velocity[1] - velocity[0])
    dv_weights[[0, -1]] *= 0.5
    forward = latent * dv_weights[None, :]
    truth = np.exp(-0.5*(velocity/7e5)**2)
    truth /= np.trapz(truth, velocity)
    measured = forward @ truth
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, measured, laser_wavelength_nm=laser,
        scattering_angle_rad=angle, instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(measured), velocity_grid_m_s=velocity,
        regularization_strength=1e-3,
    )
    assert result["status"] in {"completed", "conditional_regularized_candidate"}
    candidate = result["regularized_candidate_evdf_s_m"]
    assert np.isclose(np.trapz(candidate, velocity), 1.0, rtol=1e-8)
    if result["status"] == "completed":
        assert result["projected_evdf_s_m"] is not None
    else:
        assert result["projected_evdf_s_m"] is None
    assert result["absolute_density_inferred"] is False
    assert "Compton recoil" in " ".join(result["assumptions"])


def test_velocity_quadrature_is_included_and_grid_integral_is_resolution_consistent():
    from tokamak_analysis.thomson_gap_closure import _trapezoid_quadrature_weights

    coarse = np.linspace(-2.0e6, 2.0e6, 101)
    fine = np.linspace(-2.0e6, 2.0e6, 401)
    coarse_weights = _trapezoid_quadrature_weights(coarse, "velocity_grid_m_s")
    fine_weights = _trapezoid_quadrature_weights(fine, "velocity_grid_m_s")
    coarse_integral = np.sum(np.ones_like(coarse) * coarse_weights)
    fine_integral = np.sum(np.ones_like(fine) * fine_weights)
    np.testing.assert_allclose(coarse_integral, 4.0e6, rtol=1e-12)
    np.testing.assert_allclose(fine_integral, coarse_integral, rtol=1e-12)
    assert coarse_weights[0] == 0.5 * (coarse[1] - coarse[0])
    assert coarse_weights[-1] == coarse_weights[0]


def test_projected_compton_kinematics_matches_rest_electron_compton_shift():
    from tokamak_analysis.thomson_gap_closure import (
        relativistic_projected_compton_kinematics,
    )

    laser = 532.0
    angle = np.pi / 2.0
    velocity = np.linspace(-3.0e6, 3.0e6, 121)
    result = relativistic_projected_compton_kinematics(
        velocity, laser, angle, incident_polarization=[0.0, 0.0, 1.0],
    )
    zero = velocity.size // 2
    photon_energy_over_rest = (
        1239.8419843320026 / laser / 510_998.95069
    )
    expected = laser * (1.0 + photon_energy_over_rest)
    assert result["compton_scattered_wavelength_nm"][zero] == pytest.approx(
        expected, rel=2e-13
    )
    assert result["thomson_scattered_wavelength_nm"][zero] == pytest.approx(
        laser, rel=2e-15
    )
    assert result["maximum_recoil_correction_fraction"] > 0.0
    weights = result["relative_lab_differential_scattering_weight"]
    assert weights.shape == velocity.shape
    assert np.all(np.isfinite(weights))
    assert np.all(weights >= 0.0)
    thomson_weights = result["relative_lab_thomson_differential_scattering_weight"]
    assert thomson_weights.shape == velocity.shape
    assert np.all(np.isfinite(thomson_weights))
    assert np.all(thomson_weights >= 0.0)


def test_projected_compton_inverse_uses_recoil_and_polarized_kn_weight():
    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    angle = np.pi / 2.0
    velocity = np.linspace(-3.0e6, 3.0e6, 121)
    measured = np.exp(-0.5 * ((wavelength - laser) / 2.0) ** 2)
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, measured, laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(measured),
        velocity_grid_m_s=velocity,
        incident_polarization=[0.0, 0.0, 1.0],
        include_compton_recoil=True,
        regularization_strength=1e-3,
    )
    assert result["recoil_included"] is True
    assert result["scattering_cross_section_model"] == (
        "polarized_single_scattering_Klein_Nishina_in_electron_rest_frame"
    )
    assert result["maximum_recoil_correction_fraction"] > 0.0
    assert result["relative_lab_differential_scattering_weight"].shape == velocity.shape
    assert result["absolute_density_inferred"] is False


def test_projected_thomson_inverse_rejects_out_of_domain_no_recoil_approximation():
    import pytest

    wavelength = np.linspace(0.08, 0.12, 241)
    laser = 0.1
    velocity = np.linspace(-3.0e6, 3.0e6, 121)
    measured = np.exp(-0.5 * ((wavelength - laser) / 0.01) ** 2)
    with pytest.raises(ValueError, match="exceeds the selected correction limit"):
        reconstruct_relativistic_projected_thomson_evdf(
            wavelength, measured, laser_wavelength_nm=laser,
            scattering_angle_rad=np.pi / 2.0,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.zeros_like(measured),
            velocity_grid_m_s=velocity,
            regularization_strength=1e-3,
        )


def test_projected_compton_inverse_requires_incident_polarization():
    import pytest

    wavelength = np.linspace(520.0, 544.0, 241)
    laser = 532.0
    velocity = np.linspace(-3.0e6, 3.0e6, 121)
    measured = np.exp(-0.5 * ((wavelength - laser) / 2.0) ** 2)
    with pytest.raises(ValueError, match="incident_polarization is required"):
        reconstruct_relativistic_projected_thomson_evdf(
            wavelength, measured, laser_wavelength_nm=laser,
            scattering_angle_rad=np.pi / 2.0,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.zeros_like(measured),
            velocity_grid_m_s=velocity,
            include_compton_recoil=True,
            regularization_strength=1e-3,
        )


def test_compton_kinematics_is_rotation_covariant_for_arbitrary_3d_rays():
    from tokamak_analysis.thomson_gap_closure import (
        relativistic_projected_compton_kinematics,
    )

    velocity = np.linspace(-2.0e7, 2.0e7, 101)
    angle = 1.13
    sine, cosine = np.sin(angle / 2.0), np.cos(angle / 2.0)
    incident = np.array([-sine, cosine, 0.0])
    scattered = np.array([sine, cosine, 0.0])
    polarization = np.array([0.0, 0.0, 1.0])
    reference = relativistic_projected_compton_kinematics(
        velocity, 532.0, angle, incident_polarization=polarization,
    )

    # A proper 3-D rotation moves both rays and the polarization vector while
    # preserving all scalar Compton and polarized-KN observables.
    azimuth, tilt = 0.71, -0.43
    rz = np.array([
        [np.cos(azimuth), -np.sin(azimuth), 0.0],
        [np.sin(azimuth), np.cos(azimuth), 0.0],
        [0.0, 0.0, 1.0],
    ])
    ry = np.array([
        [np.cos(tilt), 0.0, np.sin(tilt)],
        [0.0, 1.0, 0.0],
        [-np.sin(tilt), 0.0, np.cos(tilt)],
    ])
    rotation = rz @ ry
    rotated = relativistic_projected_compton_kinematics(
        velocity, 532.0, angle,
        incident_direction=rotation @ incident,
        scattered_direction=rotation @ scattered,
        incident_polarization=rotation @ polarization,
    )
    np.testing.assert_allclose(
        rotated["compton_scattered_wavelength_nm"],
        reference["compton_scattered_wavelength_nm"], rtol=2e-13, atol=0.0,
    )
    np.testing.assert_allclose(
        rotated["relative_lab_differential_scattering_weight"],
        reference["relative_lab_differential_scattering_weight"],
        rtol=2e-12, atol=1e-14,
    )
    np.testing.assert_allclose(
        rotated["scattering_vector_unit"],
        rotation @ reference["scattering_vector_unit"], rtol=1e-12, atol=1e-12,
    )
    assert rotated["geometry_source"].startswith("researcher_supplied")


def test_compton_kinematics_rejects_inconsistent_arbitrary_ray_geometry():
    from tokamak_analysis.thomson_gap_closure import (
        relativistic_projected_compton_kinematics,
    )

    with pytest.raises(ValueError, match="must match"):
        relativistic_projected_compton_kinematics(
            np.linspace(-1e6, 1e6, 41), 532.0, np.pi / 2,
            incident_direction=[0.0, 0.0, 1.0],
            scattered_direction=[1.0, 0.0, 0.0],
        )
    with pytest.raises(ValueError, match="unit vectors"):
        relativistic_projected_compton_kinematics(
            np.linspace(-1e6, 1e6, 41), 532.0, np.pi / 2,
            incident_direction=[0.0, 0.0, 2.0],
            scattered_direction=[1.0, 0.0, 0.0],
        )



def test_psf_aware_profile_selects_regularization_by_gcv_when_omitted():
    mixing = np.array([
        [0.80, 0.20, 0.00, 0.00],
        [0.20, 0.60, 0.20, 0.00],
        [0.00, 0.20, 0.60, 0.20],
        [0.00, 0.00, 0.20, 0.80],
        [0.10, 0.30, 0.30, 0.10],
    ])
    truth = np.array([1.0, 2.0, 2.0, 1.0])
    measured = mixing @ truth + np.array([0.01, -0.02, 0.01, 0.0, 0.01])
    result = reconstruct_psf_aware_thomson_profile(
        measured, mixing, measurement_uncertainty=np.full(measured.size, 0.02),
    )
    assert result["regularization_selection"] == "generalized_cross_validation_active_set"
    assert result["regularization_strength"] > 0.0
    assert result["gcv_candidate_strengths"].size == 31
    assert result["gcv_scores"].shape == result["gcv_candidate_strengths"].shape
    assert np.all(np.isfinite(result["gcv_scores"]))
    assert result["data_only_rank"] == np.linalg.matrix_rank(mixing)
    assert result["status"] in {"identified_from_measurement", "conditional_regularized_candidate"}


def test_psf_aware_profile_gcv_does_not_promote_rank_deficient_modes():
    mixing = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    result = reconstruct_psf_aware_thomson_profile(np.array([2.0, 2.0]), mixing)
    assert result["regularization_selection"] == "generalized_cross_validation_active_set"
    assert result["latent_profile"] is None
    assert result["status"] == "conditional_regularized_candidate"
    assert result["null_space_dimension"] >= 1


def test_psf_aware_profile_rejects_invalid_gcv_controls():
    mixing = np.eye(3)
    with pytest.raises(ValueError, match="gcv_candidate_count"):
        reconstruct_psf_aware_thomson_profile(np.ones(3), mixing, gcv_candidate_count=3)
    with pytest.raises(ValueError, match="regularization_strength"):
        reconstruct_psf_aware_thomson_profile(np.ones(3), mixing, regularization_strength=0.0)



def test_psf_profile_propagates_supplied_mixing_operator_covariance():
    mixing = np.array([
        [0.90, 0.10, 0.00],
        [0.10, 0.80, 0.10],
        [0.00, 0.10, 0.90],
    ])
    truth = np.array([1.0, 2.0, 1.0])
    measured = mixing @ truth
    operator_covariance = np.eye(mixing.size) * 1.0e-6
    result = reconstruct_psf_aware_thomson_profile(
        measured, mixing, regularization_strength=1.0e-3,
        measurement_uncertainty=np.full(3, 0.01),
        spatial_mixing_covariance=operator_covariance,
    )
    assert result["psf_uncertainty_propagated"] is True
    assert result["profile_covariance_with_psf_uncertainty"] is not None
    contribution = result["psf_operator_covariance_contribution"]
    assert contribution.shape == (3, 3)
    np.testing.assert_allclose(contribution, contribution.T, atol=1e-12)
    assert np.linalg.eigvalsh(contribution).min() >= -1e-12
    assert np.trace(result["profile_covariance_with_psf_uncertainty"]) >= np.trace(
        result["profile_covariance"]
    )


def test_psf_profile_rejects_non_psd_operator_covariance():
    mixing = np.eye(3)
    invalid = np.eye(9)
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_psf_aware_thomson_profile(
            np.array([1.0, 2.0, 1.0]), mixing,
            regularization_strength=1.0e-3,
            spatial_mixing_covariance=invalid,
        )


def test_projected_evdf_adds_independent_model_discrepancy_covariance_once():
    wavelength = np.linspace(520.0, 544.0, 121)
    velocity = np.linspace(-3.0e6, 3.0e6, 61)
    measured = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2)
    discrepancy = np.eye(wavelength.size) * 2.5e-4
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, measured, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(measured),
        velocity_grid_m_s=velocity,
        model_discrepancy_covariance=discrepancy,
        regularization_strength=1e-3,
    )
    assert result["model_discrepancy_included"] is True
    assert result["measurement_covariance"] is None
    np.testing.assert_allclose(
        result["model_discrepancy_covariance"], discrepancy, rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        result["effective_spectral_covariance"], discrepancy, rtol=0.0, atol=0.0
    )
    assert result["noise_model"] == "model_discrepancy_only_no_measurement_noise_covariance"


def test_projected_evdf_rejects_non_psd_model_discrepancy_covariance():
    wavelength = np.linspace(520.0, 544.0, 81)
    velocity = np.linspace(-2.0e6, 2.0e6, 41)
    measured = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2)
    discrepancy = np.eye(wavelength.size)
    discrepancy[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_relativistic_projected_thomson_evdf(
            wavelength, measured, laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0,
            instrument_response_matrix=np.eye(wavelength.size),
            background_signal=np.zeros_like(measured),
            velocity_grid_m_s=velocity,
            model_discrepancy_covariance=discrepancy,
            regularization_strength=1e-3,
        )


def test_projected_evdf_normalizes_on_nonuniform_velocity_grid():
    wavelength = np.linspace(520.0, 544.0, 121)
    coordinate = np.linspace(-1.0, 1.0, 61)
    velocity = 3.0e6 * (coordinate + 0.15 * coordinate**3)
    measured = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2)
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, measured, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(measured),
        velocity_grid_m_s=velocity,
        regularization_strength=1e-3,
    )
    weights = result["velocity_quadrature_weights_m_s"]
    candidate = result["regularized_candidate_evdf_s_m"]
    assert result["velocity_grid_spacing_uniform"] is False
    assert result["regularization_operator"] == (
        "nonuniform_three_point_second_derivative_on_interior_velocity_nodes"
    )
    np.testing.assert_allclose(np.dot(candidate, weights), 1.0, rtol=1e-10, atol=1e-10)

def test_nonuniform_velocity_grid_uses_composite_trapezoid_weights():
    wavelength = np.linspace(520.0, 544.0, 81)
    velocity = np.linspace(-1.0, 1.0, 41) ** 3 * 1.0e6
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2)
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, signal, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=np.zeros_like(signal),
        velocity_grid_m_s=velocity, regularization_strength=1.0e-3,
    )
    steps = np.diff(velocity)
    expected = np.r_[0.5 * steps[0], 0.5 * (steps[:-1] + steps[1:]), 0.5 * steps[-1]]
    np.testing.assert_allclose(result["velocity_quadrature_weights_m_s"], expected)
    assert result["regularization_discretization"] == "nonuniform_quadratic_second_derivative_with_dual_cell_quadrature"


def test_projected_evdf_reports_total_fit_including_structured_background():
    wavelength = np.linspace(520.0, 544.0, 101)
    velocity = np.linspace(-2.0e6, 2.0e6, 41)
    background = 0.03 + 0.01 * (wavelength - wavelength.mean()) / np.ptp(wavelength)
    scattered = np.exp(-0.5 * ((wavelength - 532.0) / 2.0) ** 2)
    measured = background + scattered
    result = reconstruct_relativistic_projected_thomson_evdf(
        wavelength, measured, laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response_matrix=np.eye(wavelength.size),
        background_signal=background,
        velocity_grid_m_s=velocity,
        regularization_strength=1e-3,
    )
    np.testing.assert_allclose(
        result["fitted_spectrum"],
        result["fitted_background_spectrum"] + result["fitted_scattered_spectrum"],
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        result["residual"], measured - result["fitted_spectrum"],
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(result["fitted_background_spectrum"], background)
    assert result["residual_definition"] == (
        "measured_total_signal_minus_fitted_background_and_scattered_signal"
    )


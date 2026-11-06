import numpy as np
import pytest

from tokamak_analysis.thomson_anisotropic_inference import (
    fit_gyrotropic_bi_maxwellian_thomson,
    _apply_response_in_bin_integrated_space,
    _gaussian_bin_average,
    _kappa_bin_average,
)


def _synthetic_spectra():
    coordinate = np.linspace(-1.0, 1.0, 161)
    wavelength = 1064.0 + 2.0 * (coordinate + 0.12 * coordinate**3)
    laser = 1064.0
    angle = np.pi / 2.0
    b = np.array([0.0, 0.0, 1.0])
    q = np.array([
        [0.0, 0.0, 1.0],
        [np.sqrt(3.0) / 2.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
    ])
    tpar, tperp = 900.0, 250.0
    mu2 = (q @ b) ** 2
    directional = tperp + (tpar - tperp) * mu2
    flows = np.array([1.0e4, -1.5e4, 2.0e4])
    spectra = []
    for i, tq in enumerate(directional):
        sigma_v = np.sqrt(1.602176634e-19 * tq / 9.1093837015e-31)
        center = laser * (1.0 - 2.0*np.sin(angle/2.0)*flows[i]/299792458.0)
        sigma_lam = laser * 2.0*np.sin(angle/2.0)*sigma_v/299792458.0
        profile, _ = _gaussian_bin_average(wavelength, center, sigma_lam)
        spectra.append((0.03 + 0.8*profile + 1.0e-4*(wavelength-laser)))
    return wavelength, np.asarray(spectra), b, q


def test_joint_anisotropic_optical_fit_recovers_parallel_and_perpendicular_temperatures():
    wavelength, spectra, b, q = _synthetic_spectra()
    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, spectra, laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2,
        magnetic_field_unit=b, scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=700.0,
        initial_temperature_perpendicular_eV=350.0,
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    assert result["reportable_as_inferred_parameter"] is True
    assert result["electron_temperature_parallel_eV"] == pytest.approx(900.0, rel=2e-2)
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(250.0, rel=2e-2)
    covariance = result["thermal_velocity_covariance_tensor_per_view_m2_s2"]
    assert covariance.shape == (3, 3, 3)
    np.testing.assert_allclose(
        result["projected_velocity_variance_from_tensor_m2_s2"],
        result["projected_velocity_variance_expected_m2_s2"],
        rtol=1e-12, atol=1e-6,
    )
    assert np.max(np.abs(result["projected_velocity_variance_relative_residual"])) < 1e-12
    assert result["temperature_anisotropy_ratio_parallel_over_perpendicular"] == pytest.approx(900.0 / 250.0)
    for tensor in covariance:
        np.testing.assert_allclose(tensor, tensor.T, rtol=1e-12, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(tensor)) > 0.0
    assert result["electron_velocity_projection_m_s"] == pytest.approx([1.0e4, -1.5e4, 2.0e4], abs=2.0e3)
    assert np.max(np.abs(result["residual_spectra"])) < 1e-5
    assert result["model_validity"]["relativistic_corrections"] is False
    tail = result["relativistic_tail_diagnostic"]
    assert tail["requires_relativistic_forward_model"] is False
    assert tail["maximum_projected_superluminal_probability_lower_bound"] < 1.0e-8


def test_single_pitch_angle_does_not_report_separate_anisotropy_temperatures():
    wavelength, spectra, b, q = _synthetic_spectra()
    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, spectra[:1], laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2,
        magnetic_field_unit=b, scattering_vector_unit=q[:1],
        instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=2.0,
        initial_temperature_perpendicular_eV=0.5,
        temperature_bounds_eV=(0.1, 5.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    assert result["anisotropy_identifiability"] == "non_identifiable"
    assert result["electron_temperature_parallel_eV"] is None
    assert result["electron_temperature_perpendicular_eV"] is None
    assert np.isfinite(result["directional_temperature_eV"][0])


def test_anisotropic_fit_rejects_non_normalized_response_and_bad_geometry():
    wavelength, spectra, b, q = _synthetic_spectra()
    response = np.eye(wavelength.size)
    response[:, 0] *= 0.5
    with pytest.raises(ValueError, match="columns must sum to one"):
        fit_gyrotropic_bi_maxwellian_thomson(
            wavelength, spectra, laser_wavelength_nm=1064.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
            scattering_vector_unit=q, instrument_response_matrix=response,
        )
    with pytest.raises(ValueError, match="unit norm"):
        fit_gyrotropic_bi_maxwellian_thomson(
            wavelength, spectra, laser_wavelength_nm=1064.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=[0, 0, 2],
            scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
        )


def test_nonuniform_response_operator_conserves_bin_integrated_spectral_power():
    wavelength = np.array([500.0, 500.2, 500.7, 501.5, 502.8])
    widths = np.diff(np.r_[wavelength[0] - 0.5*(wavelength[1]-wavelength[0]),
                           0.5*(wavelength[:-1] + wavelength[1:]),
                           wavelength[-1] + 0.5*(wavelength[-1]-wavelength[-2])])
    intrinsic = np.array([0.2, 1.1, 2.7, 0.8, 0.1])
    response = np.array([
        [0.8, 0.1, 0.0, 0.0, 0.0],
        [0.2, 0.7, 0.1, 0.0, 0.0],
        [0.0, 0.2, 0.7, 0.1, 0.0],
        [0.0, 0.0, 0.2, 0.7, 0.1],
        [0.0, 0.0, 0.0, 0.2, 0.9],
    ])
    measured_density = _apply_response_in_bin_integrated_space(
        response, intrinsic, wavelength
    )
    assert np.sum(measured_density * widths) == pytest.approx(
        np.sum(intrinsic * widths), rel=1e-12
    )
    assert not np.allclose(measured_density, response @ intrinsic)


def _synthetic_bi_kappa_spectra():
    c, e, me = 299792458.0, 1.602176634e-19, 9.1093837015e-31
    coordinate = np.linspace(-1.0, 1.0, 281)
    wavelength = 500.0 + 0.35 * (coordinate + 0.08 * coordinate**3)
    laser, angle, kappa = 500.0, np.pi/2, 4.2
    qmag = 2.0*np.sin(angle/2.0)
    b = np.array([0.0, 0.0, 1.0])
    pitch = np.array([0.0, 0.5, 1.0])
    q = np.column_stack((np.sqrt(1.0-pitch), np.zeros(3), np.sqrt(pitch)))
    tpar, tperp = 900.0, 320.0
    directional = tperp + (tpar-tperp)*pitch
    spectra = []
    for temp, flow in zip(directional, [1.2e5, -0.7e5, 0.4e5]):
        theta2 = ((2.0*kappa-3.0)/kappa)*(e*temp/me)
        theta = np.sqrt(theta2)
        feature, _ = _kappa_bin_average(wavelength, laser, qmag, flow, theta, kappa)
        spectra.append(2.5e3*feature + 3.0 + 0.2*(wavelength-laser))
    return wavelength, np.asarray(spectra), b, q


def test_gyrotropic_bi_kappa_fit_recovers_second_moment_temperatures_and_kappa():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    result = fit_gyrotropic_bi_kappa_thomson(
        wavelength, spectra, laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=750.0,
        initial_temperature_perpendicular_eV=400.0, initial_kappa=5.0,
    )
    assert result["reportable_as_inferred_parameter"]
    assert result["electron_temperature_parallel_eV"] == pytest.approx(900.0, rel=2e-3)
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(320.0, rel=2e-3)
    assert result["kappa_index"] == pytest.approx(4.2, rel=2e-3)
    assert result["model_validity"]["distribution"].startswith("gyrotropic bi-Kappa")
    assert result["parameter_covariance"].shape[0] == len(result["parameter_order"])
    assert np.all((result["captured_spectral_probability"] > 0.0) & (result["captured_spectral_probability"] <= 1.0))
    assert result["validity_screening"]["status"] == "within_declared_nonrelativistic_screen"
    tail = result["relativistic_tail_diagnostic"]
    assert tail["status"] == "below_projected_tail_tolerance"
    assert tail["requires_relativistic_forward_model"] is False
    assert tail["maximum_projected_superluminal_probability_lower_bound"] < 1.0e-8
    tail = result["relativistic_tail_diagnostic"]
    assert tail["status"] == "below_projected_tail_tolerance"
    assert tail["requires_relativistic_forward_model"] is False
    assert tail["maximum_projected_superluminal_probability_lower_bound"] < 1.0e-8
    recovered_probability = np.trapz(result["projected_velocity_distribution_m_s_inverse"], result["projected_velocity_grid_m_s"], axis=1)
    np.testing.assert_allclose(recovered_probability, result["captured_spectral_probability"], rtol=2e-3, atol=2e-4)


def test_gyrotropic_bi_kappa_does_not_report_anisotropy_from_one_pitch_angle():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    result = fit_gyrotropic_bi_kappa_thomson(
        wavelength, spectra[:1], laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q[:1], instrument_response_matrix=np.eye(wavelength.size),
    )
    assert result["anisotropy_identifiability"] == "non_identifiable"
    assert result["electron_temperature_parallel_eV"] is None
    assert result["electron_temperature_perpendicular_eV"] is None


def test_gyrotropic_bi_kappa_rejects_infinite_variance_domain():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    with pytest.raises(ValueError, match="kappa_bounds"):
        fit_gyrotropic_bi_kappa_thomson(
            wavelength, spectra, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
            scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
            initial_kappa=1.4,
        )


def test_gyrotropic_bi_kappa_rejects_fit_outside_nonrelativistic_speed_screen():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    with pytest.raises(ValueError, match="non-relativistic validity screen"):
        fit_gyrotropic_bi_kappa_thomson(
            wavelength, spectra, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
            scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
            maximum_projected_rms_beta=0.01,
        )


def test_gaussian_thomson_forward_model_integrates_exact_finite_wavelength_bins():
    from scipy.special import ndtr

    wavelength = np.array([499.70, 499.79, 499.94, 500.02, 500.21, 500.34])
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    centre, sigma = 500.0, 0.075

    density, probability = _gaussian_bin_average(wavelength, centre, sigma)
    expected = np.diff(ndtr((edges - centre) / sigma))

    np.testing.assert_allclose(probability, expected, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(density * np.diff(edges), expected, rtol=1e-13, atol=1e-15)
    assert np.all(density >= 0.0)
    assert np.sum(probability) < 1.0  # finite measured window; no tail renormalization


def test_kappa_thomson_forward_model_integrates_exact_student_t_bin_probabilities():
    from scipy.special import stdtr

    wavelength = np.array([499.70, 499.79, 499.94, 500.02, 500.21, 500.34])
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5 * (wavelength[:-1] + wavelength[1:])
    edges[0] = wavelength[0] - 0.5 * (wavelength[1] - wavelength[0])
    edges[-1] = wavelength[-1] + 0.5 * (wavelength[-1] - wavelength[-2])
    laser, angle, flow, theta, kappa = 500.0, np.pi / 2.0, 2.0e4, 1.5e6, 4.2
    q_factor = 2.0 * np.sin(angle / 2.0)
    density, probability = _kappa_bin_average(
        wavelength, laser, q_factor, flow, theta, kappa
    )
    velocity_edges = (299792458.0 / q_factor) * (1.0 - edges / laser) - flow
    dof = 2.0 * kappa - 1.0
    scale = theta * np.sqrt(kappa / dof)
    expected = (
        stdtr(dof, velocity_edges[:-1] / scale)
        - stdtr(dof, velocity_edges[1:] / scale)
    )

    np.testing.assert_allclose(probability, expected, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(density * np.diff(edges), expected, rtol=1e-12, atol=1e-14)
    assert np.all(probability >= 0.0)
    assert np.sum(probability) < 1.0


def test_bimaxwellian_rejects_invalid_nonrelativistic_validity_screen():
    wavelength, spectra, b, q = _synthetic_spectra()
    with pytest.raises(ValueError, match="maximum_projected_rms_beta"):
        fit_gyrotropic_bi_maxwellian_thomson(
            wavelength, spectra, laser_wavelength_nm=1064.0,
            scattering_angle_rad=np.pi / 2.0,
            magnetic_field_unit=b, scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            maximum_projected_rms_beta=1.0,
        )


def test_bimaxwellian_withholds_nonrelativistic_fit_outside_speed_screen():
    wavelength, spectra, b, q = _synthetic_spectra()
    with pytest.raises(ValueError, match="non-relativistic validity screen"):
        fit_gyrotropic_bi_maxwellian_thomson(
            wavelength, spectra, laser_wavelength_nm=1064.0,
            scattering_angle_rad=np.pi / 2.0,
            magnetic_field_unit=b, scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            maximum_projected_rms_beta=0.001,
        )


def test_bimaxwellian_model_discrepancy_is_not_added_to_a_synthetic_identity_covariance():
    wavelength, spectra, b, q = _synthetic_spectra()
    discrepancy = np.eye(spectra.size) * 2.5e-8
    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, spectra, laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        model_discrepancy_covariance=discrepancy,
        initial_temperature_parallel_eV=700.0,
        initial_temperature_perpendicular_eV=350.0,
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    assert result["measurement_covariance"] is None
    np.testing.assert_array_equal(result["model_discrepancy_covariance"], discrepancy)
    np.testing.assert_array_equal(result["effective_covariance"], discrepancy)
    assert result["covariance_provenance"]["effective_covariance_model"] == "model_discrepancy_only"
    assert result["covariance_provenance"]["parameter_covariance_scale"] == "absolute_model_discrepancy_covariance"


def test_bikappa_model_discrepancy_is_not_added_to_a_synthetic_identity_covariance():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    discrepancy = np.eye(spectra.size) * 1.0e-7
    result = fit_gyrotropic_bi_kappa_thomson(
        wavelength, spectra, laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q,
        instrument_response_matrix=np.eye(wavelength.size),
        model_discrepancy_covariance=discrepancy,
        initial_temperature_parallel_eV=750.0,
        initial_temperature_perpendicular_eV=400.0,
        initial_kappa=5.0,
    )
    np.testing.assert_array_equal(result["effective_covariance"], discrepancy)
    assert result["covariance_provenance"]["effective_covariance_model"] == "model_discrepancy_only"
    assert result["covariance_provenance"]["parameter_covariance_scale"] == "absolute_model_discrepancy_covariance"

def test_flow_projection_covariance_survives_unidentifiable_temperature_anisotropy():
    """Flow shifts remain estimable when every view has the same magnetic pitch."""
    wavelength = np.linspace(495.0, 505.0, 1001)
    laser, angle = 500.0, np.pi / 2.0
    b = np.array([0.0, 0.0, 1.0])
    pitch = 0.5
    transverse = np.sqrt(1.0 - pitch**2)
    q = np.array([
        [transverse, 0.0, pitch],
        [-0.5*transverse, np.sqrt(3.0)*0.5*transverse, pitch],
        [-0.5*transverse, -np.sqrt(3.0)*0.5*transverse, pitch],
    ])
    true_flow = np.array([1.2e4, -8.0e3, 5.0e3])
    projections = q @ true_flow
    directional_temperature = 0.875
    q_factor = 2.0*np.sin(angle/2.0)
    sigma_v = np.sqrt(1.602176634e-19*directional_temperature/9.1093837015e-31)
    spectra = []
    for flow in projections:
        centre = laser*(1.0-q_factor*flow/299792458.0)
        sigma_nm = laser*q_factor*sigma_v/299792458.0
        feature, _ = _gaussian_bin_average(wavelength, centre, sigma_nm)
        spectra.append(0.03 + 0.8*feature + 1.0e-4*(wavelength-laser))

    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, np.asarray(spectra), laser_wavelength_nm=laser,
        scattering_angle_rad=angle, magnetic_field_unit=b,
        scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=2.0,
        initial_temperature_perpendicular_eV=0.5,
        temperature_bounds_eV=(0.1, 5.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    assert result["geometry_design_rank"] == 1
    assert result["anisotropy_identifiability"] == "non_identifiable"
    assert result["parameter_covariance"] is None
    assert result["flow_projection_covariance_status"] == "identified_independently_of_temperature_anisotropy"
    assert result["flow_projection_covariance_m2_s2"].shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(result["flow_projection_covariance_m2_s2"]) > 0.0)
    np.testing.assert_allclose(result["electron_velocity_projection_m_s"], projections, atol=2.0e3)


def test_bikappa_exact_projected_tail_exposes_superluminal_newtonian_probability():
    from tokamak_analysis.thomson_anisotropic_inference import (
        _kappa_projected_superluminal_tail_probability,
    )

    c = 299792458.0
    cold_tail = _kappa_projected_superluminal_tail_probability(0.0, 0.01*c, 4.0)
    assert 0.0 <= cold_tail < 1.0e-8

    hot_tail = _kappa_projected_superluminal_tail_probability(0.0, 0.25*c, 4.0)
    assert hot_tail > cold_tail
    assert hot_tail > 1.0e-8

    shifted_tail = _kappa_projected_superluminal_tail_probability(0.2*c, 0.25*c, 4.0)
    assert 0.0 < shifted_tail <= 1.0

def test_maxwellian_projected_superluminal_tail_uses_stable_gaussian_survival_probability():
    from tokamak_analysis.thomson_anisotropic_inference import (
        _maxwellian_projected_superluminal_tail_probability,
    )

    c = 299792458.0
    cold = _maxwellian_projected_superluminal_tail_probability(0.0, 0.01*c)
    hot = _maxwellian_projected_superluminal_tail_probability(0.0, 0.25*c)
    shifted = _maxwellian_projected_superluminal_tail_probability(0.2*c, 0.25*c)
    assert cold == 0.0
    assert hot > cold
    assert shifted > hot

def test_bikappa_fit_withholds_physical_parameters_when_projected_tail_exceeds_tolerance():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    result = fit_gyrotropic_bi_kappa_thomson(
        wavelength, spectra, laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=750.0,
        initial_temperature_perpendicular_eV=400.0,
        initial_kappa=5.0,
        maximum_projected_superluminal_tail_probability=1.0e-14,
    )
    assert result["parameter_reporting_status"] == "model_outside_nonrelativistic_validity_domain"
    assert result["reportable_as_inferred_parameter"] is False
    assert result["electron_temperature_parallel_eV"] is None
    assert result["electron_temperature_perpendicular_eV"] is None
    assert result["kappa_index"] is None
    assert result["relativistic_tail_diagnostic"]["requires_relativistic_forward_model"] is True



def test_gyrotropic_velocity_covariance_tensor_matches_second_moment_physics():
    from tokamak_analysis.thomson_anisotropic_inference import (
        _gyrotropic_velocity_covariance_tensors,
    )

    electron_charge = 1.602176634e-19
    electron_mass = 9.1093837015e-31
    t_parallel, t_perpendicular = 900.0, 250.0
    field = np.array([
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
    ])
    tensors = _gyrotropic_velocity_covariance_tensors(
        t_parallel, t_perpendicular, field
    )
    expected_eigenvalues = np.array([
        electron_charge * t_perpendicular / electron_mass,
        electron_charge * t_perpendicular / electron_mass,
        electron_charge * t_parallel / electron_mass,
    ])
    assert tensors.shape == (2, 3, 3)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(tensors[0]), expected_eigenvalues, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(tensors[1]), expected_eigenvalues, rtol=1e-12
    )
    np.testing.assert_allclose(tensors[0][0, 0], tensors[1][2, 2], rtol=1e-12)
    np.testing.assert_allclose(tensors[0][2, 2], tensors[1][0, 0], rtol=1e-12)


def test_gyrotropic_velocity_covariance_is_isotropic_when_temperatures_match():
    from tokamak_analysis.thomson_anisotropic_inference import (
        _gyrotropic_velocity_covariance_tensors,
    )

    temperature = 400.0
    tensors = _gyrotropic_velocity_covariance_tensors(
        temperature, temperature,
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
    )
    expected = (1.602176634e-19 * temperature / 9.1093837015e-31) * np.eye(3)
    np.testing.assert_allclose(tensors[0], expected, rtol=1e-12)
    np.testing.assert_allclose(tensors[1], expected, rtol=1e-12)


def test_public_thomson_spectral_fit_routes_geometry_uncertainty_to_3d_flow_closure():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    coordinate = np.linspace(-1.0, 1.0, 161)
    wavelength = 1064.0 + 2.0 * (coordinate + 0.12 * coordinate**3)
    laser, angle = 1064.0, np.pi / 2.0
    magnetic_field = np.array([0.0, 0.0, 1.0])
    directions = np.eye(3)
    common_flow = np.array([1.2e4, -0.8e4, 0.5e4])
    flows = directions @ common_flow
    temperatures = np.array([250.0, 250.0, 900.0])
    spectra = []
    for temperature, flow in zip(temperatures, flows):
        sigma_v = np.sqrt(1.602176634e-19 * temperature / 9.1093837015e-31)
        centre = laser * (1.0 - 2.0*np.sin(angle/2.0)*flow/299792458.0)
        sigma_lambda = laser * 2.0*np.sin(angle/2.0)*sigma_v/299792458.0
        feature, _ = _gaussian_bin_average(wavelength, centre, sigma_lambda)
        spectra.append(0.03 + 0.8*feature + 1.0e-4*(wavelength-laser))

    result = run_advanced_thomson_physics(
        "gyrotropic_bi_maxwellian_spectral_fit",
        wavelength,
        np.asarray(spectra),
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        magnetic_field_unit=magnetic_field,
        scattering_vector_unit=directions,
        instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=800.0,
        initial_temperature_perpendicular_eV=300.0,
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
        scattering_vector_covariance=np.eye(9)*1.0e-8,
        maximum_geometry_condition_number=1.0e3,
        minimum_projection_consistency_p_value=1.0e-3,
    )
    flow_result = result["three_dimensional_flow_inference"]
    assert flow_result["geometry_uncertainty_included"] is True
    assert flow_result["geometry_condition_status"] == "well_conditioned"
    assert flow_result["cross_view_consistency_status"] == "not_testable_exactly_three_views"
    assert flow_result["reportable_as_inferred_parameter"] is True
    np.testing.assert_allclose(
        flow_result["electron_velocity_vector_m_s"],
        common_flow,
        rtol=2.0e-2,
        atol=2.0e3,
    )


def test_canonical_gyrotropic_workflow_preserves_3d_flow_geometry_options():
    from tokamak_analysis.thomson_workflow import fit_gyrotropic_thomson_spectra

    coordinate = np.linspace(-1.0, 1.0, 121)
    wavelength = 1064.0 + 1.8 * (coordinate + 0.1*coordinate**3)
    directions = np.eye(3)
    magnetic_field = np.array([0.0, 0.0, 1.0])
    common_flow = np.array([8.0e3, -6.0e3, 1.0e4])
    temperatures = [300.0, 300.0, 850.0]
    spectra = []
    for q, temperature in zip(directions, temperatures):
        flow = float(q @ common_flow)
        sigma_v = np.sqrt(1.602176634e-19*temperature/9.1093837015e-31)
        centre = 1064.0*(1.0-2.0*flow/299792458.0)
        sigma_lambda = 1064.0*2.0*sigma_v/299792458.0
        feature, _ = _gaussian_bin_average(wavelength, centre, sigma_lambda)
        spectra.append(0.02+0.9*feature)

    result = fit_gyrotropic_thomson_spectra(
        wavelength,
        np.asarray(spectra),
        distribution_model="bi_maxwellian",
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2.0,
        magnetic_field_unit=magnetic_field,
        scattering_vector_unit=directions,
        instrument_response_matrix=np.eye(wavelength.size),
        initial_temperature_parallel_eV=700.0,
        initial_temperature_perpendicular_eV=350.0,
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
        scattering_vector_covariance=np.eye(9)*1.0e-8,
        maximum_geometry_condition_number=1.0e3,
        minimum_projection_consistency_p_value=1.0e-3,
    )
    assert result["analysis_entry_point"] == "fit_gyrotropic_thomson_spectra"
    assert result["three_dimensional_flow_inference"]["geometry_uncertainty_included"] is True
    assert result["three_dimensional_flow_inference"]["reportable_as_inferred_parameter"] is True
    np.testing.assert_allclose(
        result["three_dimensional_flow_inference"]["electron_velocity_vector_m_s"],
        common_flow,
        rtol=3.0e-2,
        atol=2.0e3,
    )


def test_bikappa_response_normalization_is_checked_independently_for_each_view():
    from tokamak_analysis.thomson_anisotropic_inference import (
        fit_gyrotropic_bi_kappa_thomson,
    )

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    response = np.broadcast_to(
        np.eye(wavelength.size),
        (spectra.shape[0], wavelength.size, wavelength.size),
    ).copy()
    # A pooled sum over views is exactly identity, but neither diagnostic is
    # probability-conserving: each true wavelength bin loses 60%/40% of its
    # photons. The old axis=0 check incorrectly accepted this pair.
    response[0] *= 0.4
    response[1] *= 0.6
    with pytest.raises(ValueError, match="independently per view"):
        fit_gyrotropic_bi_kappa_thomson(
            wavelength, spectra[:2], laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
            scattering_vector_unit=q[:2],
            instrument_response_matrix=response[:2],
        )


def test_bikappa_accepts_distinct_probability_conserving_response_per_view():
    from tokamak_analysis.thomson_anisotropic_inference import (
        fit_gyrotropic_bi_kappa_thomson,
        _apply_response_in_bin_integrated_space,
    )

    wavelength, intrinsic_spectra, b, q = _synthetic_bi_kappa_spectra()
    n = wavelength.size
    identity = np.eye(n)
    # A non-symmetric, finite-volume column-stochastic redistribution kernel.
    blurred = 0.90 * identity
    blurred[1:, :-1] += 0.10 * np.eye(n - 1)
    blurred[-1, -1] = 1.0  # retain all probability in the final input bin
    response = np.stack((identity, blurred, identity))
    measured = intrinsic_spectra.copy()
    measured[1] = _apply_response_in_bin_integrated_space(
        blurred, intrinsic_spectra[1] - (3.0 + 0.2*(wavelength-500.0)), wavelength
    ) + 3.0 + 0.2*(wavelength-500.0)

    result = fit_gyrotropic_bi_kappa_thomson(
        wavelength, measured, laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q, instrument_response_matrix=response,
        initial_temperature_parallel_eV=750.0,
        initial_temperature_perpendicular_eV=400.0,
        initial_kappa=5.0,
    )
    assert result["reportable_as_inferred_parameter"] is True
    assert result["electron_temperature_parallel_eV"] == pytest.approx(900.0, rel=3e-2)
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(320.0, rel=3e-2)
    assert result["kappa_index"] == pytest.approx(4.2, rel=3e-2)


def test_bikappa_response_conservation_reaches_public_api_and_dashboard_model():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    from tis.ui.advanced_thomson_physics_page import MODEL_LABELS, EXAMPLES

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    response = np.broadcast_to(
        np.eye(wavelength.size),
        (spectra.shape[0], wavelength.size, wavelength.size),
    ).copy()
    model = "gyrotropic_bi_kappa_spectral_fit"
    assert model in MODEL_LABELS
    assert model in EXAMPLES

    result = run_advanced_thomson_physics(
        model,
        wavelength,
        spectra,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2,
        magnetic_field_unit=b,
        scattering_vector_unit=q,
        instrument_response_matrix=response,
        initial_temperature_parallel_eV=750.0,
        initial_temperature_perpendicular_eV=400.0,
        initial_kappa=5.0,
        systematic_covariance=np.eye(spectra.size) * 2.0e-8,
    )
    assert result["systematic_covariance"].shape == (spectra.size, spectra.size)
    assert result["covariance_provenance"]["systematic_supplied"] is True
    assert result["public_physics_entrypoint"] == model
    assert result["automatic_model_substitution"] is False
    assert result["reportable_as_inferred_parameter"] is True
    assert result["electron_temperature_parallel_eV"] == pytest.approx(900.0, rel=3e-2)
    assert result["electron_temperature_perpendicular_eV"] == pytest.approx(320.0, rel=3e-2)
    assert result["kappa_index"] == pytest.approx(4.2, rel=3e-2)


def test_bikappa_gls_adds_measurement_discrepancy_and_systematic_covariance():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    size = spectra.size
    measurement = np.eye(size) * 4.0e-8
    discrepancy = np.eye(size) * 1.0e-8
    systematic = np.eye(size) * 2.0e-8
    result = fit_gyrotropic_bi_kappa_thomson(
        wavelength, spectra, laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
        scattering_vector_unit=q, instrument_response_matrix=np.eye(wavelength.size),
        measurement_covariance=measurement,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
        initial_temperature_parallel_eV=750.0,
        initial_temperature_perpendicular_eV=400.0,
        initial_kappa=5.0,
    )
    np.testing.assert_allclose(
        result["effective_covariance"], measurement + discrepancy + systematic,
        rtol=0.0, atol=1.0e-20,
    )
    np.testing.assert_array_equal(result["systematic_covariance"], systematic)
    assert result["covariance_provenance"]["systematic_supplied"] is True
    assert result["covariance_provenance"]["effective_covariance_model"] == (
        "measurement_plus_model_discrepancy_plus_systematic"
    )
    assert result["covariance_provenance"]["parameter_covariance_scale"] == (
        "supplied_absolute_covariance"
    )
    assert result["reportable_as_inferred_parameter"] is True


def test_bikappa_gls_rejects_non_psd_systematic_covariance():
    from tokamak_analysis.thomson_anisotropic_inference import fit_gyrotropic_bi_kappa_thomson

    wavelength, spectra, b, q = _synthetic_bi_kappa_spectra()
    invalid = np.eye(spectra.size)
    invalid[0, 0] = -1.0
    with pytest.raises(ValueError, match="systematic_covariance must be positive semidefinite"):
        fit_gyrotropic_bi_kappa_thomson(
            wavelength, spectra, laser_wavelength_nm=500.0,
            scattering_angle_rad=np.pi/2, magnetic_field_unit=b,
            scattering_vector_unit=q,
            instrument_response_matrix=np.eye(wavelength.size),
            systematic_covariance=invalid,
        )

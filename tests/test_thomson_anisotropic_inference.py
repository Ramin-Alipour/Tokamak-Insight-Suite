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
    assert result["electron_velocity_projection_m_s"] == pytest.approx([1.0e4, -1.5e4, 2.0e4], abs=2.0e3)
    assert np.max(np.abs(result["residual_spectra"])) < 1e-5
    assert result["model_validity"]["relativistic_corrections"] is False


def test_single_pitch_angle_does_not_report_separate_anisotropy_temperatures():
    wavelength, spectra, b, q = _synthetic_spectra()
    result = fit_gyrotropic_bi_maxwellian_thomson(
        wavelength, spectra[:1], laser_wavelength_nm=1064.0,
        scattering_angle_rad=np.pi/2,
        magnetic_field_unit=b, scattering_vector_unit=q[:1],
        instrument_response_matrix=np.eye(wavelength.size),
        temperature_bounds_eV=(20.0, 3000.0),
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

import numpy as np
import pytest

from tokamak_analysis.advanced_inference import (
    build_relativistic_3d_thomson_kernels,
    reconstruct_relativistic_optical_thomson_3d_evdf,
)


def _view_geometry(qhat):
    qhat = np.asarray(qhat, dtype=float)
    qhat = qhat / np.linalg.norm(qhat)
    reference = np.array([0.0, 0.0, 1.0]) if abs(qhat[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    transverse = np.cross(qhat, reference)
    transverse /= np.linalg.norm(transverse)
    return (-qhat + transverse) / np.sqrt(2.0), (qhat + transverse) / np.sqrt(2.0)


def test_zero_velocity_thomson_limit_recovers_unpolarized_angular_cross_section():
    wavelength = np.linspace(499.0, 501.0, 201)
    velocity = np.zeros((1, 3))
    incident = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    angles = np.array([np.pi / 2.0, np.pi / 3.0])
    scattered = np.array([
        [np.cos(angle), np.sin(angle), 0.0] for angle in angles
    ])
    result = build_relativistic_3d_thomson_kernels(
        wavelength, velocity, laser_wavelength_nm=500.0,
        incident_direction=incident, scattered_direction=scattered,
        view_gain=np.ones(2), instrument_response_matrix=np.eye(wavelength.size),
    )
    np.testing.assert_allclose(result["line_centres_nm"], 500.0, rtol=0.0, atol=1e-12)
    expected = 0.5 * (1.0 + np.cos(angles)**2)
    np.testing.assert_allclose(result["relative_lab_differential_cross_section"][:, 0], expected, rtol=2e-13)
    np.testing.assert_allclose(result["scattered_photon_energy_over_incident"][:, 0], 1.0, atol=1e-14)
    widths = result["wavelength_bin_widths_nm"]
    integrated = np.sum(result["spectral_kernels"][:, :, 0] * widths[None, :], axis=1)
    np.testing.assert_allclose(integrated, expected, rtol=2e-13)


def test_rest_electron_compton_kernel_recovers_exact_compton_wavelength_shift():
    wavelength = np.linspace(499.0, 501.0, 401)
    incident = np.array([[1.0, 0.0, 0.0]])
    theta = np.pi / 2.0
    scattered = np.array([[np.cos(theta), np.sin(theta), 0.0]])
    result = build_relativistic_3d_thomson_kernels(
        wavelength, np.zeros((1, 3)), laser_wavelength_nm=500.0,
        incident_direction=incident, scattered_direction=scattered,
        view_gain=np.ones(1), instrument_response_matrix=np.eye(wavelength.size),
        include_compton_recoil=True,
    )
    compton_wavelength_nm = (
        6.62607015e-34 / (9.1093837015e-31 * 299792458.0) * 1.0e9
    )
    expected = 500.0 + compton_wavelength_nm * (1.0 - np.cos(theta))
    assert result["line_centres_nm"][0, 0] == pytest.approx(expected, rel=2e-13)
    assert result["scattered_photon_energy_over_incident"][0, 0] == pytest.approx(500.0 / expected, rel=2e-13)
    assert result["rest_frame_compton_recoil_fraction"][0, 0] > 0.0


def test_exact_relativistic_thomson_doppler_map_has_correct_first_order_q_projection_limit():
    wavelength = np.linspace(499.0, 501.0, 201)
    incident = np.array([[1.0, 0.0, 0.0]])
    scattered = np.array([[0.0, 1.0, 0.0]])
    velocity = np.array([[1.0e4, 2.0e4, -3.0e3]])
    result = build_relativistic_3d_thomson_kernels(
        wavelength, velocity, laser_wavelength_nm=500.0,
        incident_direction=incident, scattered_direction=scattered,
        view_gain=np.ones(1), instrument_response_matrix=np.eye(wavelength.size),
    )
    beta = velocity[0] / 299792458.0
    q = scattered[0] - incident[0]
    first_order = 500.0 * (1.0 - beta @ q)
    assert result["line_centres_nm"][0, 0] == pytest.approx(first_order, abs=2e-4)


def test_relativistic_3d_evdf_inverse_uses_generated_per_view_physical_kernels():
    wavelength = np.linspace(499.85, 500.15, 121)
    velocity = np.array([[0.0, 0.0, 0.0], [1.0e4, 0.0, 0.0]])
    q_directions = np.eye(3)
    geometry = [_view_geometry(q) for q in q_directions]
    incident = np.asarray([item[0] for item in geometry])
    scattered = np.asarray([item[1] for item in geometry])
    background = np.full((3, wavelength.size), 0.05)
    response = np.eye(wavelength.size)
    volume = np.ones(2)
    seed = np.array([0.7, 0.3])
    kernels = build_relativistic_3d_thomson_kernels(
        wavelength, velocity, laser_wavelength_nm=500.0,
        incident_direction=incident, scattered_direction=scattered,
        view_gain=np.ones(3), instrument_response_matrix=response,
    )["spectral_kernels"]
    measured = kernels @ (seed * volume) + background
    result = reconstruct_relativistic_optical_thomson_3d_evdf(
        wavelength, measured, incident, scattered, velocity,
        laser_wavelength_nm=500.0, view_gain=np.ones(3),
        instrument_response_matrix=response, background_spectra=background,
        voxel_volume_m3_s3=volume, regularization_matrix=np.array([[-1.0, 1.0]]),
        regularization_strength=1.0e-8,
        measurement_uncertainty=np.full_like(measured, 0.01),
    )
    assert result["physical_forward_model"] == "general_3d_relativistic_unpolarized_thomson_with_optional_compton_recoil"
    assert result["status"] == "identified_from_multi_view_measurements"
    assert result["evdf_m3_s3"] is not None
    assert result["data_only_jacobian_rank"] == 2
    assert result["physical_spectral_kernels"].shape == (3, wavelength.size, 2)
    np.testing.assert_allclose(np.sum(result["evdf_m3_s3"] * volume), 1.0, rtol=1e-12)


def test_relativistic_3d_evdf_public_dispatch_and_dashboard_route():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS

    model = "thomson_relativistic_3d_evdf"
    assert model in MODEL_LABELS
    assert model in EXAMPLES
    example = EXAMPLES[model]
    result = run_advanced_thomson_physics(model, **example)
    assert result["public_physics_entrypoint"] == model
    assert result["validity_domain"]["relativistic_doppler_kinematics_included"] is True
    assert result["validity_domain"]["polarization_resolved"] is False

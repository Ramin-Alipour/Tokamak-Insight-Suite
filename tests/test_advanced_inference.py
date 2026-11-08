import numpy as np
import pytest

from tokamak_analysis.advanced_inference import (
    poisson_deviance_residual,
    poisson_negative_log_likelihood,
    gaussian_process_discrepancy_covariance,
    propagate_paraxial_beam_covariance,
    bootstrap_reference_pca_basis,
    reconstruct_multiview_3d_evdf,
    reconstruct_optical_thomson_3d_evdf,
    assess_multiview_evdf_regularization_sensitivity,
)
from tokamak_analysis.cts_em_background import (
    planck_spectral_radiance,
    thermal_bremsstrahlung_emissivity_si,
    integrate_thermal_microwave_background,
    radiance_to_receiver_spectral_power,
)


def test_poisson_likelihood_is_independent_count_model():
    y = np.array([0, 2, 10, 100])
    mu = np.array([0.5, 2.5, 9.0, 110.0])
    residual = poisson_deviance_residual(y, mu)
    assert residual.shape == y.shape
    assert residual[0] < 0 and residual[-1] < 0
    assert np.sum(residual**2) >= 0
    assert poisson_negative_log_likelihood(y, mu) > 0
    with pytest.raises(ValueError, match="integer-valued"):
        poisson_deviance_residual([0.2, 1.0], [1.0, 1.0])


def test_gp_discrepancy_is_correlated_and_positive_definite():
    x = np.linspace(-1, 1, 31)
    cov = gaussian_process_discrepancy_covariance(x, amplitude=0.2, length_scale=0.3)
    assert cov.shape == (x.size, x.size)
    assert np.allclose(cov, cov.T)
    assert np.linalg.eigvalsh(cov).min() > -1e-10
    assert cov[0, 1] > cov[0, -1]


def test_paraxial_beam_covariance_propagates_symplectically():
    sigma = np.diag([1e-4, 2e-4, 1e-3, 2e-3])
    free_space = np.array([[1, 0, 0.2, 0], [0, 1, 0, 0.2],
                           [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)
    result = propagate_paraxial_beam_covariance(sigma, [free_space, free_space])
    assert result["phase_space_covariance"].shape == (3, 4, 4)
    assert result["transverse_position_covariance_m2"].shape == (3, 2, 2)
    assert np.all(np.linalg.eigvalsh(result["phase_space_covariance"][-1]) > 0)


def test_reference_pca_bootstrap_reports_basis_uncertainty():
    rng = np.random.default_rng(4)
    x = np.linspace(-1, 1, 41)
    refs = np.asarray([3 + a*x + b*x*x + rng.normal(0, 0.01, x.size)
                       for a, b in zip(np.linspace(-1, 1, 12), np.linspace(0.2, -0.2, 12))])
    result = bootstrap_reference_pca_basis(refs, components=2, bootstrap_samples=40)
    assert result["bootstrap_parameter_covariance"].shape == (refs.shape[1] * 3, refs.shape[1] * 3)
    assert result["pca_components"].shape == (refs.shape[1], 2)


def test_multiview_3d_evdf_requires_data_supported_rank():
    # Identity spectral kernels make all six velocity voxels independently identifiable.
    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    true = np.array([1., 2., 3., 4., 5., 6.])
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (true * volume)).reshape(3, 2)
    penalty = np.zeros((1, 6))
    result = reconstruct_multiview_3d_evdf(
        kernels, data, np.arange(18).reshape(6, 3), voxel_volume_m3_s3=volume,
        regularization_matrix=penalty, regularization_strength=1e-6,
    )
    assert result["status"] == "identified_from_multi_view_measurements"
    assert result["evdf_m3_s3"] is not None
    assert result["mean_velocity_vector_m_s"].shape == (3,)
    assert result["velocity_covariance_tensor_m2_s2"].shape == (3, 3)
    assert result["mean_velocity_covariance_m2_s2"].shape == (3, 3)
    assert result["velocity_covariance_tensor_covariance_m4_s4"].shape == (9, 9)
    expected_probability = true * volume / np.sum(true * volume)
    expected_mean = expected_probability @ np.arange(18).reshape(6, 3)
    centered = np.arange(18).reshape(6, 3) - expected_mean
    expected_covariance = np.einsum(
        "i,ij,ik->jk", expected_probability, centered, centered
    )
    np.testing.assert_allclose(result["mean_velocity_vector_m_s"], expected_mean, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result["velocity_covariance_tensor_m2_s2"], expected_covariance, rtol=1e-6, atol=1e-6)
    assert result["mean_velocity_covariance_m2_s2"].shape == (3, 3)
    assert result["velocity_covariance_tensor_covariance_m4_s4"].shape == (9, 9)
    assert result["velocity_moment_status"] == "identified_from_multi_view_measurements"
    one_view = reconstruct_multiview_3d_evdf(
        kernels[:1], data[:1], np.arange(18).reshape(6, 3),
        voxel_volume_m3_s3=volume, regularization_matrix=penalty,
        regularization_strength=1e-6,
    )
    assert one_view["status"] == "conditional_regularized_3d_candidate"
    assert one_view["evdf_m3_s3"] is None
    assert one_view["mean_velocity_vector_m_s"] is None
    assert one_view["candidate_mean_velocity_vector_m_s"].shape == (3,)
    assert one_view["velocity_moment_status"] == "conditional_candidate_only_velocity_moments_not_identified"



def test_multiview_3d_evdf_rejects_superluminal_velocity_voxels():
    kernels = np.ones((2, 5, 2))
    data = np.ones((2, 5))
    with pytest.raises(ValueError, match="strictly subluminal"):
        reconstruct_multiview_3d_evdf(
            kernels, data,
            np.array([[0.0, 0.0, 0.0], [299792458.0, 0.0, 0.0]]),
            voxel_volume_m3_s3=np.ones(2),
            regularization_matrix=np.eye(2),
            regularization_strength=1e-3,
        )


def test_free_free_and_ece_radiative_transfer():
    nu = np.linspace(90e9, 110e9, 5)
    b = planck_spectral_radiance(nu, 2e7)
    assert np.all(b > 0)
    j = thermal_bremsstrahlung_emissivity_si(
        nu, 2e19, [1e19], [1], 1500.0, gaunt_factor=1.2
    )
    assert np.all(j > 0)
    result = integrate_thermal_microwave_background(
        nu, [0.1, 0.2], [1000., 1200.], electron_density_m3=[2e19, 2e19],
        ion_densities_m3=[[1e19], [1e19]], ion_charge_states=[1],
        ece_absorption_coefficient_m_inverse=np.full((2, 5), 0.01),
    )
    assert result["spectral_radiance_W_m2_sr_Hz"].shape == nu.shape
    assert np.all(result["ece_optical_depth"] > 0)



def test_nonlinear_nonnegative_distribution_fit_uses_full_forward_callback():
    from tokamak_analysis.advanced_inference import fit_nonlinear_nonnegative_distribution

    rng = np.random.default_rng(12)
    kernel = rng.uniform(0.2, 1.2, size=(24, 6))
    weights = np.linspace(0.5, 1.0, 6)
    truth = np.array([0.05, 0.1, 0.2, 0.25, 0.25, 0.15])
    truth /= weights @ truth
    measured = 3.0 * (kernel @ truth) + 0.1
    initial = np.ones(6) / (weights @ np.ones(6))
    regularizer = np.diff(np.eye(6), n=2, axis=0)
    result = fit_nonlinear_nonnegative_distribution(
        measured, initial, weights,
        forward_model=lambda pdf: kernel @ pdf,
        regularization_matrix=regularizer,
        regularization_strength=1e-5,
        measurement_uncertainty=np.full(measured.size, 0.01),
        max_nfev=500,
    )
    assert result["status"] == "identified_conditional_distribution_shape"
    assert result["distribution"] is not None
    assert np.all(result["distribution"] >= 0)
    assert result["normalization"] == pytest.approx(1.0, rel=1e-8)



def test_optical_maxwellian_fit_accepts_independent_poisson_count_likelihood():
    from tokamak_analysis.thomson_inference import fit_optical_thomson_poisson_counts

    wavelength = np.linspace(530.0, 532.0, 121)
    center = 531.0
    sigma = 0.18
    rng = np.random.default_rng(21)
    expected = 450.0*np.exp(-0.5*((wavelength-center)/sigma)**2) + 18.0 + 2.0*(wavelength-center)
    counts = rng.poisson(np.maximum(expected, 1.0)).astype(int)
    result = fit_optical_thomson_poisson_counts(
        wavelength, counts, laser_wavelength_nm=531.0,
        scattering_angle_rad=np.pi/2.0,
        instrument_response=np.eye(wavelength.size),
    )
    assert result["likelihood"] == "independent_Poisson"
    assert result["density_inference_available"] is False
    assert "integrated_scattered_power_W" not in result
    assert result["integrated_signal_counts"] > 0
    assert np.isfinite(result["electron_temperature_eV"])
    assert result["reduced_chi_square"] is None


def test_cts_accepts_independent_correlated_model_discrepancy_covariance():
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum, cts_spectral_density
    frequency = np.linspace(-5.0e9, 5.0e9, 81)
    f0 = 140e9
    geometry = {"scattering_angle_rad": np.deg2rad(110.0)}
    species = [{"name": "D+", "charge_state": 1, "mass_amu": 2.014}]
    truth = dict(
        electron_density_m3=2.0e19, electron_temperature_eV=1200.0,
        ion_temperature_eV=800.0, electron_velocity_projection_m_s=0.0,
        ion_velocity_projection_m_s=1.5e4,
    )
    physical = cts_spectral_density(
        frequency, f0, geometry, **truth, ion_species=species, ion_fractions=[1.0]
    )
    measured = 2.4e4*physical + 1.0
    discrepancy = np.eye(frequency.size) * (1e-3*np.max(measured))**2
    result = analyze_cts_spectrum(
        frequency, measured, probe_frequency_Hz=f0,
        scattering_geometry=geometry, ion_species=species, ion_fractions=[1.0],
        instrument_response=np.eye(frequency.size),
        electron_density_initial_m3=truth["electron_density_m3"],
        electron_temperature_initial_eV=truth["electron_temperature_eV"],
        ion_temperature_initial_eV=truth["ion_temperature_eV"],
        ion_velocity_initial_m_s=truth["ion_velocity_projection_m_s"],
        model_discrepancy_covariance=discrepancy,
        global_search=False, inversion_method="local_least_squares",
    )
    assert result["independent_model_discrepancy_supplied"] is True
    assert result["model_discrepancy_model"].startswith("researcher_supplied_correlated")
    assert "model_discrepancy_scale" not in result



def test_radiance_to_receiver_power_applies_etendue_and_efficiency_once():
    radiance = np.array([1.0, 2.0, 3.0])
    power = radiance_to_receiver_spectral_power(
        radiance, receiver_etendue_m2_sr=0.02,
        end_to_end_efficiency=0.5,
    )
    np.testing.assert_allclose(power, radiance*0.01)



def test_optical_thomson_uses_independent_model_discrepancy_covariance():
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra

    wavelength = np.linspace(530.0, 532.0, 121)
    expected = 1.2e-3*np.exp(-0.5*((wavelength-531.0)/0.18)**2) + 3.0e-5
    discrepancy = gaussian_process_discrepancy_covariance(
        wavelength, amplitude=2.0e-5, length_scale=0.12,
        kernel="matern32", nugget=1.0e-6,
    )
    result = analyze_thomson_spectra(
        wavelength, expected[None, :], np.array([0.5]),
        laser_wavelength_nm=531.0, scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(wavelength.size),
        model="non_collective_maxwellian",
        model_discrepancy_covariance=discrepancy,
        background_model="linear",
    )
    channel = result["channels"][0]
    assert channel["independent_model_discrepancy_supplied"] is True
    assert channel["model_discrepancy_diagnostics"]["covariance_included"] is True
    assert channel["model_discrepancy_diagnostics"]["independent_of_reduced_chi_square"] is True
    assert "model_discrepancy_scale" not in channel



def test_multiview_evdf_regularization_sensitivity_reports_l_curve_without_auto_selection():
    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    truth = np.array([1., 2., 3., 4., 5., 6.])
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (truth * volume)).reshape(3, 2)
    penalty = np.diff(np.eye(6), axis=0)
    result = assess_multiview_evdf_regularization_sensitivity(
        kernels, data, np.arange(18).reshape(6, 3),
        voxel_volume_m3_s3=volume,
        regularization_matrix=penalty,
        regularization_strengths=np.array([1e-8, 1e-5, 1e-2]),
    )
    assert len(result["levels"]) == 3
    assert result["candidate_distributions_m3_s3"].shape == (3, 6)
    assert result["selection_policy"].startswith("no automatic lambda selection")
    assert all(level["data_only_jacobian_rank"] == 6 for level in result["levels"])
    assert all(level["null_space_dimension"] == 0 for level in result["levels"])


def test_3d_evdf_covariance_ensemble_is_executable_through_public_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    truth = np.arange(1.0, 7.0)
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (truth * volume)).reshape(3, 2)
    result = run_advanced_thomson_physics(
        "thomson_multiview_3d_evdf_regularization_sensitivity",
        kernels, data, np.arange(18, dtype=float).reshape(6, 3),
        voxel_volume_m3_s3=volume,
        regularization_matrix=np.diff(np.eye(6), axis=0),
        regularization_strengths=[1e-8, 1e-5, 1e-2],
        measurement_uncertainty=np.full((3, 2), 0.01),
        model_discrepancy_covariance=np.eye(6) * 2e-5,
        systematic_covariance=np.eye(6) * 3e-5,
    )
    assert result["covariance_components_included"] == {
        "measurement": True, "model_discrepancy": True, "systematic": True,
    }
    assert result["regularization_selection_uncertainty"][
        "physical_moment_covariance_moment_units_squared"
    ].shape == (12, 12)


def test_multiview_3d_evdf_rejects_tiny_scale_indefinite_covariance():
    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    truth = np.arange(1.0, 7.0)
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (truth * volume)).reshape(3, 2)
    covariance = np.eye(6) * 1.0e-20
    covariance[0, 0] = -1.0e-22
    with pytest.raises(ValueError, match="positive semidefinite"):
        reconstruct_multiview_3d_evdf(
            kernels, data, np.arange(18, dtype=float).reshape(6, 3),
            voxel_volume_m3_s3=volume,
            regularization_matrix=np.diff(np.eye(6), axis=0),
            measurement_covariance=covariance,
            regularization_strength=1.0e-4,
        )


def test_multiview_evdf_regularization_ensemble_propagates_all_spectral_covariance_components():
    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    truth = np.array([1., 2., 3., 4., 5., 6.])
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (truth * volume)).reshape(3, 2)
    measurement = np.eye(6) * 1.0e-4
    discrepancy = np.eye(6) * 2.0e-5
    discrepancy[0, 1] = discrepancy[1, 0] = 2.0e-6
    systematic = np.eye(6) * 3.0e-5
    result = assess_multiview_evdf_regularization_sensitivity(
        kernels, data, np.arange(18, dtype=float).reshape(6, 3),
        voxel_volume_m3_s3=volume,
        regularization_matrix=np.diff(np.eye(6), axis=0),
        regularization_strengths=np.array([1e-8, 1e-5, 1e-2]),
        measurement_covariance=measurement,
        model_discrepancy_covariance=discrepancy,
        systematic_covariance=systematic,
    )
    assert result["covariance_components_included"] == {
        "measurement": True, "model_discrepancy": True, "systematic": True,
    }
    assert result["noise_model"] == "measurement_plus_model_discrepancy_plus_systematic_covariance"
    assert np.allclose(
        result["effective_spectral_covariance"],
        measurement + discrepancy + systematic,
    )
    assert result["regularization_selection_uncertainty"]["physical_moment_covariance_moment_units_squared"].shape == (12, 12)


def test_multiview_evdf_regularization_spread_propagates_to_3d_physical_moments():
    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    truth = np.array([1., 2., 3., 4., 5., 6.])
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (truth * volume)).reshape(3, 2)
    velocity = np.array([
        [-3., 0., 0.], [-2., 1., 0.], [-1., 0., 1.],
        [1., -1., 0.], [2., 0., -1.], [3., 1., 1.],
    ])
    result = assess_multiview_evdf_regularization_sensitivity(
        kernels, data, velocity,
        voxel_volume_m3_s3=volume,
        regularization_matrix=np.diff(np.eye(6), axis=0),
        regularization_strengths=np.array([1e-8, 1e-5, 1e-2]),
    )
    spread = result["regularization_selection_uncertainty"]
    assert spread["candidate_count"] == 3
    assert spread["distribution_covariance_m6_s6"].shape == (6, 6)
    assert spread["physical_moment_candidate_values"].shape == (3, 12)
    covariance = spread["physical_moment_covariance_moment_units_squared"]
    assert covariance.shape == (12, 12)
    assert np.allclose(covariance, covariance.T, atol=1e-12)
    assert np.linalg.eigvalsh(covariance).min() >= -1e-10
    assert "not measurement uncertainty" in spread["interpretation"]


def test_multiview_3d_evdf_is_available_through_public_thomson_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    kernels = np.zeros((3, 2, 6))
    for view in range(3):
        kernels[view, :, 2*view:2*view+2] = np.eye(2)
    truth = np.array([1., 2., 3., 4., 5., 6.])
    volume = np.full(6, 0.1)
    data = (kernels.reshape(6, 6) @ (truth * volume)).reshape(3, 2)
    result = run_advanced_thomson_physics(
        "thomson_multiview_3d_evdf",
        spectral_kernels=kernels,
        measured_spectra=data,
        velocity_grid_m_s=np.arange(18).reshape(6, 3),
        voxel_volume_m3_s3=volume,
        regularization_matrix=np.zeros((1, 6)),
        regularization_strength=1e-6,
    )
    assert result["public_physics_entrypoint"] == "thomson_multiview_3d_evdf"
    assert result["automatic_model_substitution"] is False
    assert result["status"] == "identified_from_multi_view_measurements"
    assert result["evdf_m3_s3"] is not None
    assert result["view_count"] == 3
    expected_temperature = (9.1093837139e-31 / 1.602176634e-19) * result[
        "velocity_covariance_tensor_m2_s2"
    ]
    np.testing.assert_allclose(
        result["temperature_tensor_eV"], expected_temperature,
        rtol=1e-12, atol=1e-12,
    )
    np.testing.assert_allclose(
        result["temperature_tensor_eigenvalues_eV"],
        np.linalg.eigvalsh(expected_temperature), rtol=1e-12, atol=1e-12,
    )


def test_physical_optical_thomson_3d_kernel_conserves_bin_power_and_applies_doppler_projection():
    wavelength = np.linspace(499.85, 500.15, 151)
    directions = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    velocity = np.array([[2.0e4, 0., 0.], [-2.0e4, 0., 0.]])
    result = reconstruct_optical_thomson_3d_evdf(
        wavelength, np.ones((3, wavelength.size)), directions, velocity,
        laser_wavelength_nm=500.0, scattering_angle_rad=np.pi / 2,
        view_gain=np.ones(3),
        instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=np.zeros((3, wavelength.size)),
        voxel_volume_m3_s3=np.ones(2), regularization_matrix=np.eye(2),
        regularization_strength=1e-4,
    )
    expected = 500.0 * (1.0 - 2.0*np.sin(np.pi/4.0)
        * (directions @ velocity.T) / 299792458.0)
    np.testing.assert_allclose(result["mapped_line_centres_nm"], expected, rtol=0, atol=1e-12)
    integrated = np.sum(result["physical_spectral_kernels"]
        * result["wavelength_bin_widths_nm"][None, :, None], axis=1)
    np.testing.assert_allclose(integrated, 0.5*np.ones_like(integrated), rtol=1e-12, atol=1e-12)
    assert result["physical_forward_model"] == "elastic_single_particle_unpolarized_nonrelativistic_optical_thomson"
    assert result["validity_domain"]["relativistic_corrections_included"] is False


def test_physical_optical_thomson_3d_kernel_uses_per_view_response_normalization():
    wavelength = np.linspace(499.85, 500.15, 101)
    with pytest.raises(ValueError, match="conserve integrated spectral power"):
        reconstruct_optical_thomson_3d_evdf(
            wavelength, np.ones((2,101)), [[1.,0.,0.],[0.,1.,0.]],
            [[1.0e4,0.,0.],[-1.0e4,0.,0.]],
            laser_wavelength_nm=500.0, scattering_angle_rad=[np.pi/2,np.pi/2],
            view_gain=[1.0,1.0],
            instrument_response_matrix=np.stack([np.eye(101),0.5*np.eye(101)]),
            background_spectra=np.zeros((2,101)), voxel_volume_m3_s3=np.ones(2),
            regularization_matrix=np.eye(2), regularization_strength=1e-3,
        )


def test_physical_optical_thomson_3d_rejects_nonconserving_view_hidden_by_other_view():
    wavelength = np.linspace(499.85, 500.15, 101)
    # The first view conserves power; the second loses all of it. Summing
    # over views would incorrectly make this stack appear normalized.
    response = np.stack([np.eye(101), np.zeros((101, 101))])
    with pytest.raises(ValueError, match="every view's columns must sum to one"):
        reconstruct_optical_thomson_3d_evdf(
            wavelength, np.ones((2, 101)),
            [[1., 0., 0.], [0., 1., 0.]],
            [[1.0e4, 0., 0.], [-1.0e4, 0., 0.]],
            laser_wavelength_nm=500.0,
            scattering_angle_rad=[np.pi/2, np.pi/2],
            view_gain=[1.0, 1.0],
            instrument_response_matrix=response,
            background_spectra=np.zeros((2, 101)),
            voxel_volume_m3_s3=np.ones(2),
            regularization_matrix=np.eye(2),
            regularization_strength=1e-3,
        )


def test_multiview_evdf_preserves_signed_background_subtracted_noise_residuals():
    kernels = np.zeros((1, 5, 2))
    kernels[0, :, 0] = [1.0, 0.8, 0.5, 0.2, 0.1]
    kernels[0, :, 1] = [0.1, 0.2, 0.5, 0.8, 1.0]
    truth = np.array([0.4, 0.6])
    volume = np.ones(2)
    signal = (kernels[0] @ truth)[None, :]
    signal[0, 0] -= 0.2  # a valid negative residual after background subtraction
    result = reconstruct_multiview_3d_evdf(
        kernels, signal, np.array([[-1.0e4, 0., 0.], [1.0e4, 0., 0.]]),
        voxel_volume_m3_s3=volume,
        regularization_matrix=np.zeros((1, 2)),
        measurement_uncertainty=np.full((1, 5), 0.1),
        regularization_strength=1e-8,
    )
    assert result["residual_spectra"][0, 0] < 0.0
    assert np.all(result["regularized_candidate_evdf_m3_s3"] >= 0.0)
    assert result["noise_model"] == "full_stacked_measurement_plus_model_discrepancy_plus_systematic_covariance"


def test_physical_optical_thomson_3d_accepts_negative_background_subtracted_noise_bin():
    wavelength = np.linspace(499.85, 500.15, 101)
    velocity = np.array([[-1.0e4, 0., 0.], [1.0e4, 0., 0.]])
    direction = np.array([[1., 0., 0.]])
    pilot = reconstruct_optical_thomson_3d_evdf(
        wavelength, np.ones((1, wavelength.size)), direction, velocity,
        laser_wavelength_nm=500.0, scattering_angle_rad=np.pi/2,
        view_gain=[1.0], instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=np.zeros((1, wavelength.size)),
        voxel_volume_m3_s3=np.ones(2), regularization_matrix=np.zeros((1, 2)),
        regularization_strength=1e-8,
    )
    truth = np.array([0.35, 0.65])
    measured = (pilot["physical_spectral_kernels"][0] @ truth)[None, :]
    background = np.zeros_like(measured)
    background[0, 0] = measured[0, 0] + 1e-8
    result = reconstruct_optical_thomson_3d_evdf(
        wavelength, measured, direction, velocity,
        laser_wavelength_nm=500.0, scattering_angle_rad=np.pi/2,
        view_gain=[1.0], instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=background,
        voxel_volume_m3_s3=np.ones(2), regularization_matrix=np.zeros((1, 2)),
        measurement_uncertainty=np.full_like(measured, 1e-7),
        regularization_strength=1e-8,
    )
    assert result["background_subtracted_spectra"][0, 0] < 0.0
    assert np.all(result["regularized_candidate_evdf_m3_s3"] >= 0.0)


def test_physical_optical_thomson_3d_route_rejects_relativistic_velocity_grid():
    wavelength = np.linspace(499.0, 501.0, 101)
    with pytest.raises(ValueError, match="non-relativistic"):
        reconstruct_optical_thomson_3d_evdf(
            wavelength, np.ones((1,101)), [[1.,0.,0.]],
            [[0.11*299792458.0,0.,0.]],
            laser_wavelength_nm=500.0, scattering_angle_rad=np.pi/2,
            view_gain=[1.0],
            instrument_response_matrix=np.eye(101),
            background_spectra=np.zeros((1,101)), voxel_volume_m3_s3=np.ones(1),
            regularization_matrix=np.eye(1), regularization_strength=1e-3,
        )


def test_physical_multiview_evdf_is_reachable_through_public_analysis1_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    wavelength = np.linspace(499.85, 500.15, 101)
    result = run_advanced_thomson_physics(
        "thomson_multiview_3d_evdf_physical",
        wavelength,
        np.ones((2, wavelength.size)),
        np.array([[1.,0.,0.],[0.,1.,0.]]),
        np.array([[1.0e4,0.,0.],[-1.0e4,0.,0.]]),
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2,
        view_gain=np.ones(2),
        instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=np.zeros((2,wavelength.size)),
        voxel_volume_m3_s3=np.ones(2),
        regularization_matrix=np.eye(2),
        regularization_strength=1e-3,
    )
    assert result["physical_forward_model"] == "elastic_single_particle_unpolarized_nonrelativistic_optical_thomson"
    assert result["physical_spectral_kernels"].shape == (2, wavelength.size, 2)


def test_primary_analysis_registry_exposes_physical_multiview_evdf_workflow():
    from tokamak_analysis.analysis_registry import get_analysis
    primary = get_analysis(1)
    modes = primary["execution_contract"]["workflow_modes"]
    entry = next(item for item in modes if item["name"] == "physical_multiview_3d_evdf_inversion")
    assert entry["model_selector"] == "thomson_multiview_3d_evdf_physical"
    assert "scattering_vector_unit (view × 3 unit vectors)" in entry["inputs"]
    assert "physical_spectral_kernels (view × wavelength × voxel)" in entry["outputs"]


def test_physical_optical_thomson_3d_uses_per_view_scattering_angles_and_angular_factor():
    wavelength = np.linspace(499.8, 500.2, 201)
    directions = np.array([[1.,0.,0.],[0.,1.,0.]])
    velocity = np.array([[1.0e4,0.,0.],[-1.0e4,0.,0.]])
    angles = np.array([np.pi/3.0, np.pi/2.0])
    result = reconstruct_optical_thomson_3d_evdf(
        wavelength, np.ones((2,wavelength.size)), directions, velocity,
        laser_wavelength_nm=500.0, scattering_angle_rad=angles,
        view_gain=np.array([2.0,0.5]),
        instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=np.zeros((2,wavelength.size)),
        voxel_volume_m3_s3=np.ones(2), regularization_matrix=np.eye(2),
        regularization_strength=1e-4,
    )
    qfactor = 2.0*np.sin(angles/2.0)
    expected_centres = 500.0*(1.0-qfactor[:,None]*(directions@velocity.T)/299792458.0)
    np.testing.assert_allclose(result["mapped_line_centres_nm"], expected_centres, atol=1e-12)
    expected_scale = np.array([2.0,0.5])*0.5*(1.0+np.cos(angles)**2)
    integrated = np.sum(result["physical_spectral_kernels"]*result["wavelength_bin_widths_nm"][None,:,None],axis=1)
    np.testing.assert_allclose(integrated, expected_scale[:,None]*np.ones((2,2)), rtol=1e-12, atol=1e-12)


def test_physical_optical_thomson_3d_inverse_recovers_known_two_voxel_distribution():
    wavelength = np.linspace(499.85, 500.15, 151)
    directions = np.array([[1.,0.,0.]])
    velocity = np.array([[-1.0e4,0.,0.],[1.0e4,0.,0.]])
    volumes = np.ones(2)
    regularizer = np.zeros((1,2))
    common = dict(
        scattering_vector_unit=directions,
        velocity_grid_m_s=velocity,
        laser_wavelength_nm=500.0,
        scattering_angle_rad=np.pi/2,
        view_gain=[1.0],
        instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=np.zeros((1,wavelength.size)),
        voxel_volume_m3_s3=volumes,
        regularization_matrix=regularizer,
        regularization_strength=1e-8,
    )
    pilot = reconstruct_optical_thomson_3d_evdf(
        wavelength, np.ones((1,wavelength.size)), **common
    )
    truth = np.array([0.25,0.75])
    synthetic_spectrum = (
        pilot["physical_spectral_kernels"][0] @ (truth*volumes)
    )[None,:]
    recovered = reconstruct_optical_thomson_3d_evdf(
        wavelength, synthetic_spectrum, **common
    )
    assert recovered["status"] == "identified_from_multi_view_measurements"
    np.testing.assert_allclose(
        recovered["evdf_m3_s3"]*volumes, truth, rtol=2e-5, atol=2e-7
    )
    assert recovered["unpolarized_differential_cross_section_m2_sr"][0] > 0.0


def test_physical_optical_thomson_3d_rejects_nonunit_scattering_directions():
    wavelength = np.linspace(499.85, 500.15, 101)
    with pytest.raises(ValueError, match="must already be a unit vector"):
        reconstruct_optical_thomson_3d_evdf(
            wavelength, np.ones((1,101)), [[2.,0.,0.]], [[0.,0.,0.]],
            laser_wavelength_nm=500.0, scattering_angle_rad=np.pi/2,
            view_gain=[1.0], instrument_response_matrix=np.eye(101),
            background_spectra=np.zeros((1,101)), voxel_volume_m3_s3=[1.0],
            regularization_matrix=np.eye(1), regularization_strength=1e-3,
        )


def test_physical_optical_thomson_3d_withholds_distribution_for_one_projection_null_space():
    wavelength = np.linspace(499.85, 500.15, 101)
    velocity = np.array([
        [vx,vy,vz]
        for vx in [-1.0e4,1.0e4]
        for vy in [-1.0e4,1.0e4]
        for vz in [-1.0e4,1.0e4]
    ])
    result = reconstruct_optical_thomson_3d_evdf(
        wavelength, np.ones((1,wavelength.size)), [[1.,0.,0.]], velocity,
        laser_wavelength_nm=500.0, scattering_angle_rad=np.pi/2,
        view_gain=[1.0], instrument_response_matrix=np.eye(wavelength.size),
        background_spectra=np.zeros((1,wavelength.size)),
        voxel_volume_m3_s3=np.ones(8), regularization_matrix=np.eye(8),
        regularization_strength=1e-3,
    )
    assert result["status"] == "conditional_regularized_3d_candidate"
    assert result["evdf_m3_s3"] is None
    assert result["regularized_candidate_evdf_m3_s3"] is not None
    assert result["null_space_dimension"] > 0
    assert result["mean_velocity_vector_m_s"] is None

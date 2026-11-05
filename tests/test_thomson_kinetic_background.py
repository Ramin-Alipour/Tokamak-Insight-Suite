import numpy as np

from tokamak_analysis.thomson_kinetic_background import (
    fit_noncollective_kinetic_thomson_with_background,
    _kappa_bin_average_spectral_density,
    _wavelength_edges,
)


def test_kappa_structured_background_fit_retains_joint_cross_covariance():
    wavelength = np.linspace(529.0, 535.0, 181)
    laser = 532.0
    angle = np.pi / 2.0
    response = np.eye(wavelength.size)
    truth = np.array([300.0, 8.0e4, 3.5])
    q = 2.0 * np.sin(angle / 2.0)
    center = laser * (1.0 - q * truth[1] / 299792458.0)
    sigma = laser * 2.0 * np.sin(angle / 2.0) * np.sqrt(truth[0] / 510998.95069)
    df = 2.0 * truth[2] - 1.0
    scale = sigma * np.sqrt((df - 2.0) / df)
    shape = (1.0 + ((wavelength - center) / scale) ** 2 / df) ** (-truth[2])
    z = 2.0 * (wavelength - wavelength.min()) / np.ptp(wavelength) - 1.0
    measured = 2.3 * shape + 0.04 + 0.006 * z
    result = fit_noncollective_kinetic_thomson_with_background(
        wavelength, measured,
        physical_model="kappa",
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        instrument_response=response,
        physical_initial=[260.0, 0.0, 4.2],
        background_model="linear",
        measurement_uncertainty=np.full(wavelength.size, 0.003),
        model_discrepancy_covariance=np.eye(wavelength.size) * 1.0e-6,
        systematic_covariance=np.eye(wavelength.size) * 4.0e-7,
        max_nfev=5000,
    )
    assert result["optimizer_success"]
    assert result["joint_parameter_covariance"].shape == (6, 6)
    assert result["physical_background_cross_covariance"].shape == (3, 2)
    assert result["jacobian_rank"] == result["jacobian_parameter_count"]
    assert np.allclose(result["physical_parameters"], truth, rtol=0.04, atol=1e-3)
    assert result["absolute_density_inferred"] is False
    assert result["model_discrepancy_covariance_included"] is True
    assert result["systematic_covariance_included"] is True
    assert result["covariance_components"] == {
        "measurement_covariance_supplied": True,
        "model_discrepancy_covariance_supplied": True,
        "systematic_covariance_supplied": True,
        "combined_covariance_used_for_whitening": True,
    }
    assert result["noise_model"] == "measurement_plus_model_discrepancy_plus_systematic_covariance"
    assert result["parameter_identifiability"]["status"] == "locally_identifiable_after_joint_structured_background_fit"
    assert np.allclose(
        result["joint_parameter_covariance"],
        result["joint_parameter_covariance"].T,
        rtol=1e-10,
        atol=1e-14,
    )
    assert np.all(np.linalg.eigvalsh(result["joint_parameter_covariance"]) >= -1e-12)
    assert result["cross_section_model"] == "isotropic_finite_variance_Kappa_projection"


def test_structured_background_fit_rejects_non_psd_model_discrepancy_covariance():
    import pytest

    wavelength = np.linspace(529.0, 535.0, 41)
    measured = 1.0 + np.exp(-0.5 * ((wavelength - 532.0) / 0.5) ** 2)
    with pytest.raises(ValueError, match="model_discrepancy_covariance must be positive semidefinite"):
        fit_noncollective_kinetic_thomson_with_background(
            wavelength, measured,
            physical_model="kappa",
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2.0,
            instrument_response=np.eye(wavelength.size),
            physical_initial=[300.0, 0.0, 3.5],
            model_discrepancy_covariance=-np.eye(wavelength.size),
        )


def test_kappa_forward_model_integrates_exact_finite_wavelength_bin_probabilities():
    wavelength = np.linspace(529.0, 535.0, 181)
    spectrum = _kappa_bin_average_spectral_density(
        wavelength, 532.0, np.pi / 2.0, 300.0, 0.0, 3.5
    )
    widths = np.diff(_wavelength_edges(wavelength))
    assert np.all(spectrum >= 0.0)
    assert np.sum(spectrum * widths) > 0.99
    assert np.sum(spectrum * widths) < 1.0


def test_relativistic_compton_fit_convolves_measured_laser_profile_in_photon_flux_space():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(480.0, 584.0, 81)
    laser = 532.0
    angle = np.pi / 2.0
    forward = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 100.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
        include_compton_recoil=True,
    )
    measured = 2.0 * forward["spectral_density_per_nm"] + 0.03
    source_wavelength = np.array([531.9, 531.95, 532.0, 532.05, 532.1])
    source_power_density = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
    result = fit_noncollective_kinetic_thomson_with_background(
        wavelength, measured,
        physical_model="relativistic_maxwell_juttner",
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        instrument_response=np.eye(wavelength.size),
        physical_initial=[100.0, 0.0],
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
        include_compton_recoil=True,
        laser_spectrum_wavelength_nm=source_wavelength,
        laser_spectrum_density_per_nm=source_power_density,
        laser_spectrum_quadrature_order=3,
        max_nfev=120,
    )
    assert result["optimizer_success"]
    assert result["recoil_included"] is True
    assert result["laser_source_profile_used"] is True
    assert result["laser_source_quadrature_nodes"] == 3
    assert min(result["laser_source_quadrature_wavelength_nm"]) >= 531.95
    assert max(result["laser_source_quadrature_wavelength_nm"]) <= 532.05
    assert "photon_flux" in result["laser_source_profile_convolution"]
    assert "Klein_Nishina" in result["laser_source_profile_convolution"]


def test_relativistic_source_profile_requires_a_valid_positive_spectrum():
    import pytest

    wavelength = np.linspace(480.0, 584.0, 41)
    measured = 1.0 + np.exp(-0.5*((wavelength-532.0)/2.0)**2)
    with pytest.raises(ValueError, match="non-negative"):
        fit_noncollective_kinetic_thomson_with_background(
            wavelength, measured,
            physical_model="relativistic_maxwell_juttner",
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2.0,
            instrument_response=np.eye(wavelength.size),
            physical_initial=[100.0, 0.0],
            incident_polarization=[0.0, 0.0, 1.0],
            laser_spectrum_wavelength_nm=[531.9, 532.0, 532.1],
            laser_spectrum_density_per_nm=[0.0, -1.0, 0.0],
        )


def test_kinetic_background_model_is_routed_through_public_advanced_dashboard():
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    model = "thomson_kinetic_background_fit"
    assert model in MODEL_LABELS and model in EXAMPLES
    config = dict(EXAMPLES[model])
    result = run_advanced_thomson_physics(model, **config)
    assert result["public_physics_entrypoint"] == model
    assert result["physical_model"] == "kappa"
    assert result["absolute_density_inferred"] is False
    assert result["parameter_identifiability"]["status"] == "locally_identifiable_after_joint_structured_background_fit"
    assert result["joint_parameter_covariance"].shape[0] == len(result["joint_parameter_names"])


def test_kappa_fit_enforces_nonrelativistic_temperature_and_flow_domain():
    import pytest
    wavelength = np.linspace(529.0, 535.0, 81)
    measured = 1.0 + np.exp(-0.5 * ((wavelength - 532.0) / 0.5) ** 2)
    common = dict(wavelength_nm=wavelength, measured_spectrum=measured, physical_model="kappa",
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size))
    with pytest.raises(ValueError, match="valid, strictly ordered physical_bounds"):
        fit_noncollective_kinetic_thomson_with_background(**common, physical_initial=[6000.0, 0.0, 3.5])
    with pytest.raises(ValueError, match="valid, strictly ordered physical_bounds"):
        fit_noncollective_kinetic_thomson_with_background(**common, physical_initial=[300.0, 0.11 * 299792458.0, 3.5])


def test_kappa_fit_reports_declared_asymptotic_validity_domain():
    wavelength = np.linspace(529.0, 535.0, 181)
    te, flow, kappa = 300.0, 8.0e4, 3.5
    q = 2.0 * np.sin(np.pi / 4.0)
    center = 532.0 * (1.0 - q * flow / 299792458.0)
    sigma = 532.0 * q * np.sqrt(te / 510998.95069)
    df = 2.0 * kappa - 1.0
    scale = sigma * np.sqrt((df - 2.0) / df)
    shape = (1.0 + ((wavelength - center) / scale) ** 2 / df) ** (-kappa)
    measured = 2.0 * shape + 0.03
    result = fit_noncollective_kinetic_thomson_with_background(
        wavelength, measured, physical_model="kappa", laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(wavelength.size),
        physical_initial=[280.0, 0.0, 4.0],
        measurement_uncertainty=np.full(wavelength.size, 0.003))
    assert result["model_validity_status"] == "within_declared_nonrelativistic_Kappa_domain"
    assert result["model_validity_limits"]["electron_temperature_over_rest_energy_max"] == 0.01
    assert result["electron_temperature_eV"] / 510998.95069 <= 0.01
    assert abs(result["electron_velocity_projection_m_s"]) / 299792458.0 <= 0.1

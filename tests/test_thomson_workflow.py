import numpy as np
"""Regression tests for the canonical measured-spectrum Thomson workflow."""

import pytest
import tokamak_analysis.thomson_scattering as scattering
import tokamak_analysis.thomson_workflow as workflow


def test_prederived_profile_only_entry_points_are_removed():
    for name in (
        "ThomsonProfile", "ThomsonSummary", "build_profile",
        "profile_summary", "interpolate_profile", "weighted_mean",
        "reconstruct_thomson_profile",
    ):
        assert not hasattr(scattering, name)
    assert not hasattr(workflow, "analyze_thomson_profile")


def test_registered_spatial_profile_workflow_is_spectrum_driven():
    assert callable(workflow.analyze_thomson_spectra)
    assert callable(workflow.reconstruct_thomson_profile_from_spectra)



def test_profile_reconstruction_rejects_unordered_channel_coordinates_without_reordering():
    with pytest.raises(ValueError, match="input order is preserved and never repaired"):
        workflow._reconstruct_inferred_parameter_profile(
            [0.2, 0.1], [2.0, 1.0], method="pchip"
        )


def test_joint_spatial_spectral_thomson_workflow_is_exposed():
    assert callable(workflow.reconstruct_thomson_joint_spatial_spectra)
    from tokamak_analysis.thomson_joint_spatial_spectral import reconstruct_joint_thomson_spatial_spectra
    assert workflow.reconstruct_thomson_joint_spatial_spectra is not reconstruct_joint_thomson_spatial_spectra


def test_registered_spatial_profile_analysis_dispatches_joint_spatial_spectral_mode(monkeypatch):
    from tokamak_analysis.analysis_registry import resolve_analysis
    import tokamak_analysis.thomson_joint_spatial_spectral as joint

    captured = {}
    def fake(wavelength, spectra, radius, mixing, **kwargs):
        captured["mixing"] = mixing
        captured["kwargs"] = kwargs
        return {"analysis_status": "joint-dispatch-tested"}

    monkeypatch.setattr(joint, "reconstruct_joint_thomson_spatial_spectra", fake)
    entrypoint = resolve_analysis(46)
    mixing = np.eye(3)
    result = entrypoint(
        np.linspace(520., 544., 9), np.ones((3, 9)), [0.1, 0.2, 0.3],
        joint_spatial_spectral=True, spatial_mixing_matrix=mixing,
        laser_wavelength_nm=532., scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(9),
        spatial_regularization_strength=0.02,
    )
    assert result["analysis_status"] == "joint-dispatch-tested"
    np.testing.assert_array_equal(captured["mixing"], mixing)
    assert captured["kwargs"]["regularization_strength"] == pytest.approx(0.02)



def test_joint_spatial_dispatch_forwards_relativistic_model_and_compton_selection(monkeypatch):
    captured = {}

    def fake_joint(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"analysis_status": "joint-dispatch-tested"}

    import tokamak_analysis.thomson_joint_spatial_spectral as joint
    monkeypatch.setattr(joint, "reconstruct_joint_thomson_spatial_spectra", fake_joint)
    result = workflow.reconstruct_thomson_profile_from_spectra(
        np.linspace(520.0, 544.0, 9), np.ones((3, 9)), np.array([0.1, 0.2, 0.3]),
        joint_spatial_spectral=True, spatial_mixing_matrix=np.eye(3),
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(9), model="non_collective_relativistic_maxwell_juttner",
        spatial_psf_sigma_m=0.02,
        incident_polarization=[1.0, 0.0, 0.0], maximum_recoil_correction_fraction=0.02,
        include_compton_recoil=True,
    )
    assert result["analysis_status"] == "joint-dispatch-tested"
    assert captured["kwargs"]["physical_model"] == "relativistic_maxwell_juttner"
    assert captured["kwargs"]["spatial_psf_sigma_m"] == pytest.approx(0.02)
    assert captured["kwargs"]["incident_polarization"] == [1.0, 0.0, 0.0]
    assert captured["kwargs"]["maximum_recoil_correction_fraction"] == pytest.approx(0.02)
    assert captured["kwargs"]["include_compton_recoil"] is True



def test_joint_spatial_dispatch_forwards_gaussian_psf_calibration_uncertainty(monkeypatch):
    captured = {}

    def fake_joint(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"analysis_status": "joint-dispatch-tested"}

    import tokamak_analysis.thomson_joint_spatial_spectral as joint
    monkeypatch.setattr(joint, "reconstruct_joint_thomson_spatial_spectra", fake_joint)
    result = workflow.reconstruct_thomson_profile_from_spectra(
        np.linspace(520.0, 544.0, 9), np.ones((3, 9)), np.array([0.1, 0.2, 0.3]),
        joint_spatial_spectral=True, spatial_mixing_matrix=np.eye(3),
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(9), model="non_collective_maxwellian",
        spatial_psf_sigma_m=0.02,
        spatial_psf_sigma_uncertainty_m=0.003,
    )
    assert result["analysis_status"] == "joint-dispatch-tested"
    assert captured["kwargs"]["spatial_psf_sigma_m"] == pytest.approx(0.02)
    assert captured["kwargs"]["spatial_psf_sigma_uncertainty_m"] == pytest.approx(0.003)

def test_gls_smoothing_accepts_small_but_positive_covariance_without_rescaling():
    radius = np.linspace(0.0, 1.0, 5)
    values = np.array([1.0, 1.2, 1.1, 1.4, 1.3])
    covariance = np.eye(5) * 1e-18

    fitted, gradient, diagnostics = workflow._generalized_least_squares_smoothing_spline(
        radius, values, covariance, np.linspace(0.0, 1.0, 11)
    )

    assert fitted.shape == (11,)
    assert gradient.shape == (11,)
    assert np.all(np.isfinite(fitted))
    assert np.all(np.isfinite(gradient))
    assert diagnostics["covariance_model"] == "researcher_supplied_full_cross_channel_covariance"


def test_exact_relativistic_flow_projection_is_exposed_by_canonical_workflow():
    assert callable(workflow.thomson_velocity_projection_relativistic_exact)
    result = workflow.thomson_velocity_projection_relativistic_exact(
        532.0, 532.0, np.pi / 2.0
    )
    assert result["electron_velocity_projection_m_s"] == pytest.approx(0.0)
    assert result["identifiability"] == "one_projected_velocity_only"


def test_registered_gyrotropic_workflow_dispatches_bimaxwellian_and_bikappa_without_fallback(monkeypatch):
    import tokamak_analysis.thomson_anisotropic_inference as anisotropic
    calls = []
    def fake_maxwellian(wavelength, spectra, **kwargs):
        calls.append(("bi_maxwellian", kwargs.copy()))
        return {"model": "bi_maxwellian"}
    def fake_kappa(wavelength, spectra, **kwargs):
        calls.append(("bi_kappa", kwargs.copy()))
        return {"model": "bi_kappa"}
    monkeypatch.setattr(anisotropic, "fit_gyrotropic_bi_maxwellian_thomson", fake_maxwellian)
    monkeypatch.setattr(anisotropic, "fit_gyrotropic_bi_kappa_thomson", fake_kappa)
    wavelength = np.linspace(499.0, 501.0, 17)
    spectra = np.ones((2, wavelength.size))
    common = {"laser_wavelength_nm": 500.0, "scattering_angle_rad": np.pi / 2,
              "magnetic_field_unit": [0, 0, 1],
              "scattering_vector_unit": [[0, 0, 1], [1, 0, 0]],
              "instrument_response_matrix": np.eye(wavelength.size)}
    default = workflow.fit_gyrotropic_thomson_spectra(wavelength, spectra, **common)
    kappa = workflow.fit_gyrotropic_thomson_spectra(wavelength, spectra, distribution_model="bi_kappa", initial_kappa=5.0, **common)
    assert default["selected_distribution_model"] == "bi_maxwellian"
    assert kappa["selected_distribution_model"] == "bi_kappa"
    assert default["model_switch_was_automatic"] is False
    assert calls[0][0] == "bi_maxwellian" and calls[1][0] == "bi_kappa"
    assert calls[1][1]["initial_kappa"] == 5.0

def test_gyrotropic_workflow_rejects_unknown_distribution_without_fallback():
    with pytest.raises(ValueError, match="distribution_model"):
        workflow.fit_gyrotropic_thomson_spectra(
            np.linspace(499.0, 501.0, 17), np.ones((1, 17)), distribution_model="automatic",
            laser_wavelength_nm=500.0, scattering_angle_rad=np.pi / 2,
            magnetic_field_unit=[0, 0, 1], scattering_vector_unit=[[0, 0, 1]],
            instrument_response_matrix=np.eye(17))


def test_joint_spatial_dispatch_forwards_full_psf_and_systematic_covariances(monkeypatch):
    captured = {}
    def fake_joint(*args, **kwargs):
        captured["kwargs"] = kwargs
        return {"analysis_status": "joint-dispatch-tested"}
    import tokamak_analysis.thomson_joint_spatial_spectral as joint
    monkeypatch.setattr(joint, "reconstruct_joint_thomson_spatial_spectra", fake_joint)
    wavelength = np.linspace(520.0, 544.0, 9)
    psf_covariance = np.eye(9)
    spectrum_psf_cross = np.zeros((3*wavelength.size, 9))
    systematic = np.eye(3*wavelength.size) * 1.0e-6
    result = workflow.reconstruct_thomson_profile_from_spectra(
        wavelength, np.ones((3, wavelength.size)), np.array([0.1, 0.2, 0.3]),
        joint_spatial_spectral=True, spatial_mixing_matrix=np.eye(3),
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2.0,
        instrument_response=np.eye(wavelength.size), model="non_collective_maxwellian",
        signal_uncertainty=np.full((3, wavelength.size), 0.01),
        spatial_psf_matrix=np.eye(3),
        spatial_psf_matrix_covariance=psf_covariance,
        spectral_psf_cross_covariance=spectrum_psf_cross,
        systematic_covariance=systematic,
    )
    assert result["analysis_status"] == "joint-dispatch-tested"
    np.testing.assert_array_equal(captured["kwargs"]["spatial_psf_matrix_covariance"], psf_covariance)
    np.testing.assert_array_equal(captured["kwargs"]["spectral_psf_cross_covariance"], spectrum_psf_cross)
    np.testing.assert_array_equal(captured["kwargs"]["systematic_covariance"], systematic)


def test_spatial_profile_registry_exposes_full_psf_matrix_uncertainty_contract():
    from tokamak_analysis.analysis_registry import get_analysis
    entry = get_analysis(46)
    inputs = {item["name"] for item in entry["execution_contract"]["input_specs"]}
    outputs = {item["name"] for item in entry["execution_contract"]["output_specs"]}
    assert "spatial_psf_matrix_covariance" in inputs
    assert "spectral_psf_cross_covariance" in inputs
    assert "systematic_covariance" in inputs
    assert "psf_matrix_calibration_parameter_covariance" in outputs


def test_product_api_routes_optical_poisson_counts_to_the_physical_solver():
    from tis.products.thomson.api import fit_optical_thomson_poisson_counts

    wavelength = np.linspace(530.0, 532.0, 121)
    expected = (
        450.0 * np.exp(-0.5 * ((wavelength - 531.0) / 0.18) ** 2)
        + 18.0 + 2.0 * (wavelength - 531.0)
    )
    counts = np.random.default_rng(21).poisson(np.maximum(expected, 1.0)).astype(int)
    result = fit_optical_thomson_poisson_counts(
        wavelength, counts, laser_wavelength_nm=531.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
    )
    assert result["analysis_id"] == 1
    assert result["likelihood"] == "independent_Poisson"
    assert result["density_inference_available"] is False
    assert result["reduced_chi_square"] is None
    assert result["integrated_signal_counts"] > 0.0
    assert np.isfinite(result["electron_temperature_eV"])

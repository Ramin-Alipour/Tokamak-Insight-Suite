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


def test_psf_aware_profile_reconstruction_is_explicitly_exposed():
    latent = np.array([1.0, 2.0, 1.0])
    mixing = np.array([
        [0.8, 0.2, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.2, 0.8],
    ])
    measured = mixing @ latent
    out = workflow.reconstruct_thomson_profile_with_psf(
        measured, mixing,
        measurement_uncertainty=np.full(3, 0.01),
        regularization=1e-6,
    )
    assert out["mixing_matrix_is_external_fixed_input"] is True
    assert out["fitted_measured_profile"].shape == measured.shape
    np.testing.assert_allclose(out["fitted_measured_profile"], measured, atol=1e-6)

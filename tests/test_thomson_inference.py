import numpy as np
import pytest

from tokamak_analysis.thomson_inference import (
    analyze_thomson_spectra as _analyze_thomson_spectra,
    _gaussian_bin_averaged_shape,
    _kappa_bin_averaged_shape,
)


def analyze_thomson_spectra(*args, **kwargs):
    """Keep legacy test fixtures explicit while testing the production API separately."""
    kwargs.setdefault("model", "non_collective_maxwellian")
    return _analyze_thomson_spectra(*args, **kwargs)


def _spectral_case(te_ev, center_nm=532.0, laser_nm=532.0, angle=np.pi/2, n=801):
    x = np.linspace(laser_nm-40.0, laser_nm+40.0, n)
    sigma = 2.0*laser_nm*np.sqrt(te_ev/510998.95069)*np.sin(angle/2.0)
    feature, _ = _gaussian_bin_averaged_shape(x, 1200.0, center_nm, sigma)
    y = feature+25.0+0.1*(x-center_nm)
    return x, y, sigma


def _asymmetric_lsf(n):
    response = np.zeros((n,n))
    for j in range(n):
        response[j,j] += 0.65
        if j > 0:
            response[j-1,j] += 0.25
        if j < n-1:
            response[j+1,j] += 0.10
    response /= response.sum(axis=0, keepdims=True)
    return response


def test_multi_channel_spectra_recover_channel_resolved_temperature_and_radius():
    x, y1, _ = _spectral_case(100.0)
    _, y2, _ = _spectral_case(150.0)
    out = analyze_thomson_spectra(
        x, np.vstack([y1,y2]), [0.12,0.24],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
    )
    assert out["channel_count"] == 2
    np.testing.assert_allclose(out["radius_m"], [0.12,0.24])
    np.testing.assert_allclose(out["electron_temperature_profile_eV"], [100.0,150.0], rtol=0.04)
    assert np.isnan(out["electron_density_profile_m3"]).all()
    assert out["channels"][0]["density_status"] == "not_identifiable_from_spectral_shape_alone"
    assert out["channels"][0]["spectral_bin_integration"] == (
        "exact_Gaussian_CDF_finite_volume_bin_averages"
    )
    assert 0.0 < out["channels"][0]["captured_spectral_probability"] <= 1.0


def test_non_gaussian_instrument_response_matrix_is_used_in_forward_fit():
    x, intrinsic, _ = _spectral_case(100.0)
    response = _asymmetric_lsf(x.size)
    measured = response @ (intrinsic - 25.0) + 25.0
    out = analyze_thomson_spectra(
        x, measured[None,:], [0.2], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=response,
    )
    assert out["channels"][0]["instrument_response_used"] is True
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(100.0, rel=0.05)


def test_instrument_response_conserves_spectral_power_on_nonuniform_wavelength_grid():
    from tokamak_analysis.thomson_inference import _apply_response

    u = np.linspace(-40.0, 40.0, 801)
    x = 532.0 + u + 0.002 * u**2
    sigma = 2.0 * 532.0 * np.sqrt(100.0 / 510998.95069) * np.sin(np.pi / 4)
    intrinsic, _ = _gaussian_bin_averaged_shape(x, 1200.0, 532.0, sigma)
    background = 25.0 + 0.1 * (x - 532.0)
    response = _asymmetric_lsf(x.size)
    measured = _apply_response(response, intrinsic, x) + background

    out = analyze_thomson_spectra(
        x, measured[None, :], [0.2], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=response,
    )
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(100.0, rel=0.05)


def test_index_space_response_kernel_is_rejected_for_nonuniform_wavelength_grid():
    from tokamak_analysis.thomson_inference import _response_for_channel

    x = np.array([500.0, 501.0, 502.2, 503.8, 506.0])
    with pytest.raises(ValueError, match="requires a uniform wavelength grid"):
        _response_for_channel(np.array([0.2, 0.6, 0.2]), 0, 1, x.size, x)


def test_calibrated_1d_response_kernel_is_preserved_with_finite_window_edge_loss():
    from tokamak_analysis.thomson_inference import _response_for_channel

    wavelength = np.linspace(500.0, 510.0, 11)
    kernel = np.array([0.2, 0.6, 0.2])
    original = kernel.copy()
    response = _response_for_channel(kernel, 0, 1, wavelength.size, wavelength)

    np.testing.assert_array_equal(kernel, original)
    np.testing.assert_allclose(response[:, 5], np.r_[0, 0, 0, 0, 0.2, 0.6, 0.2, 0, 0, 0, 0])
    assert response[:, 0].sum() == pytest.approx(0.8)
    assert response[:, -1].sum() == pytest.approx(0.8)
    assert np.allclose(response[:, 1:-1].sum(axis=0), 1.0)


def test_calibrated_1d_response_kernel_is_not_silently_normalized():
    from tokamak_analysis.thomson_inference import _response_for_channel

    wavelength = np.linspace(500.0, 510.0, 11)
    with pytest.raises(ValueError, match="must already have unit total area"):
        _response_for_channel(np.array([0.2, 0.5, 0.2]), 0, 1, wavelength.size, wavelength)


def test_absolute_density_requires_explicit_watt_per_nm_calibration():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="spectral_power_density_calibrated"):
        analyze_thomson_spectra(
            x, (y*1e-18)[None,:], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size), laser_power_w=2.0,
            interaction_length_m=0.1, collection_solid_angle_sr=1e-3,
            optical_efficiency=0.5, incident_polarization=[0,0,1],
            incident_direction=[1,0,0], scattered_direction=[0,1,0],
        )


def test_absolute_density_and_uncertainty_are_reported_when_full_geometry_is_supplied():
    x, y, _ = _spectral_case(100.0)
    out = analyze_thomson_spectra(
        x, (y*1e-18)[None,:], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size), signal_uncertainty=np.full((1,x.size),0.8e-18), laser_power_w=2.0,
        laser_power_uncertainty_W=0.02, interaction_length_m=0.1,
        interaction_length_uncertainty_m=0.001, collection_solid_angle_sr=1e-3,
        collection_solid_angle_uncertainty_sr=1e-5, optical_efficiency=0.5,
        optical_efficiency_uncertainty=0.01, incident_polarization=[0,0,1],
        incident_direction=[1,0,0], scattered_direction=[0,1,0], spectral_power_density_calibrated=True,
    )
    channel = out["channels"][0]
    assert channel["electron_density_m3"] > 0
    assert channel["electron_density_uncertainty_m3"] > 0
    assert channel["density_method"].startswith("absolute integrated")



def test_one_dimensional_response_kernel_uncertainty_is_monte_carlo_propagated():
    x, intrinsic, _ = _spectral_case(100.0, n=161)
    kernel = np.array([0.2, 0.6, 0.2])
    kernel_uncertainty = np.array([0.005, 0.01, 0.005])
    from tokamak_analysis.thomson_inference import _response_for_channel
    response = _response_for_channel(kernel, 0, 1, x.size, x)
    measured = response @ (intrinsic - 25.0) + 25.0
    result = analyze_thomson_spectra(
        x, measured[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=kernel,
        instrument_response_uncertainty=kernel_uncertainty,
        instrument_response_mc_samples=20,
        instrument_response_mc_min_success_fraction=0.5,
    )
    channel = result["channels"][0]
    assert channel["instrument_response_uncertainty_supplied"] is True
    assert channel["instrument_response_uncertainty_propagated"] is True
    assert channel["instrument_response_mc_successful_samples"] >= 10
    assert channel["electron_temperature_uncertainty_eV"] > 0.0


def test_instrument_response_uncertainty_propagates_into_absolute_density():
    x, y, _ = _spectral_case(100.0, n=161)
    response = np.eye(x.size)
    response_uncertainty = np.full_like(response, 1e-7)
    out = analyze_thomson_spectra(
        x, (y * 1e-18)[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=response,
        instrument_response_uncertainty=response_uncertainty,
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0, 0, 1], incident_direction=[1, 0, 0],
        scattered_direction=[0, 1, 0], spectral_power_density_calibrated=True,
    )
    channel = out["channels"][0]
    assert channel["instrument_response_density_uncertainty_propagated"] is True
    assert channel["integrated_scattered_power_lsf_uncertainty_W"] >= 0.0
    assert channel["electron_density_uncertainty_m3"] > 0.0
    assert channel["integrated_scattered_power_W"] > 0.0
    assert np.isfinite(channel["power_cross_section_relative_covariance"])


def test_optical_analysis_1_routes_collective_dielectric_models_to_physical_validation():
    x, y, _ = _spectral_case(100.0)
    # These models must reach their dedicated physical-input validation rather
    # than being rejected by the non-collective model shortlist. A real fit
    # cannot proceed without explicit ion composition/flows and bounded seeds.
    for model in ("collective_maxwellian", "collective_kappa"):
        with pytest.raises(ValueError, match="Collective Thomson models require explicit initial electron density and temperature estimates"):
            analyze_thomson_spectra(
                x, y[None, :], [0.1], laser_wavelength_nm=532.0,
                scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
                model=model,
            )

def test_collective_kappa_is_executable_through_canonical_optical_analysis():
    from tokamak_analysis.thomson_kinetic_models import collective_kappa_spectral_density

    wavelength = np.linspace(1048.0, 1080.0, 101)
    angle = np.pi / 2.0
    laser = 1064.0
    incident = np.array([-np.sin(angle / 2.0), np.cos(angle / 2.0), 0.0])
    scattered = np.array([np.sin(angle / 2.0), np.cos(angle / 2.0), 0.0])
    plasma = dict(
        electron_density_m3=1.0e19,
        electron_temperature_eV=150.0,
        kappa=3.5,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        incident_direction=incident,
        scattered_direction=scattered,
    )
    physical_spectrum = collective_kappa_spectral_density(
        wavelength, laser, angle, **plasma
    )["spectral_density_per_nm"]
    signal = 2.0e4 * physical_spectrum + 10.0 + 0.001 * (wavelength - laser)
    result = _analyze_thomson_spectra(
        wavelength, signal[None, :], [0.1],
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full((1, wavelength.size), 0.01 * np.max(signal)),
        model="collective_kappa",
        ion_species=plasma["ion_species"],
        ion_fractions=plasma["ion_fractions"],
        ion_temperature_eV=plasma["ion_temperature_eV"],
        electron_velocity_m_s=plasma["electron_velocity_m_s"],
        ion_velocity_m_s=plasma["ion_velocity_m_s"],
        collective_initial_density_m3=1.0e19,
        collective_initial_temperature_eV=150.0,
        collective_density_bounds_m3=[5.0e18, 2.0e19],
        collective_temperature_bounds_eV=[75.0, 300.0],
        collective_kappa_initial=3.5,
        collective_kappa_bounds=[2.0, 6.0],
        collective_external_parameter_mc_samples=20,
        enable_evdf_reconstruction=False,
    )
    channel = result["channels"][0]
    assert result["model"] == "collective_kappa"
    assert channel["optimizer_success"] is True
    assert channel["electron_density_m3"] == pytest.approx(1.0e19, rel=2e-3)
    assert channel["electron_temperature_eV"] == pytest.approx(150.0, rel=2e-3)
    assert channel["kappa_index"] == pytest.approx(3.5, rel=2e-3)
    assert channel["parameter_provenance"]["electron_distribution"].startswith("kappa")

def test_channel_specific_wavelength_axes_are_supported():
    x1, y1, _ = _spectral_case(100.0)
    x2, y2, _ = _spectral_case(150.0, center_nm=532.02)
    out = analyze_thomson_spectra(
        np.vstack([x1,x2]), np.vstack([y1,y2]), [0.1,0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2, instrument_response=np.eye(x1.size),
    )
    assert out["channel_count"] == 2
    assert out["channels"][1]["fitted_center_nm"] == pytest.approx(532.02, abs=0.02)


def test_non_maxwellian_kappa_model_returns_temperature_and_kappa_index():
    x = np.linspace(492.0,572.0,801)
    te = 120.0
    sigma = 2.0*532.0*np.sqrt(te/510998.95069)*np.sin(np.pi/4)
    kappa = 4.0
    df = 2.0*kappa-1.0
    scale = sigma*np.sqrt((df-2.0)/df)
    feature, _ = _kappa_bin_averaged_shape(x, 900.0, 532.0, sigma, kappa)
    y = feature+20.0
    out = analyze_thomson_spectra(
        x,y[None,:],[0.15],laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,instrument_response=np.eye(x.size),model="non_collective_kappa",
    )
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(te,rel=0.05)
    assert out["channels"][0]["kappa_index"] == pytest.approx(kappa,rel=0.05)
    diagnostic = out["channels"][0]["kappa_index_identifiability"]
    assert diagnostic["status"] == "locally_identifiable"
    assert diagnostic["reportable_as_inferred_parameter"] is True
    assert diagnostic["jacobian_rank"] == diagnostic["parameter_count"] == 6
    assert np.isfinite(diagnostic["normalized_jacobian_condition_number"])
    assert diagnostic["conditional_kappa_sensitivity"] > 1e-8
    profile = out["channels"][0]["kappa_profile_likelihood"]
    assert profile["status"] in {"finite_profile_interval", "grid_censored"}
    assert profile["delta_chi_square_threshold"] == pytest.approx(1.0)
    assert profile["noise_scale"].startswith("unweighted_fit_residual_variance_estimated_from_best_fit")
    assert profile["numerical_profile_domain"][0] > 1.5
    assert profile["numerical_profile_domain"][1] >= 20.0
    assert 0.0 < out["channels"][0]["captured_spectral_probability"] <= 1.0
    optimizer = out["channels"][0]["optimizer"]
    assert optimizer["method"] == "deterministic_multistart_bounded_least_squares"
    assert optimizer["attempted_starts"] == 18
    assert optimizer["successful_starts"] >= 1
    assert optimizer["selected_weighted_residual_sum_squares"] >= 0.0


def test_legacy_relativistic_model_alias_is_rejected():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="Unsupported optical Thomson model"):
        analyze_thomson_spectra(
            x, y[None, :], [0.12], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            model="relativistic_maxwell_juttner",
        )


def test_nonrelativistic_flow_projection_uses_q_equals_ks_minus_ki_sign():
    x, y, _ = _spectral_case(100.0, center_nm=531.98)
    out = analyze_thomson_spectra(
        x, y[None, :], [0.12], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
    )
    expected = -299792458.0 * (531.98 - 532.0) / (532.0 * np.sqrt(2.0))
    assert out["channels"][0]["electron_velocity_projection_m_s"] == pytest.approx(
        expected, rel=0.03
    )


def test_registry_resolves_primary_thomson_spectral_workflow():
    from tokamak_analysis.analysis_registry import resolve_analysis
    entrypoint=resolve_analysis(1)
    assert callable(entrypoint)
    assert entrypoint.__name__ == "analyze_thomson_spectra"


def test_collective_external_covariance_indices_use_full_channel_stride():
    from tokamak_analysis.thomson_inference import _collective_external_parameter_indices

    assert _collective_external_parameter_indices(0, 2) == (0, 1, 2, (3, 4))
    assert _collective_external_parameter_indices(1, 2) == (5, 6, 7, (8, 9))


def test_collective_uncertainty_completeness_respects_supplied_flow_and_geometry_errors():
    from tokamak_analysis.thomson_inference import _collective_uncertainty_inputs_missing

    assert _collective_uncertainty_inputs_missing(
        ion_temperature_uncertainty=[10.0, 12.0],
        ion_velocity_uncertainty=np.ones((2, 3)),
        electron_velocity_uncertainty=np.ones(3),
        scattering_angle_uncertainty=None,
        incident_direction_uncertainty=np.full(3, 1e-4),
        scattered_direction_uncertainty=None,
    ) == []

    assert _collective_uncertainty_inputs_missing(
        ion_temperature_uncertainty=None,
        ion_velocity_uncertainty=None,
        electron_velocity_uncertainty=None,
        scattering_angle_uncertainty=None,
        incident_direction_uncertainty=None,
        scattered_direction_uncertainty=None,
    ) == [
        "ion_temperature_uncertainty_eV",
        "ion_velocity_uncertainty_m_s",
        "electron_velocity_uncertainty_m_s",
        "scattering_geometry_uncertainty",
    ]


def test_relativistic_model_propagates_laser_angle_and_polarization_uncertainties():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    x = np.linspace(480.0, 584.0, 81)
    physical = relativistic_maxwell_juttner_spectrum(
        x, 532.0, np.pi / 2, 100.0,
        flow_projection_m_s=0.0,
        incident_polarization=[0.0, 0.0, 1.0],
        radial_order=12, angular_order=8,
    )
    y = 1000.0 * physical["spectral_density_per_nm"] + 20.0
    result = analyze_thomson_spectra(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        model="non_collective_relativistic_maxwell_juttner",
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        incident_polarization=[0.0, 0.0, 1.0],
        laser_wavelength_uncertainty_nm=0.01,
        scattering_angle_uncertainty_rad=1e-4,
        incident_polarization_uncertainty=[1e-4, 1e-4, 1e-4],
        laser_power_w=2.0,
        interaction_length_m=0.1,
        collection_solid_angle_sr=0.01,
        optical_efficiency=0.5,
        spectral_power_density_calibrated=True,
        instrument_response_mc_samples=20,
        instrument_response_mc_min_success_fraction=0.5,
    )
    channel = result["channels"][0]
    assert channel["relativistic_external_parameter_uncertainty_propagated"] is True
    assert channel["relativistic_external_parameter_mc_successful_samples"] >= 10
    assert channel["relativistic_external_temperature_uncertainty_eV"] > 0.0
    assert channel["effective_differential_cross_section_m2_sr"] > 0.0
    assert channel["effective_differential_cross_section_uncertainty_m2_sr"] > 0.0
    evdf = channel["electron_velocity_distribution"]
    assert evdf["analysis_status"] in {"completed", "conditional_regularized_candidate"}
    assert evdf["recoil_included"] is False
    assert evdf["scattering_cross_section_model"] == (
        "polarized_single_scattering_Thomson_in_electron_rest_frame"
    )
    assert evdf["forward_model_recoil_included"] is False
    expected_density = channel["integrated_scattered_power_W"] / (
        2.0 * 0.1 * channel["effective_differential_cross_section_m2_sr"] * 0.01 * 0.5
    )
    assert channel["electron_density_m3"] == pytest.approx(expected_density, rel=1e-12)
    assert channel["electron_density_uncertainty_m3"] > 0.0


def test_thomson_profile_subanalysis_resolves_to_measured_spectrum_inference():
    from tokamak_analysis.analysis_registry import resolve_analysis

    entrypoint = resolve_analysis(46)
    assert callable(entrypoint)
    assert entrypoint.__name__ == "reconstruct_thomson_profile_from_spectra"
    x, y, _ = _spectral_case(100.0)
    result = entrypoint(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        model="non_collective_maxwellian",
    )
    assert result["registered_subanalysis"] == "Thomson spatial profile reconstruction from calibrated spectra"
    assert result["electron_temperature_profile_eV"][0] == pytest.approx(100.0, rel=0.05)


def test_instrument_response_uncertainty_is_monte_carlo_propagated():
    x,y,_=_spectral_case(100.0)
    response=np.eye(x.size)
    response_unc=np.full_like(response,1e-7)
    out=analyze_thomson_spectra(
        x,y[None,:],[0.1],laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,instrument_response=response,
        instrument_response_uncertainty=response_unc,
    )
    assert out["channels"][0]["instrument_response_uncertainty_supplied"] is True
    assert out["channels"][0]["instrument_response_uncertainty_propagated"] is True
    assert out["channels"][0]["electron_temperature_uncertainty_eV"] > 0


def test_polarized_thomson_cross_section_uses_full_consistent_geometry():
    from tokamak_analysis.thomson_scattering import thomson_differential_cross_section_m2_sr
    from tokamak_analysis.thomson_scattering import _ELECTRON_CLASSICAL_RADIUS_M
    sigma=thomson_differential_cross_section_m2_sr(
        np.pi/2,incident_polarization_vector=[0,0,1],
        incident_direction_vector=[1,0,0],scattered_direction_vector=[0,1,0],
    )
    assert sigma == pytest.approx(_ELECTRON_CLASSICAL_RADIUS_M**2,rel=1e-12)
    with pytest.raises(ValueError,match="perpendicular"):
        thomson_differential_cross_section_m2_sr(
            np.pi/2,incident_polarization_vector=[1,0,0],
            incident_direction_vector=[1,0,0],scattered_direction_vector=[0,1,0],
        )


def test_thomson_differential_cross_section_rejects_both_optical_geometry_endpoints():
    from tokamak_analysis.thomson_scattering import thomson_differential_cross_section_m2_sr

    for angle in (0.0, np.pi):
        with pytest.raises(ValueError, match="strictly between 0 and pi"):
            thomson_differential_cross_section_m2_sr(angle)


def test_thomson_density_formula_rejects_both_optical_geometry_endpoints():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    for angle in (0.0, np.pi):
        with pytest.raises(ValueError, match="strictly between 0 and pi"):
            density_from_thomson_scattered_power(
                scattered_power_W=1e-6,
                incident_power_W=1.0,
                interaction_length_m=0.1,
                scattering_angle_rad=angle,
                collection_solid_angle_sr=0.01,
                optical_efficiency=0.8,
            )


def test_instrument_response_accepts_kernel_array_and_callable():
    from tokamak_analysis.thomson_inference import _response_for_channel
    x,y,_=_spectral_case(100.0)
    kernel=np.array([0.2,0.6,0.2])
    matrix=_response_for_channel(kernel,0,1,x.size,x)
    measured=matrix@(y-25.0)+25.0
    out=analyze_thomson_spectra(
        x,measured[None,:],[0.1],laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,instrument_response=kernel,
    )
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(100.0,rel=0.05)
    callable_out=analyze_thomson_spectra(
        x,y[None,:],[0.1],laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,
        instrument_response=lambda wavelength: np.eye(len(wavelength)),
    )
    assert callable_out["electron_temperature_profile_eV"][0] == pytest.approx(100.0,rel=0.05)


def test_measured_instrument_response_is_mandatory():
    x,y,_=_spectral_case(100.0)
    with pytest.raises(ValueError,match="instrument-response matrix is required"):
        analyze_thomson_spectra(
            x,y[None,:],[0.1],laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2,
        )


def test_kappa_fit_is_invariant_to_absolute_intensity_scaling_and_integrates_full_model():
    x = np.linspace(492.0, 572.0, 801)
    sigma = 2.0 * 532.0 * np.sqrt(120.0 / 510998.95069) * np.sin(np.pi / 4)
    kappa = 4.0
    df = 2.0 * kappa - 1.0
    scale = sigma * np.sqrt((df - 2.0) / df)
    feature, _ = _kappa_bin_averaged_shape(x, 900.0, 532.0, sigma, kappa)
    signal = feature + 20.0

    first = analyze_thomson_spectra(
        x, signal[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        model="non_collective_kappa",
    )["channels"][0]
    second = analyze_thomson_spectra(
        x, (signal * 1e6)[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        model="non_collective_kappa",
    )["channels"][0]

    assert second["electron_temperature_eV"] == pytest.approx(first["electron_temperature_eV"], rel=1e-6)
    assert second["kappa_index"] == pytest.approx(first["kappa_index"], rel=1e-6)
    assert second["integrated_spectral_signal_input_units_nm"] == pytest.approx(
        first["integrated_spectral_signal_input_units_nm"] * 1e6, rel=1e-6
    )
    from scipy.special import gammaln
    fitted_kappa = first["kappa_index"]
    fitted_df = 2.0*fitted_kappa - 1.0
    fitted_scale = first["intrinsic_sigma_nm"]*np.sqrt((fitted_df-2.0)/fitted_df)
    expected_full_area = (
        first["amplitude"]*fitted_scale*np.sqrt(fitted_df*np.pi)
        * np.exp(gammaln(fitted_kappa-0.5)-gammaln(fitted_kappa))
    )
    assert first["integrated_scattered_power_W"] == pytest.approx(expected_full_area, rel=1e-10)
    assert first["spectral_bin_integration"] == "exact_Student_t_CDF_finite_volume_bin_averages"
    assert first["absolute_scattered_power_available"] is False
    assert first["integrated_spectral_signal_unit"] == "input spectral-intensity unit × nm"


def test_spectral_inference_rejects_boolean_and_string_measurements():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="real numeric values"):
        analyze_thomson_spectra(
            x, y.astype(str)[None, :], [0.1],
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2,
            instrument_response=np.eye(x.size),
        )
    with pytest.raises(ValueError, match="real numeric values"):
        analyze_thomson_spectra(
            x, y.astype(bool)[None, :], [0.1],
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2,
            instrument_response=np.eye(x.size),
        )


def test_thomson_scalar_physical_inputs_reject_booleans():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="laser_wavelength_nm"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=True,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        )
    with pytest.raises(ValueError, match="scattering_angle_rad"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=True, instrument_response=np.eye(x.size),
        )


def test_descriptive_moments_report_negative_residual_clipping_caveat():
    x = np.linspace(500.0, 564.0, 401)
    y = 100.0*np.exp(-0.5*((x-532.0)/8.0)**2) + 10.0
    y[0] = 0.0
    y[-1] = 0.0
    result = analyze_thomson_spectra(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        model="non_collective_maxwellian",
    )
    out = result["channels"][0]["spectral_distribution"]
    assert "projected_evdf_spectral_moment_closure" in out
    closure = out["projected_evdf_spectral_moment_closure"]
    assert closure["status"] == "conditional_moment_closure"
    assert np.isfinite(closure["centroid_closure_residual_nm"])
    assert np.isfinite(closure["variance_closure_residual_nm2"])
    assert "physical_parameter_extraction" not in out
    assert "negative_area_fraction_removed_by_clipping" in out
    assert "clipping" in out["moment_clipping_bias_warning"].lower()

def test_collective_only_inputs_are_rejected_in_optical_analysis_1():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="Collective-only inputs are not accepted"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            model="non_collective_maxwellian", ion_species=["D+"],
            ion_fractions=[1.0], collective_initial_density_m3=1.0e18,
        )

def test_nonrelativistic_maxwellian_rejects_temperature_outside_declared_regime():
    laser = 532.0
    angle = np.pi / 2
    te = 6000.0
    sigma = 2.0 * laser * np.sqrt(te / 510998.95069) * np.sin(angle / 2)
    x = np.linspace(laser - 350.0, laser + 350.0, 1401)
    y = 1200.0 * np.exp(-0.5 * ((x - laser) / sigma) ** 2) + 25.0
    with pytest.raises(ValueError, match="not safely small relative to the electron rest energy"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=laser,
            scattering_angle_rad=angle, instrument_response=np.eye(x.size),
        )


def test_nonrelativistic_maxwellian_rejects_large_flow_projection():
    laser = 532.0
    angle = np.pi / 2
    velocity = 0.2 * 299792458.0
    center = laser - velocity / 299792458.0 * laser * 2.0 * np.sin(angle / 2)
    x = np.linspace(laser - 250.0, laser + 250.0, 1401)
    sigma = 2.0 * laser * np.sqrt(100.0 / 510998.95069) * np.sin(angle / 2)
    y = 1200.0 * np.exp(-0.5 * ((x - center) / sigma) ** 2) + 25.0
    with pytest.raises(ValueError, match="outside the declared"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=laser,
            scattering_angle_rad=angle, instrument_response=np.eye(x.size),
        )


def test_fit_identifiability_gate_rejects_singular_covariance():
    from tokamak_analysis.thomson_inference import _fit_identifiability
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])
    result = _fit_identifiability(covariance, np.array([0.1, -0.1, 0.0]), None, 2)
    assert result["fit_accepted"] is False
    assert result["fit_quality_status"] == "rejected_unidentifiable_or_invalid"

def test_collective_default_bounds_are_scaled_to_user_initial_estimates():
    from tokamak_analysis.thomson_inference import _default_collective_bounds
    density, temperature = _default_collective_bounds(3e18, 250.0)
    np.testing.assert_allclose(density, [3e16, 3e20])
    np.testing.assert_allclose(temperature, [2.5, 25000.0])


def test_fit_identifiability_rejects_optimizer_failure_even_with_finite_covariance():
    from tokamak_analysis.thomson_inference import _fit_identifiability

    result = _fit_identifiability(
        np.eye(2), np.array([0.01, -0.01, 0.0]), None, 2,
        optimizer_success=False,
    )
    assert result["fit_accepted"] is False
    assert any("optimizer" in reason for reason in result["fit_quality_reasons"])


def test_fit_identifiability_rejects_rank_deficient_or_ill_conditioned_jacobian():
    from tokamak_analysis.thomson_inference import _fit_identifiability

    rank_deficient = _fit_identifiability(
        np.eye(2), np.array([0.01, -0.01, 0.0]), None, 2,
        jacobian_rank=1,
    )
    assert rank_deficient["fit_accepted"] is False
    assert any("Jacobian is rank deficient" in reason for reason in rank_deficient["fit_quality_reasons"])

    ill_conditioned = _fit_identifiability(
        np.eye(2), np.array([0.01, -0.01, 0.0]), None, 2,
        jacobian_rank=2, jacobian_condition_number=1e13,
    )
    assert ill_conditioned["fit_accepted"] is False
    assert any("Jacobian is ill-conditioned" in reason for reason in ill_conditioned["fit_quality_reasons"])


def test_fit_identifiability_reports_poor_reduced_chi_square_without_arbitrary_rejection():
    from tokamak_analysis.thomson_inference import _fit_identifiability

    result = _fit_identifiability(
        np.eye(2),
        np.array([10.0, -10.0, 8.0, -8.0]),
        np.ones(4),
        2,
    )
    assert result["fit_accepted"] is True
    assert result["fit_reduced_chi_square"] > 10.0
    assert result["fit_reduced_chi_square_policy"].startswith("diagnostic_only")
    assert not any("does not adequately explain the data" in reason
                   for reason in result["fit_quality_reasons"])


def test_fit_identifiability_rejects_mismatched_uncertainty_shape():
    from tokamak_analysis.thomson_inference import _fit_identifiability

    result = _fit_identifiability(
        np.eye(2), np.array([0.1, -0.1, 0.0]), np.ones(2), 2
    )
    assert result["fit_accepted"] is False
    assert any("does not match the residual" in reason
               for reason in result["fit_quality_reasons"])


def test_joint_multi_channel_gls_propagates_cross_channel_spectral_covariance():
    x=np.linspace(492.0,572.0,161)
    _,y1,_=_spectral_case(100.0,n=x.size)
    _,y2,_=_spectral_case(150.0,n=x.size)
    n=x.size
    base=np.eye(2*n)*0.5**2
    cross=np.eye(n)*0.2*0.5**2
    base[:n,n:]=cross
    base[n:,:n]=cross
    out=analyze_thomson_spectra(
        x,np.vstack([y1,y2]),[0.1,0.2],
        laser_wavelength_nm=532.0,scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(n),joint_spectral_covariance=base,
    )
    joint=out["joint_inference"]
    assert joint["method"]=="joint generalized least squares across all channels"
    assert joint["joint_parameter_covariance"].shape==(10,10)
    assert joint["physical_parameter_covariance"].shape==(4,4)
    assert joint["total_physical_parameter_covariance"][0,2] != pytest.approx(0.0,abs=1e-18)
    np.testing.assert_allclose(out["electron_temperature_profile_eV"],[100.0,150.0],rtol=0.05)
    assert all(ch["joint_fit_used"] for ch in out["channels"])


def test_joint_gls_reports_large_reduced_chi_square_without_arbitrary_rejection():
    x, y1, _ = _spectral_case(100.0, n=161)
    _, y2, _ = _spectral_case(140.0, n=161)
    perturbation = 8.0 * np.sin(np.linspace(0.0, 24.0 * np.pi, x.size))
    measured = np.vstack([y1 + perturbation, y2 - perturbation])
    covariance = np.eye(measured.size) * 0.1**2

    out = analyze_thomson_spectra(
        x, measured, [0.1, 0.2], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        joint_spectral_covariance=covariance,
    )

    joint = out["joint_inference"]
    assert joint["reduced_chi_square"] > 10.0
    assert joint["reduced_chi_square_policy"].startswith("diagnostic_only")
    assert all(channel["joint_fit_used"] for channel in out["channels"])
    assert all(channel["fit_accepted"] for channel in out["channels"])


def test_joint_multi_channel_gls_rejects_non_positive_definite_covariance():
    x,y,_=_spectral_case(100.0,n=81)
    covariance=np.eye(2*x.size)
    covariance[0,1]=covariance[1,0]=2.0
    with pytest.raises(ValueError,match="positive definite"):
        analyze_thomson_spectra(
            x,np.vstack([y,y]),[0.1,0.2],laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2,instrument_response=np.eye(x.size),
            joint_spectral_covariance=covariance,
        )


def test_joint_multi_channel_gls_supports_kappa_projected_spectra():
    x=np.linspace(492.0,572.0,161)
    kappa=4.0
    def spectrum(te):
        sigma=2.0*532.0*np.sqrt(te/510998.95069)*np.sin(np.pi/4)
        df=2.0*kappa-1.0
        scale=sigma*np.sqrt((df-2.0)/df)
        feature, _ = _kappa_bin_averaged_shape(x, 900.0, 532.0, sigma, kappa)
        return feature+20.0
    y=np.vstack([spectrum(100.0),spectrum(150.0)])
    n=x.size
    covariance=np.eye(2*n)*0.4**2
    covariance[:n,n:]=np.eye(n)*0.15*0.4**2
    covariance[n:,:n]=np.eye(n)*0.15*0.4**2
    out=analyze_thomson_spectra(
        x,y,[0.1,0.2],laser_wavelength_nm=532.0,scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(n),model="non_collective_kappa",
        joint_spectral_covariance=covariance,
    )
    assert out["joint_inference"]["joint_parameter_count_per_channel"]==6
    np.testing.assert_allclose(out["electron_temperature_profile_eV"],[100.0,150.0],rtol=0.08)
    assert all("kappa_index" in ch for ch in out["channels"])


def test_thomson_fit_reports_separate_model_discrepancy_covariance():
    x,y,_=_spectral_case(100.0,n=161)
    perturbation=0.12*np.sin(np.linspace(0.0,14.0*np.pi,x.size))
    measured=y+perturbation
    out=analyze_thomson_spectra(
        x,measured[None,:],[0.1],laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,instrument_response=np.eye(x.size),
        signal_uncertainty=np.full((1,x.size),0.1),
    )
    channel=out["channels"][0]
    assert "model_discrepancy_covariance" in channel
    assert channel["model_discrepancy_scale"] >= 0
    assert channel["total_fit_covariance_including_discrepancy"].shape==(5,5)


def test_joint_spectral_covariance_is_mutually_exclusive_with_pointwise_uncertainty():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="not both"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            signal_uncertainty=np.ones((1, x.size)),
            joint_spectral_covariance=np.eye(x.size),
        )


def test_density_relative_covariance_does_not_conflict_with_area_uncertainty():
    x, y, _ = _spectral_case(100.0)
    out = analyze_thomson_spectra(
        x, (y * 1e-18)[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0, 0, 1], incident_direction=[1, 0, 0],
        scattered_direction=[0, 1, 0], spectral_power_density_calibrated=True,
        density_relative_covariance=np.zeros((6, 6)),
    )
    assert out["channels"][0]["electron_density_m3"] > 0.0
    uncertainty = out["channels"][0]["electron_density_uncertainty_m3"]
    assert np.isfinite(uncertainty) and uncertainty >= 0.0


def test_joint_gls_recomputes_absolute_density_from_joint_integrated_power():
    x, y1, _ = _spectral_case(100.0)
    _, y2, _ = _spectral_case(150.0)
    spectra = np.vstack([y1, y2]) * 1e-18
    n = spectra.size
    out = analyze_thomson_spectra(
        x, spectra, [0.1, 0.2], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        joint_spectral_covariance=np.eye(n) * (1e-18)**2,
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0, 0, 1], incident_direction=[1, 0, 0],
        scattered_direction=[0, 1, 0], spectral_power_density_calibrated=True,
    )
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power
    for channel in out["channels"]:
        assert channel["fit_quality_status"] == "accepted_joint_gls_with_channel_marginal_diagnostics"
        assert channel["fit_accepted"] is True
        assert channel["joint_fit_reduced_chi_square"] == pytest.approx(
            out["joint_inference"]["reduced_chi_square"]
        )
        assert channel["fit_residual_rms"] == pytest.approx(
            np.sqrt(np.mean(channel["residual"]**2))
        )
        expected = density_from_thomson_scattered_power(
            channel["integrated_scattered_power_W"], 2.0, 0.1, np.pi / 2,
            1e-3, optical_efficiency=0.5,
            incident_polarization_vector=[0, 0, 1],
            incident_direction_vector=[1, 0, 0],
            scattered_direction_vector=[0, 1, 0],
            scattered_power_uncertainty_W=channel["integrated_scattered_power_uncertainty_W"],
        )
        assert channel["electron_density_m3"] == pytest.approx(expected["electron_density_m3"], rel=1e-10)


def test_joint_gls_preserves_laser_and_geometry_uncertainty_components():
    x, y, _ = _spectral_case(110.0)
    out = analyze_thomson_spectra(
        x, y[None, :], [0.12], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        joint_spectral_covariance=np.eye(x.size) * 0.8**2,
        laser_wavelength_uncertainty_nm=0.01,
        scattering_angle_uncertainty_rad=1e-3,
    )
    channel = out["channels"][0]
    assert channel["joint_fit_used"] is True
    assert channel["electron_temperature_external_uncertainty_eV"] > 0.0
    assert channel["electron_velocity_projection_external_uncertainty_m_s"] > 0.0
    assert channel["electron_temperature_uncertainty_eV"] >= channel["electron_temperature_external_uncertainty_eV"]
    assert channel["electron_velocity_projection_uncertainty_m_s"] >= channel["electron_velocity_projection_external_uncertainty_m_s"]


def test_beam_direction_component_uncertainty_is_rejected_outside_collective_model():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="supported only by the relativistic"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            incident_direction_uncertainty=[1e-4, 1e-4, 1e-4],
        )


def test_joint_density_relative_covariance_preserves_cross_channel_density_covariance():
    x, y1, _ = _spectral_case(100.0)
    _, y2, _ = _spectral_case(140.0)
    spectra = np.vstack([y1, y2]) * 1e-18
    marginal = np.eye(6) * 1e-4
    cross = np.eye(6) * 0.5e-4
    joint_density_covariance = np.block([[marginal, cross], [cross, marginal]])
    out = analyze_thomson_spectra(
        x, spectra, [0.1, 0.2], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0, 0, 1], incident_direction=[1, 0, 0],
        scattered_direction=[0, 1, 0], spectral_power_density_calibrated=True,
        joint_density_relative_covariance=joint_density_covariance,
    )
    density_covariance = out["electron_density_joint_covariance_m6"]
    assert out["electron_density_cross_channel_covariance_available"] is True
    assert density_covariance.shape == (2, 2)
    assert density_covariance[0, 1] > 0.0
    for index, channel in enumerate(out["channels"]):
        assert density_covariance[index, index] == pytest.approx(
            channel["electron_density_uncertainty_m3"]**2, rel=1e-10
        )


def test_joint_density_covariance_rejects_incompatible_spectral_covariance():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="zero scattered-power rows and columns"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            joint_spectral_covariance=np.eye(x.size),
            joint_density_relative_covariance=np.eye(6),
        )


def test_lsf_variance_adds_to_density_covariance_without_rescaling_cross_terms():
    x, y, _ = _spectral_case(100.0, n=161)
    response = np.eye(x.size)
    response_uncertainty = np.full_like(response, 1e-7)
    relative_covariance = np.eye(6) * 1e-4
    relative_covariance[0, :] = 0.0
    relative_covariance[:, 0] = 0.0
    out = analyze_thomson_spectra(
        x, (y * 1e-18)[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=response,
        instrument_response_uncertainty=response_uncertainty,
        density_relative_covariance=relative_covariance,
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0, 0, 1], incident_direction=[1, 0, 0],
        scattered_direction=[0, 1, 0], spectral_power_density_calibrated=True,
    )
    channel = out["channels"][0]
    assert channel["integrated_scattered_power_lsf_uncertainty_W"] > 0.0
    covariance = relative_covariance.copy()
    covariance[0, 0] += (
        channel["integrated_scattered_power_lsf_uncertainty_W"]
        / channel["integrated_scattered_power_W"]
    ) ** 2
    gradient = np.array([1.0, -1.0, -1.0, -1.0, -1.0, -1.0])
    expected_sigma = channel["electron_density_m3"] * np.sqrt(gradient @ covariance @ gradient)
    assert channel["electron_density_uncertainty_m3"] == pytest.approx(expected_sigma, rel=1e-8)


def test_per_channel_density_covariance_cannot_be_combined_with_joint_spectral_covariance():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="do not combine it with joint_spectral_covariance"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            density_relative_covariance=np.eye(6),
            joint_spectral_covariance=np.eye(x.size),
        )



def test_primary_thomson_inference_includes_spectral_distribution_diagnostics():
    x, y, _ = _spectral_case(120.0)
    out = analyze_thomson_spectra(
        x, y[None, :], [0.0], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        signal_uncertainty=np.full((1, x.size), 1.0),
    )
    distribution = out["channels"][0]["spectral_distribution"]
    assert distribution["analysis_status"] == "completed"
    assert distribution["background_source"].startswith("background nuisance")
    assert "physical_parameter_extraction" not in distribution
    assert "spectral_distribution" in out
    evdf = out["channels"][0]["electron_velocity_distribution"]
    assert evdf["analysis_status"] in {"completed", "unavailable_for_this_channel"}
    assert "electron_velocity_distribution" in out
    if evdf["analysis_status"] == "completed":
        assert evdf["scattering_vector_projection_only"] is True
        assert evdf["absolute_density_identified"] is False
        assert np.trapz(evdf["evdf_parallel_s_per_m"], evdf["velocity_parallel_m_s"]) == pytest.approx(1.0, rel=2e-5)
        assert evdf["regularization_selection"] == "generalized_cross_validation_over_logarithmic_candidate_grid"


def test_rejected_primary_fit_withholds_dependent_thomson_subanalyses(monkeypatch):
    import tokamak_analysis.thomson_inference as inference
    import tokamak_analysis.thomson_distribution as distribution_module
    import tokamak_analysis.thomson_evdf as evdf_module

    original_identifiability = inference._fit_identifiability

    def reject_primary_fit(*args, **kwargs):
        result = original_identifiability(*args, **kwargs)
        result["fit_accepted"] = False
        result["fit_quality_status"] = "rejected_unidentifiable_or_invalid"
        result["fit_quality_reasons"].append("forced primary-fit rejection for regression test")
        return result

    def forbidden_subanalysis(*args, **kwargs):
        pytest.fail("A dependent subanalysis must not run after primary-fit rejection.")

    monkeypatch.setattr(inference, "_fit_identifiability", reject_primary_fit)
    monkeypatch.setattr(
        distribution_module, "analyze_thomson_spectral_distribution",
        forbidden_subanalysis,
    )
    monkeypatch.setattr(
        evdf_module, "reconstruct_thomson_evdf",
        forbidden_subanalysis,
    )

    x, y, _ = _spectral_case(120.0)
    out = analyze_thomson_spectra(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
    )

    channel = out["channels"][0]
    assert channel["fit_accepted"] is False
    assert channel["spectral_distribution"]["analysis_status"] == "unavailable_primary_fit_rejected"
    assert channel["electron_velocity_distribution"]["analysis_status"] == "unavailable_primary_fit_rejected"
    assert out["spectral_distribution"][0]["analysis_status"] == "unavailable_primary_fit_rejected"
    assert out["electron_velocity_distribution"][0]["analysis_status"] == "unavailable_primary_fit_rejected"


def test_thomson_evdf_can_be_explicitly_disabled_without_changing_primary_physics_fit():
    x, y, _ = _spectral_case(100.0)
    out = analyze_thomson_spectra(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        enable_evdf_reconstruction=False,
    )
    assert out["channels"][0]["electron_velocity_distribution"]["analysis_status"] == "not_requested"
    assert np.isfinite(out["electron_temperature_profile_eV"][0])


def test_thomson_spatial_subanalysis_reconstructs_all_identifiable_spectral_profiles():
    from tokamak_analysis.thomson_workflow import reconstruct_thomson_profile_from_spectra

    radii = np.array([0.0, 0.1, 0.2])
    spectra = []
    wavelength = None
    for temperature in (80.0, 120.0, 160.0):
        wavelength, signal, _ = _spectral_case(temperature)
        spectra.append(signal)
    result = reconstruct_thomson_profile_from_spectra(
        wavelength, np.asarray(spectra), radii,
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2,
        instrument_response=np.eye(wavelength.size), model="non_collective_maxwellian",
        signal_uncertainty=np.ones((3, wavelength.size)),
        profile_reconstruction_method="both",
        profile_reconstruction_points=41,
        profile_reconstruction_mc_samples=20,
    )
    reconstructed = result["spatial_profile_reconstruction"]["profiles"]["electron_temperature_eV"]
    assert reconstructed["methods"]["pchip"]["status"] == "completed"
    assert reconstructed["methods"]["pchip"]["values"].shape == (41,)
    assert reconstructed["methods"]["smoothing_spline"]["status"] == "insufficient_spatial_channels"



def test_physical_model_must_be_selected_explicitly():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="Select a Thomson physical model explicitly"):
        _analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        )


def test_relativistic_fit_does_not_mutate_caller_direction_arrays():
    from tokamak_analysis.thomson_kinetic_models import fit_relativistic_maxwell_juttner

    wavelength = np.linspace(500.0, 564.0, 32)
    signal = np.ones_like(wavelength)
    incident = np.array([2.0, 0.0, 0.0])
    scattered = np.array([0.0, 3.0, 0.0])
    incident_before = incident.copy()
    scattered_before = scattered.copy()
    with pytest.raises(ValueError, match="inconsistent"):
        fit_relativistic_maxwell_juttner(
            wavelength, signal, 532.0, np.pi / 3.0,
            np.eye(wavelength.size),
            incident_direction=incident, scattered_direction=scattered,
            incident_polarization=[0.0, 0.0, 1.0],
        )
    np.testing.assert_array_equal(incident, incident_before)
    np.testing.assert_array_equal(scattered, scattered_before)


def test_shared_instrument_response_uncertainty_retains_cross_channel_covariance():
    x, y, _ = _spectral_case(100.0, n=81)
    response = np.eye(x.size)
    response_uncertainty = np.full_like(response, 1e-8)
    result = analyze_thomson_spectra(
        x, np.vstack([y, y]), [0.1, 0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2,
        instrument_response=response,
        instrument_response_uncertainty=response_uncertainty,
        instrument_response_mc_samples=20,
    )
    covariance = result["shared_instrument_response_parameter_covariance"]
    assert result["shared_instrument_response_cross_channel_covariance_available"] is True
    assert covariance.shape == (6, 6)
    assert result["shared_instrument_response_parameter_covariance_order"][0] == (
        "channel_0_electron_temperature_eV"
    )
    assert covariance[0, 3] > 0.0


def test_scattering_angle_uncertainty_is_not_dropped_with_density_relative_covariance():
    x, y, _ = _spectral_case(100.0, n=161)
    common = dict(
        wavelength_nm=x, spectra=(y * 1e-18)[None, :], radius_m=[0.1],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 3,
        instrument_response=np.eye(x.size),
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0, 1, 0], incident_direction=[1, 0, 0],
        scattered_direction=[0.5, np.sqrt(3.0) / 2.0, 0.0], spectral_power_density_calibrated=True,
        density_relative_covariance=np.eye(6) * 1e-6,
    )
    without_angle = analyze_thomson_spectra(**common)["channels"][0]
    with_angle = analyze_thomson_spectra(
        **common, scattering_angle_uncertainty_rad=0.01
    )["channels"][0]
    assert with_angle["electron_density_uncertainty_m3"] > without_angle["electron_density_uncertainty_m3"]


def test_joint_density_covariance_retains_common_scattering_angle_uncertainty():
    x, y, _ = _spectral_case(100.0, n=161)
    response = np.eye(x.size)
    density_covariance = np.eye(12) * 1e-6
    density_power_indices = [0, 6]
    density_covariance[density_power_indices, :] = 0.0
    density_covariance[:, density_power_indices] = 0.0
    common = dict(
        wavelength_nm=x, spectra=np.vstack([y, y]) * 1e-18, radius_m=[0.1, 0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 3,
        instrument_response=response,
        instrument_response_uncertainty=np.full_like(response, 1e-8),
        instrument_response_mc_samples=20,
        instrument_response_mc_min_success_fraction=0.5,
        joint_spectral_covariance=np.eye(2 * x.size) * 1e-36,
        laser_power_w=2.0, interaction_length_m=0.1,
        collection_solid_angle_sr=1e-3, optical_efficiency=0.5,
        incident_polarization=[0.0, 1.0, 0.0],
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.5, np.sqrt(3.0) / 2.0, 0.0],
        spectral_power_density_calibrated=True,
        joint_density_relative_covariance=density_covariance,
    )
    without_angle = analyze_thomson_spectra(**common)
    with_angle = analyze_thomson_spectra(
        **common, scattering_angle_uncertainty_rad=0.01
    )
    assert with_angle["electron_density_uncertainty_profile_m3"][0] > (
        without_angle["electron_density_uncertainty_profile_m3"][0]
    )
    covariance = with_angle["electron_density_joint_covariance_m6"]
    assert covariance.shape == (2, 2)
    assert covariance[0, 1] > 0.0


def test_spectral_distribution_is_a_raw_spectrum_shape_analysis_without_physical_parameter_injection():
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    x, y, _ = _spectral_case(120.0)
    background = 25.0 + 0.1 * (x - 532.0)
    result = analyze_thomson_spectral_distribution(
        x, y,
        instrument_response_matrix=np.eye(x.size),
        background_signal=background,
    )
    assert "spectral_centroid_nm" in result
    assert "physical_parameter_extraction" not in result
    assert "instrument_corrected_sigma_nm" not in result
    with pytest.raises(TypeError):
        analyze_thomson_spectral_distribution(
            x, y,
            instrument_response_matrix=np.eye(x.size),
            background_signal=background,
            physical_parameters={"electron_temperature_eV": 120.0},
        )


def test_spectral_distribution_requires_explicit_background_source():
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution

    x, y, _ = _spectral_case(120.0)
    with pytest.raises(ValueError, match="background"):
        analyze_thomson_spectral_distribution(
            x, y,
            instrument_response_matrix=np.eye(x.size),
        )


def test_spectral_distribution_response_conserves_area_on_nonuniform_bins():
    from tokamak_analysis.thomson_distribution import _apply_response_matrix, _wavelength_bin_widths

    wavelength = np.array([500.0, 501.0, 503.0, 506.0, 510.0, 515.0, 521.0])
    signal = np.array([0.2, 0.8, 1.5, 2.0, 1.2, 0.5, 0.1])
    response = 0.8*np.eye(wavelength.size) + 0.2*np.roll(np.eye(wavelength.size), 1, axis=0)
    convolved = _apply_response_matrix(response, signal, wavelength)
    widths = _wavelength_bin_widths(wavelength)
    assert np.sum(convolved*widths) == pytest.approx(np.sum(signal*widths), rel=1e-12)




def test_optical_thomson_exposes_unweighted_fit_covariance_scope():
    x, y, _ = _spectral_case(100.0)
    out = _analyze_thomson_spectra(
        x, y[None, :], [0.12], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        model="non_collective_maxwellian",
    )
    channel = out["channels"][0]
    assert channel["spectral_weighting_status"] == "unweighted_least_squares_no_measurement_uncertainty_supplied"
    assert channel["fit_covariance_interpretation"] == "residual_variance_scaled_local_covariance_not_measurement_noise_calibrated"
    assert channel["background_model_status"] == "linear_intercept_and_slope_nuisance_background"


def test_result_classification_separates_primary_conditional_and_non_identifiable_outputs():
    x, y, _ = _spectral_case(100.0)
    result = analyze_thomson_spectra(
        x, y[None, :], [0.12], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
        enable_evdf_reconstruction=False,
    )
    contract = result["result_classification"]
    assert contract["contract_version"] == "1.0"
    assert contract["input_contract"] == "measured_device_spectra_assumed_prepared_and_calibrated"
    assert contract["instrument_calibration_or_correction_performed"] is False
    assert contract["experimental_validation_claimed"] is False
    channel = contract["channels"][0]
    primary_names = {item["parameter"] for item in channel["primary"]}
    conditional_names = {item["parameter"] for item in channel["conditional"]}
    non_identifiable_names = {item["parameter"] for item in channel["non_identifiable"]}
    assert "electron_temperature_eV" in primary_names
    assert "electron_density_m3" in conditional_names
    assert "unrestricted_three_dimensional_electron_velocity_distribution" in non_identifiable_names
    density_requirement = next(
        item for item in channel["conditional"] if item["parameter"] == "electron_density_m3"
    )
    assert "absolute spectral power density in W/nm" in density_requirement["required_auxiliary_inputs"]


def test_measured_response_matrix_is_rejected_not_renormalized():
    from tokamak_analysis.thomson_inference import _response_for_channel

    response = np.eye(7)
    response[0, 0] = 0.5
    original = response.copy()
    with pytest.raises(ValueError, match="columns must already sum to one"):
        _response_for_channel(response, 0, 1, 7, np.linspace(500.0, 506.0, 7))
    np.testing.assert_array_equal(response, original)


def test_optical_thomson_rejects_backscattering_endpoint_at_pi():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="strictly between 0 and pi"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi, instrument_response=np.eye(x.size),
            model="non_collective_maxwellian",
        )


def test_relativistic_maxwell_juttner_accepts_full_within_channel_spectral_covariance():
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(526.0, 538.0, 241)
    laser = 532.0
    angle = np.pi / 2
    incident = np.array([1.0, 0.0, 0.0])
    scattered = np.array([0.0, 1.0, 0.0])
    polarization = np.array([0.0, 1.0, 0.0])
    polarization_local = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    shape = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 200.0,
        flow_projection_m_s=2.0e4,
        incident_polarization=polarization_local,
    )["spectral_density_per_nm"]
    measured = 700.0 * shape + 0.04 + 0.001 * (wavelength - laser)
    rho = 0.25
    indices = np.arange(wavelength.size)
    covariance = (0.015**2) * rho ** np.abs(indices[:, None] - indices[None, :])

    out = analyze_thomson_spectra(
        wavelength, measured[None, :], [0.2],
        laser_wavelength_nm=laser, scattering_angle_rad=angle,
        instrument_response=np.eye(wavelength.size),
        incident_direction=incident, scattered_direction=scattered,
        incident_polarization=polarization,
        model="non_collective_relativistic_maxwell_juttner",
        joint_spectral_covariance=covariance,
        enable_evdf_reconstruction=False,
    )
    fit = out["channels"][0]
    assert fit["spectral_weighting_status"] == (
        "block_diagonal_channel_major_within_channel_generalized_least_squares"
    )
    assert fit["fit_covariance_interpretation"] == (
        "local_parameter_covariance_from_supplied_measurement_noise_model"
    )
    assert fit["fit_covariance"].shape == (5, 5)
    assert fit["electron_temperature_eV"] == pytest.approx(200.0, rel=0.08)


def test_relativistic_maxwell_juttner_full_cross_channel_covariance_is_integrated_by_canonical_api():
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(526.0, 538.0, 121)
    laser = 532.0
    angle = np.pi / 2
    incident = np.array([1.0, 0.0, 0.0])
    scattered = np.array([0.0, 1.0, 0.0])
    polarization = np.array([0.0, 1.0, 0.0])
    polarization_local = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    shape = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 200.0,
        incident_polarization=polarization_local,
    )["spectral_density_per_nm"]
    measured = 700.0 * shape + 0.04
    n = wavelength.size
    covariance = np.eye(2*n) * 0.015**2
    covariance[:n, n:] = np.eye(n) * 1e-6
    covariance[n:, :n] = np.eye(n) * 1e-6
    out = analyze_thomson_spectra(
        wavelength, np.vstack([measured, 1.2*measured]), [0.2, 0.3],
        laser_wavelength_nm=laser, scattering_angle_rad=angle,
        instrument_response=np.eye(n),
        incident_direction=incident, scattered_direction=scattered,
        incident_polarization=polarization,
        model="non_collective_relativistic_maxwell_juttner",
        joint_spectral_covariance=covariance,
        enable_evdf_reconstruction=False,
    )
    assert out["joint_inference"]["spectral_weighting_status"] == (
        "full_channel_major_generalized_least_squares_including_cross_channel_covariance"
    )
    assert out["joint_inference"]["integration_status"] == "used_by_canonical_multichannel_thomson_api"
    assert out["joint_inference"]["fit_covariance"].shape == (8, 8)
    assert out["channels"][0]["spectral_weighting_status"] == (
        "within_channel_marginal_gls_cross_channel_covariance_used_in_joint_shared_parameter_fit"
    )


def test_kappa_analysis_accepts_model_discrepancy_covariance():
    wavelength = np.linspace(492.0, 572.0, 241)
    center = 532.0
    sigma = 2.0 * 532.0 * np.sqrt(120.0 / 510998.95069) * np.sin(np.pi / 4.0)
    kappa = 3.5
    df = 2.0 * kappa - 1.0
    scale = sigma * np.sqrt((df - 2.0) / df)
    feature, _ = _kappa_bin_averaged_shape(wavelength, 900.0, center, sigma, kappa)
    signal = feature + 18.0 + 0.015 * (wavelength - center)
    covariance = np.eye(wavelength.size) * 0.04
    covariance += np.eye(wavelength.size, k=1) * 0.008
    covariance += np.eye(wavelength.size, k=-1) * 0.008

    result = _analyze_thomson_spectra(
        wavelength, signal[None, :], [0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
        model="non_collective_kappa",
        model_discrepancy_covariance=covariance,
    )

    channel = result["channels"][0]
    assert channel["spectral_covariance_used"] is True
    assert channel["spectral_covariance_interpretation"].startswith("researcher_supplied")
    assert channel["spectral_weighting_status"] == (
        "within_channel_gls_with_model_discrepancy_covariance_only"
    )
    assert channel["fit_covariance_interpretation"] == (
        "local_parameter_covariance_from_supplied_model_discrepancy_covariance_only"
    )
    assert channel["fit_covariance"].shape == (6, 6)
    assert np.isfinite(channel["reduced_chi_square"])


def test_kappa_fit_rejects_simultaneous_pointwise_and_full_covariance():
    from tokamak_analysis.thomson_inference import _fit_kappa

    wavelength = np.linspace(500.0, 564.0, 65)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 3.0) ** 2) + 0.1
    with pytest.raises(ValueError, match="pointwise uncertainty or spectral_covariance"):
        _fit_kappa(
            wavelength, signal, 532.0, np.pi / 2.0, np.eye(wavelength.size),
            np.ones(wavelength.size), spectral_covariance=np.eye(wavelength.size),
        )


def test_relativistic_maxwell_juttner_adds_model_discrepancy_to_joint_spectral_covariance():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(526.0, 538.0, 121)
    laser = 532.0
    angle = np.pi / 2
    incident = np.array([1.0, 0.0, 0.0])
    scattered = np.array([0.0, 1.0, 0.0])
    polarization = np.array([0.0, 1.0, 0.0])
    polarization_local = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    shape = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 200.0,
        flow_projection_m_s=2.0e4,
        incident_polarization=polarization_local,
    )["spectral_density_per_nm"]
    measured = 700.0 * shape + 0.04 + 0.001 * (wavelength - laser)
    measurement_covariance = np.eye(wavelength.size) * 0.015**2
    discrepancy_covariance = np.eye(wavelength.size) * 0.004**2

    out = analyze_thomson_spectra(
        wavelength, measured[None, :], [0.2],
        laser_wavelength_nm=laser, scattering_angle_rad=angle,
        instrument_response=np.eye(wavelength.size),
        incident_direction=incident, scattered_direction=scattered,
        incident_polarization=polarization,
        model="non_collective_relativistic_maxwell_juttner",
        joint_spectral_covariance=measurement_covariance,
        model_discrepancy_covariance=discrepancy_covariance,
        enable_evdf_reconstruction=False,
    )

    fit = out["channels"][0]
    assert fit["spectral_weighting_status"] == (
        "block_diagonal_channel_major_within_channel_generalized_least_squares"
    )
    assert fit["fit_covariance_interpretation"] == (
        "local_parameter_covariance_from_supplied_measurement_noise_model"
    )
    assert fit["fit_covariance"].shape == (5, 5)
    assert out["joint_inference"]["fit_covariance"].shape == (4, 4)



def test_kappa_profile_likelihood_uses_full_spectral_covariance_gls():
    from tokamak_analysis.thomson_inference import _kappa_profile_likelihood_diagnostic

    wavelength = np.linspace(-10.0, 10.0, 61)
    parameters = np.array([1.0, 0.15, 2.0, 4.0, 0.025, 0.001])
    amplitude, center, width, kappa, offset, slope = parameters
    degrees = 2.0 * kappa - 1.0
    student_scale = width * np.sqrt((degrees - 2.0) / degrees)
    feature, _ = _kappa_bin_averaged_shape(wavelength, amplitude, center, width, kappa)
    signal = feature + offset + slope * (wavelength - center)
    correlation = 0.25 ** np.abs(
        np.subtract.outer(np.arange(wavelength.size), np.arange(wavelength.size))
    )
    covariance = 2.5e-5 * correlation

    result = _kappa_profile_likelihood_diagnostic(
        wavelength, signal, np.eye(wavelength.size), parameters, None,
        spectral_covariance=covariance,
    )

    assert result["status"] in {"finite_profile_interval", "grid_censored"}
    assert result["noise_scale"] == (
        "researcher_supplied_full_within_channel_spectral_covariance_GLS"
    )
    finite_profile = np.isfinite(result["profile_delta_chi_square"])
    assert finite_profile.any()
    assert np.min(result["profile_delta_chi_square"][finite_profile]) == pytest.approx(0.0)
    with pytest.raises(ValueError, match="not both"):
        _kappa_profile_likelihood_diagnostic(
            wavelength, signal, np.eye(wavelength.size), parameters,
            np.full(wavelength.size, 0.01), spectral_covariance=covariance,
        )


def test_kappa_channel_fit_supports_joint_polynomial_detector_background():
    x = np.linspace(492.0, 572.0, 401)
    te = 120.0
    sigma = 2.0 * 532.0 * np.sqrt(te / 510998.95069) * np.sin(np.pi / 4)
    kappa = 4.0
    df = 2.0 * kappa - 1.0
    scale = sigma * np.sqrt((df - 2.0) / df)
    z = 2.0 * (x - x.min()) / np.ptp(x) - 1.0
    plasma, _ = _kappa_bin_averaged_shape(x, 900.0, 532.0, sigma, kappa)
    measured_background = 20.0 + 2.5*z - 1.2*(0.5*(3.0*z*z - 1.0))
    y = plasma + measured_background
    out = analyze_thomson_spectra(
        x, y[None, :], [0.15], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size),
        model="non_collective_kappa", background_model="polynomial",
        background_degree=2,
    )
    channel = out["channels"][0]
    assert channel["background_model"] == "polynomial"
    assert channel["background_basis"].shape == (x.size, 3)
    assert channel["fit_covariance"].shape == (7, 7)
    assert channel["fit_parameter_names"] == [
        "amplitude", "center_nm", "intrinsic_sigma_nm", "kappa_index",
        "background_coefficient_0", "background_coefficient_1",
        "background_coefficient_2",
    ]
    assert channel["electron_temperature_eV"] == pytest.approx(te, rel=0.08)
    assert channel["kappa_index"] == pytest.approx(kappa, rel=0.12)
    assert np.sqrt(np.mean(channel["residual"]**2)) < 0.5


def test_relativistic_mj_channel_fit_uses_selected_structured_background_and_joint_covariance():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    x = np.linspace(480.0, 584.0, 521)
    physical = relativistic_maxwell_juttner_spectrum(
        x, 532.0, np.pi / 2.0, 1000.0,
        flow_projection_m_s=1.5e4,
        incident_polarization=np.array([0.0, 1.0, 0.0]),
        radial_order=24, angular_order=12,
        require_full_spectrum=False,
    )
    z = (x - 532.0) / 52.0
    background = 18.0 + 2.0*z - 0.8*(0.5*(3.0*z*z - 1.0))
    measured = 2.5e4 * physical["spectral_density_per_nm"] + background
    out = _analyze_thomson_spectra(
        x, measured[None, :], [0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(x.size),
        signal_uncertainty=np.full((1, x.size), 0.5),
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        incident_polarization=[0.0, 1.0, 0.0],
        model="non_collective_relativistic_maxwell_juttner",
        background_model="polynomial", background_degree=2,
        include_compton_recoil=False,
    )
    channel = out["channels"][0]
    assert channel["background_model"] == "polynomial"
    assert channel["fit_covariance_includes_background_cross_terms"] is True
    assert channel["fit_covariance"].shape == (6, 6)
    assert channel["electron_temperature_eV"] == pytest.approx(1000.0, rel=0.15)
    assert channel["integrated_scattered_power_W"] > 0.0
    assert channel["integrated_scattered_power_uncertainty_W"] > 0.0


def test_joint_channel_gls_uses_selected_polynomial_background_basis():
    wavelength = np.linspace(492.0, 572.0, 241)
    z = 2.0*(wavelength-wavelength.min())/np.ptp(wavelength)-1.0
    p2 = 0.5*(3.0*z*z-1.0)
    sigma = 2.0*532.0*np.sqrt(120.0/510998.95069)*np.sin(np.pi/4)
    first = 1200.0*np.exp(-0.5*((wavelength-532.0)/sigma)**2) + 25.0 + 0.3*z + 0.4*p2
    second = 850.0*np.exp(-0.5*((wavelength-532.04)/sigma)**2) + 18.0 - 0.2*z + 0.25*p2
    spectra = np.vstack((first, second))
    n = wavelength.size
    covariance = np.eye(2*n)*0.25
    covariance[:n, n:] = np.eye(n)*0.025
    covariance[n:, :n] = np.eye(n)*0.025
    result = _analyze_thomson_spectra(
        wavelength, spectra, [0.1, 0.2],
        laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2,
        instrument_response=np.eye(n), model="non_collective_maxwellian",
        joint_spectral_covariance=covariance,
        background_model="polynomial", background_degree=2,
    )
    for channel in result["channels"]:
        assert channel["background_model"] == "polynomial"
        assert channel["background_coefficients"].shape == (3,)
        assert channel["background_coefficient_covariance"].shape == (3, 3)
        assert channel["physical_background_cross_covariance"].shape == (3, 3)
        assert channel["joint_fit_used"] is True
    assert result["joint_inference"]["covariance_ordering"] == "channel_major"

def test_full_channel_major_model_discrepancy_covariance_preserves_cross_channel_blocks():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(526.0, 538.0, 81)
    laser = 532.0
    angle = np.pi / 2
    incident = np.array([1.0, 0.0, 0.0])
    scattered = np.array([0.0, 1.0, 0.0])
    polarization = np.array([0.0, 1.0, 0.0])
    polarization_local = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    shape = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, angle, 200.0,
        flow_projection_m_s=2.0e4,
        incident_polarization=polarization_local,
        radial_order=16, angular_order=8,
    )["spectral_density_per_nm"]
    measured = np.vstack((
        700.0 * shape + 0.04 + 0.001 * (wavelength - laser),
        520.0 * shape + 0.03 - 0.0007 * (wavelength - laser),
    ))
    n = wavelength.size
    measurement_covariance = np.eye(2*n) * 0.015**2
    discrepancy_covariance = np.eye(2*n) * 0.004**2
    cross = np.eye(n) * (0.25 * 0.004**2)
    discrepancy_covariance[:n, n:] = cross
    discrepancy_covariance[n:, :n] = cross.T

    out = analyze_thomson_spectra(
        wavelength, measured, [0.15, 0.25],
        laser_wavelength_nm=laser, scattering_angle_rad=angle,
        instrument_response=np.eye(n),
        incident_direction=incident, scattered_direction=scattered,
        incident_polarization=polarization,
        model="non_collective_relativistic_maxwell_juttner",
        joint_spectral_covariance=measurement_covariance,
        model_discrepancy_covariance=discrepancy_covariance,
        enable_evdf_reconstruction=False,
    )

    assert out["joint_inference"]["spectral_weighting_status"] == (
        "full_channel_major_generalized_least_squares_including_cross_channel_covariance"
    )
    assert out["joint_inference"]["fit_covariance"].shape[0] > 0
    assert all(
        channel["spectral_weighting_status"]
        == "within_channel_marginal_gls_cross_channel_covariance_used_in_joint_shared_parameter_fit"
        for channel in out["channels"]
    )
    assert all(
        channel["fit_covariance_interpretation"]
        == "local_parameter_covariance_from_supplied_measurement_and_model_discrepancy_covariance"
        for channel in out["channels"]
    )


def test_full_channel_major_model_discrepancy_covariance_rejects_indefinite_input():
    wavelength = np.linspace(500.0, 564.0, 41)
    measured = np.ones((2, wavelength.size))
    discrepancy = np.eye(2*wavelength.size)
    discrepancy[:wavelength.size, wavelength.size:] = 2.0*np.eye(wavelength.size)
    discrepancy[wavelength.size:, :wavelength.size] = 2.0*np.eye(wavelength.size)
    with pytest.raises(ValueError, match="positive semidefinite"):
        analyze_thomson_spectra(
            wavelength, measured, [0.1, 0.2],
            laser_wavelength_nm=532.0, scattering_angle_rad=np.pi/2,
            instrument_response=np.eye(wavelength.size),
            model="non_collective_maxwellian",
            model_discrepancy_covariance=discrepancy,
        )



def test_canonical_relativistic_thomson_workflow_runs_compton_projected_evdf_inverse():
    from tokamak_analysis.thomson_kinetic_models import relativistic_maxwell_juttner_spectrum

    wavelength = np.linspace(500.0, 564.0, 81)
    laser = 532.0
    polarization = [0.0, 0.0, 1.0]
    physical = relativistic_maxwell_juttner_spectrum(
        wavelength, laser, np.pi / 2.0, 100.0,
        flow_projection_m_s=0.0,
        incident_polarization=polarization,
        radial_order=12, angular_order=8,
        include_compton_recoil=True,
    )
    measured = 1000.0 * physical["spectral_density_per_nm"] + 2.0
    result = analyze_thomson_spectra(
        wavelength, measured[None, :], [0.1],
        laser_wavelength_nm=laser,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full(wavelength.size, 0.01),
        model="non_collective_relativistic_maxwell_juttner",
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
        incident_polarization=polarization,
        include_compton_recoil=True,
        enable_evdf_reconstruction=True,
        evdf_velocity_points=81,
    )
    evdf = result["channels"][0]["electron_velocity_distribution"]
    assert evdf["analysis_status"] in {"completed", "conditional_regularized_candidate"}
    assert evdf["recoil_included"] is True
    assert evdf["forward_model_recoil_included"] is True
    assert evdf["scattering_cross_section_model"] == (
        "polarized_single_scattering_Klein_Nishina_in_electron_rest_frame"
    )
    assert evdf["absolute_density_inferred"] is False
    assert evdf["projected_evdf_s_m"] is None or np.all(
        np.isfinite(evdf["projected_evdf_s_m"])
    )


def test_primary_optical_fit_propagates_additive_systematic_covariance():
    wavelength, measured, _ = _spectral_case(120.0, n=241)
    systematic = np.eye(wavelength.size) * 0.01**2
    result = analyze_thomson_spectra(
        wavelength, measured[None, :], [0.2],
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0,
        instrument_response=np.eye(wavelength.size),
        signal_uncertainty=np.full((1, wavelength.size), 0.1),
        systematic_covariance=systematic,
    )
    channel = result["channels"][0]
    assert channel["systematic_covariance_supplied"] is True
    assert channel["systematic_covariance_included"] is True
    assert channel["systematic_covariance_model"] == (
        "researcher_supplied_additive_correlated_spectral_systematic_covariance"
    )
    assert channel["spectral_weighting_status"] == (
        "within_channel_gls_with_measurement_and_systematic_covariance"
    )
    assert channel["fit_covariance_interpretation"] == (
        "local_parameter_covariance_from_measurement_and_systematic_covariance"
    )
    assert np.all(np.isfinite(channel["fit_covariance"]))


def test_primary_thomson_finite_bin_integrators_match_normal_and_student_t_cdfs():
    from scipy.special import gammaln, ndtr, stdtr

    wavelength = np.array([500.0, 500.15, 500.42, 500.9, 501.1])
    edges = np.empty(wavelength.size + 1)
    edges[1:-1] = 0.5*(wavelength[:-1]+wavelength[1:])
    edges[0] = wavelength[0]-0.5*(wavelength[1]-wavelength[0])
    edges[-1] = wavelength[-1]+0.5*(wavelength[-1]-wavelength[-2])
    widths = np.diff(edges)

    gaussian, gaussian_probability = _gaussian_bin_averaged_shape(
        wavelength, 2.5, 500.55, 0.31
    )
    expected_gaussian = 2.5*np.sqrt(2.0*np.pi)*0.31*np.diff(
        ndtr((edges-500.55)/0.31)
    )
    np.testing.assert_allclose(gaussian*widths, expected_gaussian, rtol=1e-13, atol=1e-15)
    np.testing.assert_allclose(gaussian_probability, np.diff(ndtr((edges-500.55)/0.31)))

    amplitude, centre, sigma, kappa = 3.0, 500.55, 0.31, 4.2
    degrees = 2.0*kappa-1.0
    scale = sigma*np.sqrt((degrees-2.0)/degrees)
    kappa_shape, kappa_probability = _kappa_bin_averaged_shape(
        wavelength, amplitude, centre, sigma, kappa
    )
    expected_probability = np.diff(stdtr(degrees, (edges-centre)/scale))
    shape_integral = scale*np.sqrt(degrees*np.pi)*np.exp(
        gammaln(kappa-0.5)-gammaln(kappa)
    )
    np.testing.assert_allclose(kappa_probability, expected_probability, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(
        kappa_shape*widths, amplitude*shape_integral*expected_probability,
        rtol=1e-12, atol=1e-14,
    )


def test_structured_kappa_background_gets_profile_likelihood_with_all_nuisance_terms():
    from tokamak_analysis.thomson_inference import _kappa_bin_averaged_shape
    x = np.linspace(492.0, 572.0, 321)
    basis_model = "polynomial"
    z = 2.0 * (x - x.min()) / np.ptp(x) - 1.0
    basis = np.polynomial.legendre.legvander(z, 2)
    feature, _ = _kappa_bin_averaged_shape(x, 900.0, 532.0, 2.8, 3.5)
    y = feature + basis @ np.array([18.0, 2.5, -1.2])
    out = analyze_thomson_spectra(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2.0, instrument_response=np.eye(x.size),
        model="non_collective_kappa", background_model=basis_model,
        background_degree=2, signal_uncertainty=np.full((1, x.size), 0.2),
    )
    profile = out["channels"][0]["kappa_profile_likelihood"]
    assert profile["status"] in {"finite_profile_interval", "grid_censored"}
    assert profile["profiled_background_model"] == "polynomial"
    assert profile["profiled_background_parameter_count"] == 3
    assert profile["profile_grid_kappa"].size >= 41


def test_structured_background_gls_covariance_keeps_absolute_measurement_scale():
    from tokamak_analysis.thomson_background import fit_structured_background
    x = np.linspace(0.0, 1.0, 40)
    wavelength = x + 500.0
    z = 2.0 * (wavelength - wavelength.min()) / np.ptp(wavelength) - 1.0
    design = np.column_stack([np.ones_like(x), z])
    y = design @ np.array([2.0, -0.5])
    covariance = np.eye(x.size) * 0.04
    result = fit_structured_background(
        wavelength, y, model="linear", covariance=covariance,
    )
    expected = np.linalg.inv(design.T @ np.linalg.solve(covariance, design))
    np.testing.assert_allclose(
        result["coefficient_covariance"], expected, rtol=1e-10, atol=1e-12,
    )

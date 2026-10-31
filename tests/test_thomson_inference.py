import numpy as np
import pytest

from tokamak_analysis.thomson_inference import analyze_thomson_spectra as _analyze_thomson_spectra


def analyze_thomson_spectra(*args, **kwargs):
    """Keep legacy test fixtures explicit while testing the production API separately."""
    kwargs.setdefault("model", "non_collective_maxwellian")
    return _analyze_thomson_spectra(*args, **kwargs)


def _spectral_case(te_ev, center_nm=532.0, laser_nm=532.0, angle=np.pi/2, n=801):
    x = np.linspace(laser_nm-40.0, laser_nm+40.0, n)
    sigma = 2.0*laser_nm*np.sqrt(te_ev/510998.95069)*np.sin(angle/2.0)
    y = 1200.0*np.exp(-0.5*((x-center_nm)/sigma)**2)+25.0+0.1*(x-center_nm)
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
    intrinsic = 1200.0 * np.exp(-0.5 * ((x - 532.0) / sigma) ** 2)
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


def test_optical_analysis_1_rejects_collective_dielectric_models():
    x, y, _ = _spectral_case(100.0)
    for model in ("collective_maxwellian", "collective_kappa"):
        with pytest.raises(ValueError, match="Optical collective dielectric models are outside this shortlist"):
            analyze_thomson_spectra(
                x, y[None, :], [0.1], laser_wavelength_nm=532.0,
                scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
                model=model,
            )

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
    y = 900.0*(1.0+((x-532.0)/scale)**2/df)**(-kappa)+20.0
    out = analyze_thomson_spectra(
        x,y[None,:],[0.15],laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi/2,instrument_response=np.eye(x.size),model="non_collective_kappa",
    )
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(te,rel=0.05)
    assert out["channels"][0]["kappa_index"] == pytest.approx(kappa,rel=0.05)
    assert 0.0 < out["channels"][0]["captured_spectral_probability"] <= 1.0


def test_legacy_relativistic_model_alias_is_rejected():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="Unsupported Thomson model"):
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


def test_relativistic_model_rejects_unpropagated_laser_and_angle_uncertainties():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="uncertainty propagation is not implemented"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            model="non_collective_relativistic_maxwell_juttner",
            incident_direction=[1.0, 0.0, 0.0],
            scattered_direction=[0.0, 1.0, 0.0],
            incident_polarization=[0.0, 0.0, 1.0],
            laser_wavelength_uncertainty_nm=0.01,
        )


def test_thomson_profile_subanalysis_resolves_to_measured_spectrum_inference():
    from tokamak_analysis.analysis_registry import resolve_analysis

    entrypoint = resolve_analysis(46)
    assert callable(entrypoint)
    assert entrypoint.__name__ == "reconstruct_thomson_profile_from_spectra"
    x, y, _ = _spectral_case(100.0)
    result = entrypoint(
        x, y[None, :], [0.1], laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
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
    signal = 900.0 * (1.0 + ((x - 532.0) / scale) ** 2 / df) ** (-kappa) + 20.0

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
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution
    x=np.linspace(500.0,564.0,401)
    y=100.0*np.exp(-0.5*((x-532.0)/8.0)**2)+10.0
    y[0]=0.0
    y[-1]=0.0
    out=analyze_thomson_spectral_distribution(x,y,instrument_sigma_nm=0.2,background_fraction=0.1)
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
        return 900.0*(1.0+((x-532.0)/scale)**2/df)**(-kappa)+20.0
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
    assert out["channels"][0]["electron_density_uncertainty_m3"] == pytest.approx(0.0)


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
    with pytest.raises(ValueError, match="not supported by the selected non-collective optical Thomson model"):
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
    with pytest.raises(ValueError, match="do not combine it with joint_spectral_covariance"):
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
    relative_covariance[0, 1] = relative_covariance[1, 0] = 0.5e-4
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
    assert "spectral_distribution" in out


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
        instrument_response=np.eye(wavelength.size),
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
            incident_direction=incident, scattered_direction=scattered,
            incident_polarization=[0.0, 0.0, 1.0],
            instrument_response=np.eye(wavelength.size),
        )
    np.testing.assert_array_equal(incident, incident_before)
    np.testing.assert_array_equal(scattered, scattered_before)

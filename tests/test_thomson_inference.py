import numpy as np
import pytest

from tokamak_analysis.thomson_inference import analyze_thomson_spectra


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


def test_collective_mode_never_runs_without_explicit_ion_species():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="ion species"):
        analyze_thomson_spectra(
            x, y[None,:], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi/2, instrument_response=np.eye(x.size), model="collective_maxwellian",
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


def test_projected_relativistic_maxwell_juttner_model_is_available():
    x=np.linspace(492.0,572.0,801)
    laser=532.0
    angle=np.pi/2
    te=300.0
    theta=te/510998.95069
    sine=np.sin(angle/2.0)
    ratio=x/laser
    beta=(ratio-1.0)/(sine*(ratio+1.0))
    valid=np.abs(beta)<1.0
    gamma=1.0/np.sqrt(1.0-beta[valid]**2)
    marginal=np.exp(-(gamma-1.0)/theta)*(gamma**2+2*theta*gamma+2*theta**2)
    jacobian=2.0/(sine*laser*(ratio[valid]+1.0)**2)
    y=np.full_like(x,20.0)
    y[valid]+=900.0*marginal*jacobian
    out=analyze_thomson_spectra(
        x,y[None,:],[0.18],laser_wavelength_nm=laser,
        scattering_angle_rad=angle,instrument_response=np.eye(x.size),model="relativistic_maxwell_juttner",
    )
    assert out["electron_temperature_profile_eV"][0] == pytest.approx(te,rel=0.08)
    assert "Maxwell-Juttner" in out["channels"][0]["model"]


def test_registry_resolves_primary_thomson_spectral_workflow():
    from tokamak_analysis.analysis_registry import resolve_analysis
    entrypoint=resolve_analysis(1)
    assert callable(entrypoint)
    assert entrypoint.__name__ == "analyze_thomson_spectra"


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
    assert second["integrated_scattered_power_W"] == pytest.approx(
        first["integrated_scattered_power_W"] * 1e6, rel=1e-6
    )


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


def test_collective_mode_requires_explicit_beam_geometry_before_optional_backend_import():
    x, y, _ = _spectral_case(100.0)
    with pytest.raises(ValueError, match="explicit incident_direction"):
        analyze_thomson_spectra(
            x, y[None, :], [0.1], laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2, instrument_response=np.eye(x.size),
            model="collective_maxwellian", ion_species=["D+"],
            ion_temperature_eV=[100.0], ion_velocity_m_s=[[0.0, 0.0, 0.0]],
            electron_velocity_m_s=[0.0, 0.0, 0.0],
        )


def test_descriptive_moments_report_negative_residual_clipping_caveat():
    from tokamak_analysis.thomson_distribution import analyze_thomson_spectral_distribution
    x=np.linspace(500.0,564.0,401)
    y=100.0*np.exp(-0.5*((x-532.0)/8.0)**2)+10.0
    y[0]=0.0
    y[-1]=0.0
    out=analyze_thomson_spectral_distribution(x,y,instrument_sigma_nm=0.2)
    assert "negative_area_fraction_removed_by_clipping" in out
    assert "clipping" in out["moment_clipping_bias_warning"].lower()

def test_collective_fit_reconstructs_spectrum_in_physical_parameter_space(monkeypatch):
    import sys
    import types

    class _Spectrum:
        def __init__(self, value):
            self.value = np.asarray(value, dtype=float)

    class _FakeThomson:
        @staticmethod
        def spectral_density(wavelength, laser_wavelength, density, **kwargs):
            x = np.asarray(wavelength, dtype=float)
            te = float(kwargs["T_e"])
            ne = float(density)
            sigma = 2.0 * np.sqrt(te / 100.0)
            broad_fraction = 0.10 * (ne / 1e18) ** 0.30
            spectrum = (
                np.exp(-0.5 * ((x - 532.0) / sigma) ** 2)
                + broad_fraction * np.exp(-0.5 * ((x - 532.0) / (4.0 * sigma)) ** 2)
            )
            return 2.0, _Spectrum(spectrum)

    fake_astropy = types.ModuleType("astropy")
    fake_astropy.units = types.SimpleNamespace(nm=1.0, eV=1.0, m=1.0, s=1.0)
    fake_plasmapy = types.ModuleType("plasmapy")
    fake_diagnostics = types.ModuleType("plasmapy.diagnostics")
    fake_diagnostics.thomson = _FakeThomson
    fake_plasmapy.diagnostics = fake_diagnostics
    monkeypatch.setitem(sys.modules, "astropy", fake_astropy)
    monkeypatch.setitem(sys.modules, "plasmapy", fake_plasmapy)
    monkeypatch.setitem(sys.modules, "plasmapy.diagnostics", fake_diagnostics)

    x = np.linspace(490.0, 574.0, 421)
    true_density = 4.0e18
    true_temperature = 160.0
    _, physical_shape = _FakeThomson.spectral_density(
        x, 532.0, true_density, T_e=true_temperature
    )
    y = 2400.0 * physical_shape.value + 18.0 + 0.015 * (x - 532.0)
    y += 0.015 * np.sin(np.linspace(0.0, 18.0 * np.pi, x.size))

    result = analyze_thomson_spectra(
        x, y[None, :], [0.2],
        laser_wavelength_nm=532.0,
        scattering_angle_rad=np.pi / 2,
        instrument_response=np.eye(x.size),
        model="collective_maxwellian",
        ion_species=["H+"],
        ion_fractions=[1.0],
        ion_temperature_eV=[20.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        incident_direction=[1.0, 0.0, 0.0],
        scattered_direction=[0.0, 1.0, 0.0],
    )

    channel = result["channels"][0]
    assert channel["optimizer_success"] is True
    assert channel["electron_density_m3"] == pytest.approx(true_density, rel=0.08)
    assert channel["electron_temperature_eV"] == pytest.approx(true_temperature, rel=0.05)
    assert np.allclose(channel["fitted_spectrum"], y, rtol=0.02, atol=0.03)
    expected_dof = max(1, x.size - 5)
    expected_chi_square = np.sum(channel["residual"] ** 2) / (
        max(float(np.max(np.abs(y))), np.finfo(float).tiny) ** 2 * expected_dof
    )
    assert channel["reduced_chi_square"] == pytest.approx(expected_chi_square, rel=1e-8)


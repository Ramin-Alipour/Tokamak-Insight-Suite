"""Smoke tests for the modular tokamak diagnostic analysis workflows."""
import importlib
import numpy as np
import pytest
import tokamak_analysis


@pytest.mark.parametrize("module_name", tokamak_analysis.__all__)
def test_all_analysis_modules_import(module_name):
    importlib.import_module(f"tokamak_analysis.{module_name}")


def test_thomson_profile_workflow():
    from tokamak_analysis.thomson_workflow import analyze_thomson_profile
    r=np.array([0.0,0.1,0.2,0.3])
    out=analyze_thomson_profile(r,2e19*(1-r),100*(1-.5*r))
    np.testing.assert_allclose(out["density_gradient_m-4"],-2e19,rtol=1e-12)
    np.testing.assert_allclose(out["temperature_gradient_eV_m-1"],-50,rtol=1e-12)


def test_beta_and_confinement():
    from tokamak_analysis.beta import analyze_beta
    from tokamak_analysis.confinement import energy_confinement_time
    assert analyze_beta(1e3,2.0)["beta"] > 0
    assert energy_confinement_time(1000,500)["energy_confinement_time_s"] == pytest.approx(2.0)


def test_doppler_temperature_and_flow():
    from tokamak_analysis.doppler_temperature import ion_temperature_from_doppler_fwhm
    from tokamak_analysis.doppler_flow import doppler_flow_velocity
    assert ion_temperature_from_doppler_fwhm(.01,500,40)["ion_temperature_eV"] > 0
    assert doppler_flow_velocity(500.001,500)["velocity_m_s"] > 0


def test_plasma_geometry_closed_circle():
    from tokamak_analysis.plasma_geometry import geometry_from_boundary
    th=np.linspace(0,2*np.pi,501)
    r=1.5+.2*np.cos(th); z=.2*np.sin(th)
    out=geometry_from_boundary(r,z)
    assert out["cross_section_area_m2"] == pytest.approx(np.pi*.2**2,rel=2e-3)
    assert out["toroidal_volume_m3"] > 0


def test_interferometry_line_average_density():
    from tokamak_analysis.interferometry import interferometry_density
    out=interferometry_density(1e-3,140e9,path_length_m=1.0)
    assert out["line_average_density_m-3"] > 0


def test_signal_synchronization():
    from tokamak_analysis.channel_synchronization import synchronize_channels
    t1=np.linspace(0,1,11); t2=np.linspace(.1,.9,9)
    out=synchronize_channels([t1,t2],[t1,t2])
    assert out["synchronized_signals"].shape[0] == 2
    assert np.all(np.diff(out["time_s"])>0)


def test_denoising_shape_and_finite():
    from tokamak_analysis.denoising import denoise_signal
    x=np.sin(np.linspace(0,4*np.pi,101))
    out=denoise_signal(x,method="savgol",window_length=9)
    assert out["filtered_signal"].shape==x.shape
    assert np.all(np.isfinite(out["filtered_signal"]))


def test_reaction_rate_and_neutral_gas():
    from tokamak_analysis.fusion_rate import fusion_reaction_rate
    from tokamak_analysis.neutral_gas import neutral_density
    assert fusion_reaction_rate(1e19,1e19,1e-22)["reaction_rate_density_m-3_s-1"] > 0
    assert neutral_density(1e-3,300)["neutral_density_m-3"] > 0


def test_invalid_input_rejected():
    from tokamak_analysis.fft_analysis import analyze_fft
    with pytest.raises(ValueError):
        analyze_fft([0,1,2,4],[1,2,3,4])


def test_analysis_registry_resolves_every_callable():
    from tokamak_analysis.analysis_registry import ANALYSES, resolve_analysis
    assert len(ANALYSES) == 80
    assert [x["id"] for x in ANALYSES] == list(range(1, 81))
    for entry in ANALYSES:
        assert callable(resolve_analysis(entry["id"]))



def test_equilibrium_reconstruction_zero_source_exact_solution():
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium
    R = np.array([1.0, 1.5, 2.0])
    Z = np.array([-0.5, 0.0, 0.5])
    boundary = np.zeros((3, 3))
    out = reconstruct_equilibrium(
        R, Z, boundary,
        psi_profile=[0.0, 1.0],
        pressure_gradient_Pa_per_Wb=[0.0, 0.0],
        ffprime_T2=[0.0, 0.0],
        max_iterations=10,
        tolerance=1e-12,
    )
    assert out["converged"] is True
    assert out["iterations"] == 1
    np.testing.assert_allclose(out["psi_Wb_rad"], np.zeros((3, 3)), atol=0.0)
    np.testing.assert_allclose(out["B_R_T"], np.zeros((3, 3)), atol=0.0)
    np.testing.assert_allclose(out["B_Z_T"], np.zeros((3, 3)), atol=0.0)



def test_shape_reconstruction_known_circular_flux_surface():
    from tokamak_analysis.shape_reconstruction import analyze_plasma_shape
    R = np.linspace(1.0, 2.0, 151)
    Z = np.linspace(-0.5, 0.5, 151)
    RR, ZZ = np.meshgrid(R, Z)
    psi_n = ((RR - 1.5)**2 + ZZ**2) / (0.2**2)
    level = 0.5
    out = analyze_plasma_shape(R, Z, psi_n, boundary_level=level)
    expected_radius = 0.2 * np.sqrt(level)
    assert out["geometric_major_radius_m"] == pytest.approx(1.5, abs=2e-3)
    assert out["minor_radius_horizontal_m"] == pytest.approx(expected_radius, abs=2e-3)
    assert out["cross_section_area_m2"] == pytest.approx(np.pi * expected_radius**2, rel=2e-2)
    assert out["elongation_estimate"] == pytest.approx(1.0, abs=2e-2)



def test_plasma_current_known_linear_ramp():
    from tokamak_analysis.plasma_current import analyze_plasma_current
    time_s = np.array([0.0, 1.0, 2.0])
    current_A = np.array([1.0, 3.0, 5.0])
    out = analyze_plasma_current(time_s, current_A)
    np.testing.assert_allclose(out["dIp_dt_A_s"], [2.0, 2.0, 2.0], atol=1e-12)
    assert out["peak_signed_current_A"] == pytest.approx(5.0)
    assert out["peak_time_s"] == pytest.approx(2.0)
    assert out["time_integral_A_s"] == pytest.approx(6.0)



def test_magnetic_field_known_3_4_5_components():
    from tokamak_analysis.magnetic_field import analyze_magnetic_field
    out = analyze_magnetic_field(3.0, 4.0)
    assert float(out["B_total_T"]) == pytest.approx(5.0)
    assert float(out["field_pitch_deg"]) == pytest.approx(np.degrees(np.arctan2(4.0, 3.0)))
    assert out["max_total_field_T"] == pytest.approx(5.0)
    assert out["rms_total_field_T"] == pytest.approx(5.0)



def test_beta_known_pressure_and_field():
    from tokamak_analysis.beta import analyze_beta
    out = analyze_beta(1000.0, 2.0)
    expected_beta = 2 * (4e-7 * np.pi) * 1000.0 / (2.0**2)
    assert float(out["beta"]) == pytest.approx(expected_beta, rel=1e-12)
    assert float(out["beta_percent"]) == pytest.approx(100 * expected_beta, rel=1e-12)
    assert expected_beta == pytest.approx(2 * np.pi * 1e-4, rel=1e-12)



def test_safety_factor_known_cylindrical_estimate():
    from tokamak_analysis.safety_factor import estimate_safety_factor
    out = estimate_safety_factor(1.5, 0.5, 2.0, 0.1)
    assert float(out["q_cylindrical_estimate"]) == pytest.approx(60.0, rel=1e-12)

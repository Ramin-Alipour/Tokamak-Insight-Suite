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
    assert out["geometric_major_radius_m"] == pytest.approx(1.5, abs=2e-3)
    assert out["toroidal_volume_m3"] == pytest.approx(2*np.pi**2*1.5*.2**2, rel=2e-3)


def test_interferometry_line_average_density():
    from tokamak_analysis.interferometry import interferometry_density
    out=interferometry_density(1e-3,140e9,path_length_m=1.0)
    expected_line_density = 1e-3 / ((1.602176634e-19**2 / (4*np.pi*8.8541878128e-12*9.1093837139e-31*299792458.0**2)) * (299792458.0/140e9))
    assert out["line_integrated_density_m-2"] == pytest.approx(expected_line_density, rel=1e-12)
    assert out["line_average_density_m-3"] == pytest.approx(expected_line_density, rel=1e-12)


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



def test_magnetic_axis_known_grid_minimum():
    from tokamak_analysis.magnetic_axis import locate_magnetic_axis
    R = np.array([1.0, 1.5, 2.0])
    Z = np.array([-1.0, 0.0, 1.0])
    psi = np.array([[3.0, 2.0, 3.0], [2.0, 0.0, 2.0], [3.0, 2.0, 3.0]])
    out = locate_magnetic_axis(R, Z, psi)
    assert out["R_axis_m"] == pytest.approx(1.5)
    assert out["Z_axis_m"] == pytest.approx(0.0)
    assert out["psi_axis_Wb_rad"] == pytest.approx(0.0)



def test_x_point_known_quadratic_saddle():
    from tokamak_analysis.x_point import find_x_point_candidates
    R = np.array([1.0, 2.0, 3.0])
    Z = np.array([-1.0, 0.0, 1.0])
    RR, ZZ = np.meshgrid(R, Z)
    psi = (RR - 2.0)**2 - ZZ**2
    out = find_x_point_candidates(R, Z, psi)
    assert out["count"] == 1
    assert out["candidates"][0]["R_m"] == pytest.approx(2.0)
    assert out["candidates"][0]["Z_m"] == pytest.approx(0.0)
    assert out["candidates"][0]["hessian_determinant"] == pytest.approx(-4.0, abs=1e-10)



def test_shafranov_shift_known_geometric_and_analytic_values():
    from tokamak_analysis.shafranov_shift import shafranov_shift
    out = shafranov_shift(1.55, 1.50, minor_radius_m=0.5, beta_p=1.0, li=1.0)
    assert out["measured_axis_shift_m"] == pytest.approx(0.05)
    assert out["normalized_axis_shift"] == pytest.approx(0.1)
    assert out["large_aspect_ratio_circular_prediction_m"] == pytest.approx(0.125)



def test_average_density_known_profile_and_line_integral():
    from tokamak_analysis.average_density import analyze_average_density
    out = analyze_average_density(
        density_profile_m3=[1e19, 2e19, 3e19],
        radius_m=[0.0, 1.0, 2.0],
        line_integrated_density_m2=6e19,
        chord_length_m=2.0,
    )
    assert out["sample_mean_density_m3"] == pytest.approx(2e19, rel=1e-12)
    assert out["peak_density_m3"] == pytest.approx(3e19, rel=1e-12)
    assert out["radial_average_density_m3"] == pytest.approx(2e19, rel=1e-12)
    assert out["line_average_density_m3"] == pytest.approx(3e19, rel=1e-12)



def test_electron_temperature_known_profile_statistics():
    from tokamak_analysis.electron_temperature import analyze_electron_temperature
    out = analyze_electron_temperature([10.0, 20.0, 30.0], coordinate=[0.0, 1.0, 2.0])
    assert out["mean_eV"] == pytest.approx(20.0)
    assert out["median_eV"] == pytest.approx(20.0)
    assert out["peak_eV"] == pytest.approx(30.0)
    assert out["standard_deviation_eV"] == pytest.approx(10.0)
    np.testing.assert_allclose(out["gradient_eV_per_coordinate"], [10.0, 10.0, 10.0])



def test_ion_temperature_known_profile_statistics():
    from tokamak_analysis.ion_temperature import analyze_ion_temperature
    out = analyze_ion_temperature([100.0, 200.0, 300.0], coordinate=[0.0, 1.0, 2.0])
    assert out["mean_eV"] == pytest.approx(200.0)
    assert out["median_eV"] == pytest.approx(200.0)
    assert out["peak_eV"] == pytest.approx(300.0)
    assert out["standard_deviation_eV"] == pytest.approx(100.0)
    np.testing.assert_allclose(out["gradient_eV_per_coordinate"], [100.0, 100.0, 100.0])



def test_langmuir_probe_known_linear_iv_curve():
    from tokamak_analysis.probe_analysis import analyze_probe_voltage_current
    V = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    I = V.copy()
    out = analyze_probe_voltage_current(V, I)
    np.testing.assert_allclose(out["dI_dV_S"], np.ones(5), atol=1e-12)
    assert out["floating_potential_V"] == pytest.approx(0.0)
    assert out["ion_saturation_current_A"] == pytest.approx(-2.0)
    assert out["electron_saturation_current_A"] == pytest.approx(2.0)



def test_ball_pen_probe_known_signals():
    from tokamak_analysis.ball_pen_probe import analyze_ball_pen_probe
    out = analyze_ball_pen_probe([0.0, 1.0, 2.0], [2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    assert out["mean_V"] == pytest.approx(4.0)
    assert out["std_V"] == pytest.approx(2.0)
    assert out["peak_to_peak_V"] == pytest.approx(4.0)
    np.testing.assert_allclose(out["potential_difference_V"], [1.0, 2.0, 3.0])
    assert out["mean_potential_difference_V"] == pytest.approx(2.0)



def test_gundestrup_probe_known_first_harmonic():
    from tokamak_analysis.gundestrup_probe import analyze_gundestrup
    angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    currents = 2.0 + 3.0*np.cos(angles) + 4.0*np.sin(angles)
    out = analyze_gundestrup(angles, currents)
    assert out["offset_current_A"] == pytest.approx(2.0, abs=1e-12)
    assert out["first_harmonic_amplitude_A"] == pytest.approx(5.0, abs=1e-12)
    assert out["flow_phase_rad"] == pytest.approx(np.arctan2(4.0, 3.0), abs=1e-12)
    assert out["residual_rms_A"] == pytest.approx(0.0, abs=1e-12)



def test_multipin_probe_known_linear_potential_gradient():
    from tokamak_analysis.multipin_probe import analyze_multipin_probe
    positions = np.array([0.0, 0.1, 0.2])
    potentials = np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]])
    out = analyze_multipin_probe([0.0, 1.0], potentials, positions)
    np.testing.assert_allclose(out["potential_gradient_V_m"], 10.0, atol=1e-12)
    np.testing.assert_allclose(out["electric_field_V_m"], -10.0, atol=1e-12)
    np.testing.assert_allclose(out["mean_potential_by_pin_V"], [0.5, 1.5, 2.5], atol=1e-12)



def test_plasma_floating_potential_and_electric_field():
    from tokamak_analysis.potential_analysis import analyze_potentials
    out = analyze_potentials([10.0, 12.0, 14.0], [2.0, 3.0, 4.0], coordinate_m=[0.0, 1.0, 2.0])
    np.testing.assert_allclose(out["difference_V"], [8.0, 9.0, 10.0])
    assert out["mean_difference_V"] == pytest.approx(9.0)
    np.testing.assert_allclose(out["electric_field_V_m"], [-2.0, -2.0, -2.0])



def test_doppler_spectroscopy_known_gaussian_line():
    from tokamak_analysis.doppler_spectroscopy import analyze_doppler_spectrum, C_M_S
    wavelength = np.linspace(499.8, 500.2, 401)
    center = 500.05
    sigma = 0.02
    intensity = 1.0 + 10.0*np.exp(-0.5*((wavelength-center)/sigma)**2)
    out = analyze_doppler_spectrum(wavelength, intensity, rest_wavelength_nm=500.0)
    assert out["center_nm"] == pytest.approx(center, abs=1e-7)
    assert out["width_fwhm_nm"] == pytest.approx(2.354820045*sigma, rel=1e-5)
    assert out["nonrelativistic_velocity_m_s"] == pytest.approx(C_M_S*(center/500.0-1.0), rel=1e-5)
    assert out["r_squared"] == pytest.approx(1.0, abs=1e-10)



def test_doppler_ion_temperature_known_thermal_width():
    from tokamak_analysis.doppler_temperature import ion_temperature_from_doppler_fwhm
    wavelength_nm = 500.0
    mass_amu = 40.0
    expected_Ti_eV = 100.0
    mass_kg = mass_amu * 1.66053906660e-27
    sigma_v = np.sqrt(expected_Ti_eV * 1.602176634e-19 / mass_kg)
    thermal_fwhm = 2.354820045 * wavelength_nm * sigma_v / 299792458.0
    instrumental_fwhm = 0.02
    observed_fwhm = np.sqrt(thermal_fwhm**2 + instrumental_fwhm**2)
    out = ion_temperature_from_doppler_fwhm(
        observed_fwhm, wavelength_nm, mass_amu,
        instrumental_fwhm_nm=instrumental_fwhm,
    )
    assert out["thermal_fwhm_nm"] == pytest.approx(thermal_fwhm, rel=1e-10)
    assert out["ion_temperature_eV"] == pytest.approx(expected_Ti_eV, rel=1e-10)



def test_doppler_flow_known_wavelength_shift():
    from tokamak_analysis.doppler_flow import doppler_flow_velocity, C
    out = doppler_flow_velocity(500.001, 500.0, uncertainty_nm=0.0001)
    assert float(out["wavelength_shift_nm"]) == pytest.approx(0.001)
    assert float(out["velocity_m_s"]) == pytest.approx(C * (0.001/500.0), rel=1e-12)
    assert float(out["velocity_uncertainty_m_s"]) == pytest.approx(C * (0.0001/500.0), rel=1e-12)



def test_impurity_concentration_known_emissivity_ratio():
    from tokamak_analysis.impurity_concentration import estimate_impurity_fraction
    out = estimate_impurity_fraction(
        4.0, 2.0, 2.0,
        hydrogen_line_emissivity=8.0,
        hydrogen_excitation_coefficient=2.0,
    )
    assert float(out["impurity_density_estimate_m-3"]) == pytest.approx(1.0)
    assert float(out["impurity_fraction_estimate"]) == pytest.approx(0.5)



def test_spectral_broadening_known_quadrature_decomposition():
    from tokamak_analysis.spectral_broadening import decompose_gaussian_linewidth
    out = decompose_gaussian_linewidth(
        0.13, instrumental_fwhm_nm=0.05,
        natural_fwhm_nm=0.03, zeeman_fwhm_equivalent_nm=0.04,
    )
    expected = np.sqrt(0.13**2 - 0.05**2 - 0.03**2 - 0.04**2)
    assert out["residual_gaussian_fwhm_nm"] == pytest.approx(expected, rel=1e-12)



def test_spectral_line_identification_known_nearest_matches():
    from tokamak_analysis.spectral_line_identification import identify_spectral_lines
    out = identify_spectral_lines(
        [500.02, 501.2], [(500.0, "A"), (501.0, "B")],
        tolerance_nm=0.05,
    )
    assert out["matches"][0]["label"] == "A"
    assert out["matches"][0]["matched"] is True
    assert out["matches"][0]["delta_nm"] == pytest.approx(0.02)
    assert out["matches"][1]["label"] == "B"
    assert out["matches"][1]["matched"] is False
    assert out["matches"][1]["delta_nm"] == pytest.approx(0.2)



def test_spectral_line_ratio_known_value_and_uncertainty():
    from tokamak_analysis.spectral_line_ratio import analyze_line_ratio
    out = analyze_line_ratio(12.0, 4.0, uncertainty_numerator=1.0, uncertainty_denominator=0.5)
    assert float(out["line_ratio"]) == pytest.approx(3.0)
    expected_uncertainty = 3.0*np.sqrt((1.0/12.0)**2 + (0.5/4.0)**2)
    assert float(out["line_ratio_uncertainty"]) == pytest.approx(expected_uncertainty, rel=1e-12)



def test_reflectometry_o_mode_cutoff_density_known_frequency():
    from tokamak_analysis.reflectometry import cutoff_density, EPS0, ME, E
    frequency = 10e9
    out = cutoff_density(frequency, mode="O")
    expected = EPS0*ME*(2*np.pi*frequency)**2/E**2
    assert float(out["cutoff_density_m-3"]) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(1.240e18, rel=2e-3)



def test_density_wave_known_sinusoidal_frequency_and_rms():
    from tokamak_analysis.density_waves import analyze_density_waves
    n_samples = 1024
    time = np.arange(n_samples) / 1000.0
    amplitude = 1e18
    frequency_Hz = 15.625
    density = 1e19 + amplitude*np.sin(2*np.pi*frequency_Hz*time)
    out = analyze_density_waves(time, density)
    expected_rms = amplitude*np.sqrt(n_samples/(2*(n_samples-1)))
    assert out["rms_fluctuation"] == pytest.approx(expected_rms, rel=1e-10)
    assert out["relative_rms"] == pytest.approx(expected_rms/1e19, rel=1e-10)
    assert out["frequency_Hz"][np.argmax(out["power_spectral_density"])] == pytest.approx(frequency_Hz)



def test_plasma_wave_spectrum_known_dominant_frequency():
    from tokamak_analysis.plasma_wave_spectrum import analyze_wave_signal
    time = np.arange(1024)/1000.0
    expected_frequency = 15.625
    signal = 2.0*np.sin(2*np.pi*expected_frequency*time)
    out = analyze_wave_signal(time, signal, nperseg=256)
    assert out["sampling_frequency_Hz"] == pytest.approx(1000.0)
    assert out["dominant_frequency_Hz"] == pytest.approx(expected_frequency)



def test_hard_xray_known_background_subtraction_and_rate():
    from tokamak_analysis.hard_xray import analyze_hard_xray
    out = analyze_hard_xray(
        [10.0, 20.0, 30.0], [1.0, 2.0, 1.0],
        live_time_s=2.0, background_counts=[0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(out["net_counts"], [1.0, 1.0, 1.0])
    assert out["total_net_counts"] == pytest.approx(3.0)
    assert out["peak_energy_keV"] == pytest.approx(10.0)
    assert out["count_weighted_mean_energy_keV"] == pytest.approx(20.0)
    assert out["count_rate_s-1"] == pytest.approx(1.5)



def test_soft_xray_known_baseline_subtraction_and_integral():
    from tokamak_analysis.soft_xray import analyze_soft_xray
    out = analyze_soft_xray([0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 3.0, 5.0])
    assert out["baseline"] == pytest.approx(1.0)
    np.testing.assert_allclose(out["baseline_subtracted_signal"], [0.0, 0.0, 2.0, 4.0])
    assert out["peak_by_channel"] == pytest.approx(4.0)
    assert out["integrated_signal_by_channel"] == pytest.approx(4.0)

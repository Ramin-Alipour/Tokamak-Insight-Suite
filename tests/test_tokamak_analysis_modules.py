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
    assert neutral_density(1e-3,300)["neutral_density_m-3"] == pytest.approx(1e-3/(1.380649e-23*300), rel=1e-12)


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
    assert float(out["q_cylindrical_estimate"]) == pytest.approx(2.0*0.5/(1.5*0.1), rel=1e-12)



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
    assert out["large_aspect_ratio_circular_prediction_m"] == pytest.approx(0.08333333333333333)



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
    assert float(out["velocity_m_s"]) == pytest.approx(C * (500.001/500.0-1.0), rel=1e-12)
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



def test_xray_spectrum_known_moments():
    from tokamak_analysis.xray_spectrum import analyze_xray_spectrum
    out = analyze_xray_spectrum([10.0, 20.0, 30.0], [1.0, 2.0, 1.0])
    assert out["total_counts"] == pytest.approx(4.0)
    assert out["mean_energy_keV"] == pytest.approx(20.0)
    assert out["energy_std_keV"] == pytest.approx(np.sqrt(50.0))
    assert out["peak_energy_keV"] == pytest.approx(20.0)



def test_bolometry_known_triangular_pulse_integral():
    from tokamak_analysis.bolometry import analyze_bolometry
    out = analyze_bolometry([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])
    assert out["time_integrated_signal_J_m2"] == pytest.approx(2.0)
    assert out["peak_signal_W_m2"] == pytest.approx(2.0)



def test_radiation_tomography_identity_response_exact_solution():
    from tokamak_analysis.radiation_tomography import reconstruct_emissivity
    A = np.eye(2)
    y = np.array([2.0, 3.0])
    out = reconstruct_emissivity(A, y, regularization=0.0)
    np.testing.assert_allclose(out["emissivity"], y, atol=1e-8)
    np.testing.assert_allclose(out["synthetic_measurements"], y, atol=1e-8)
    assert out["residual_norm"] == pytest.approx(0.0, abs=1e-8)



def test_visible_imaging_known_dark_frame_correction():
    from tokamak_analysis.visible_imaging import analyze_visible_image
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = analyze_visible_image(image, dark_frame=np.ones((2, 2)))
    np.testing.assert_allclose(out["corrected_image"], [[0.0, 1.0], [2.0, 3.0]])
    assert out["roi_mean_intensity"] == pytest.approx(1.5)
    assert out["roi_max_intensity"] == pytest.approx(3.0)
    assert out["roi_integrated_intensity"] == pytest.approx(6.0)



def test_line_radiation_known_spectral_integral_and_power():
    from tokamak_analysis.line_radiation import analyze_line_radiation
    out = analyze_line_radiation(
        [1.0, 2.0, 3.0], [0.0, 2.0, 0.0],
        solid_angle_sr=0.5, area_m2=2.0,
    )
    assert out["integrated_spectral_radiance"] == pytest.approx(2.0)
    assert out["peak_radiance"] == pytest.approx(2.0)
    assert out["estimated_radiant_power_W"] == pytest.approx(2.0)



def test_bolometric_tomography_known_three_zone_solution():
    from tokamak_analysis.radiation_tomography import reconstruct_emissivity
    emissivity_expected = np.array([1.0, 2.0, 3.0])
    out = reconstruct_emissivity(np.eye(3), emissivity_expected, regularization=0.0)
    np.testing.assert_allclose(out["emissivity"], emissivity_expected, atol=1e-8)
    assert out["residual_norm"] == pytest.approx(0.0, abs=1e-8)



def test_neutron_flux_known_linear_count_rate():
    from tokamak_analysis.neutron_flux import analyze_neutron_flux
    out = analyze_neutron_flux(
        [0.0, 1.0, 2.0], [0.0, 10.0, 20.0],
        timestamp_semantics="counter_read_times", counts_are_cumulative=True,
        efficiency=0.5, effective_area_m2=0.2,
    )
    np.testing.assert_allclose(out["count_rate_s-1"], [10.0, 10.0, 10.0])
    assert out["total_counts"] == pytest.approx(20.0)
    np.testing.assert_allclose(out["flux_estimate_m-2_s-1"], [100.0, 100.0, 100.0])



def test_neutron_spectrum_known_background_corrected_moments():
    from tokamak_analysis.neutron_spectrum import analyze_neutron_spectrum
    out = analyze_neutron_spectrum([1.0, 2.0, 3.0], [1.0, 2.0, 1.0], background=[0.0, 1.0, 0.0])
    np.testing.assert_allclose(out["net_counts"], [1.0, 1.0, 1.0])
    assert out["peak_energy_keV"] == pytest.approx(1.0)
    assert out["mean_energy_keV"] == pytest.approx(2.0)
    assert out["total_counts"] == pytest.approx(3.0)



def test_fusion_reaction_rate_known_densities_and_volume():
    from tokamak_analysis.fusion_rate import fusion_reaction_rate
    out = fusion_reaction_rate(2e19, 3e19, 1e-22, volume_m3=0.5)
    assert out["reaction_rate_density_m-3_s-1"] == pytest.approx(6e16, rel=1e-12)
    assert out["total_reactions_s-1"] == pytest.approx(3e16, rel=1e-12)
    identical = fusion_reaction_rate(2e19, 2e19, 1e-22, identical_species=True)
    assert identical["reaction_rate_density_m-3_s-1"] == pytest.approx(2e16, rel=1e-12)



def test_nuclear_gamma_known_peak_and_line_match():
    from tokamak_analysis.nuclear_gamma import analyze_gamma_spectrum
    out = analyze_gamma_spectrum(
        [10.0, 20.0, 30.0, 40.0, 50.0],
        [0.0, 2.0, 5.0, 2.0, 0.0],
        expected_lines_keV=[29.0], tolerance_keV=2.0,
    )
    assert len(out["peaks"]) == 1
    assert out["peaks"][0]["energy_keV"] == pytest.approx(30.0)
    assert out["peaks"][0]["counts"] == pytest.approx(5.0)
    assert out["expected_line_matches"][0]["within_tolerance"] is True



def test_runaway_electron_known_threshold_event():
    from tokamak_analysis.runaway_electrons import analyze_runaway_events
    out = analyze_runaway_events([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 0.0, 5.0, 5.0, 0.0], threshold=4.0)
    assert out["event_count"] == 1
    assert out["events"][0]["start_s"] == pytest.approx(2.0)
    assert out["events"][0]["end_s"] == pytest.approx(3.0)
    assert out["events"][0]["duration_s"] == pytest.approx(1.0)
    assert out["events"][0]["peak_signal"] == pytest.approx(5.0)
    assert out["above_threshold_fraction"] == pytest.approx(0.4)



def test_energetic_particle_distribution_known_bins():
    from tokamak_analysis.energetic_particles import analyze_energy_distribution
    out = analyze_energy_distribution([10.0, 20.0, 30.0], [1.0, 2.0, 1.0])
    np.testing.assert_allclose(out["distribution"], [0.25, 0.5, 0.25])
    assert out["total_counts"] == pytest.approx(4.0)
    assert out["mean_energy_keV"] == pytest.approx(20.0)
    assert out["peak_energy_keV"] == pytest.approx(20.0)
    assert out["normalized"] is True



def test_particle_loss_rate_known_linear_counts():
    from tokamak_analysis.energetic_particles import particle_loss_rate
    out = particle_loss_rate([0.0, 1.0, 2.0], [0.0, 4.0, 8.0], detection_efficiency=0.5, counts_are_cumulative=True, timestamp_semantics="counter_read_times")
    np.testing.assert_allclose(out["loss_rate_s-1"], [8.0, 8.0, 8.0])
    assert out["total_detected_counts"] == pytest.approx(8.0)



def test_energetic_particle_flux_known_counts_and_area():
    from tokamak_analysis.energetic_particles import energetic_particle_flux
    out = energetic_particle_flux([0.0, 1.0, 2.0], [0.0, 4.0, 8.0], 2.0, efficiency=0.5, counts_are_cumulative=True, timestamp_semantics="counter_read_times")
    np.testing.assert_allclose(out["particle_flux_m-2_s-1"], [4.0, 4.0, 4.0])



def test_runaway_energy_known_exponential_spectrum():
    from tokamak_analysis.runaway_energy import estimate_runaway_energy
    energy = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    counts = np.exp(-energy/100.0)
    out = estimate_runaway_energy(energy, counts)
    assert out["exponential_slope_per_keV"] == pytest.approx(-0.01, abs=1e-12)
    assert out["characteristic_energy_keV"] == pytest.approx(100.0, rel=1e-10)



def test_vacuum_pressure_known_ramp_and_integral():
    from tokamak_analysis.vacuum_pressure import analyze_vacuum_pressure
    out = analyze_vacuum_pressure([0.0, 1.0, 2.0], [1.0, 2.0, 3.0], base_pressure_Pa=1.0)
    assert out["peak_pressure_Pa"] == pytest.approx(3.0)
    assert out["mean_pressure_Pa"] == pytest.approx(2.0)
    np.testing.assert_allclose(out["pressure_rise_rate_Pa_s"], [1.0, 1.0, 1.0])
    assert out["pressure_time_integral_Pa_s"] == pytest.approx(4.0)
    np.testing.assert_allclose(out["pressure_above_base_Pa"], [0.0, 1.0, 2.0])



def test_gas_puffing_known_valve_and_calibrated_flow():
    from tokamak_analysis.gas_puffing import analyze_gas_puffing
    out = analyze_gas_puffing(
        [0.0, 1.0, 2.0], [0.0, 1.0, 1.0],
        calibrated_flow_mol_s=[0.0, 2.0, 2.0],
    )
    assert out["integrated_valve_command_s"] == pytest.approx(1.5)
    assert out["open_fraction"] == pytest.approx(2/3)
    assert out["injected_amount_mol"] == pytest.approx(3.0)



def test_fueling_rate_known_injected_particles_and_gas_inventory():
    from tokamak_analysis.fueling_rate import analyze_fueling_rate, K_B
    out = analyze_fueling_rate(
        [0.0, 1.0, 2.0], [0.0, 2.0, 0.0],
        gas_temperature_K=300.0, pressure_Pa=1e-3, volume_m3=0.01,
    )
    assert out["injected_particles"] == pytest.approx(2.0)
    assert out["peak_rate_s-1"] == pytest.approx(2.0)
    assert out["ideal_gas_inventory_particles"] == pytest.approx(1e-3*0.01/(K_B*300.0), rel=1e-12)



def test_wall_recycling_known_flux_ratio():
    from tokamak_analysis.wall_recycling import recycling_coefficient
    out = recycling_coefficient(10.0, 8.0)
    assert float(out["recycling_coefficient"]) == pytest.approx(0.8, rel=1e-12)



def test_mirnov_signal_known_frequency_and_coil_conversion():
    from tokamak_analysis.mirnov_analysis import analyze_mirnov_signal
    time = np.arange(1024)/1000.0
    frequency = 15.625
    voltage = np.sin(2*np.pi*frequency*time)
    out = analyze_mirnov_signal(time, voltage, coil_area_m2=0.01, turns=10)
    assert out["frequency_Hz"][np.argmax(out["power_spectral_density"])] == pytest.approx(frequency)
    assert out["rms_voltage_V"] == pytest.approx(1 / np.sqrt(2), rel=1e-10)
    np.testing.assert_allclose(out["approx_Bdot_T_s"], -voltage/0.1, atol=1e-12)



def test_mhd_mode_number_known_phase_winding():
    from tokamak_analysis.mhd_modes import estimate_mode_number
    angles = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])
    amplitudes = np.exp(1j*2.0*angles)
    out = estimate_mode_number(angles, amplitudes)
    assert out["mode_number_estimate"] == pytest.approx(2.0, abs=1e-12)
    assert out["nearest_integer_mode_number"] == 2
    assert out["fit_residual_rms_rad"] == pytest.approx(0.0, abs=1e-12)



def test_magnetic_mode_frequency_known_sinusoid():
    from tokamak_analysis.magnetic_mode_frequency import magnetic_mode_frequency
    time = np.arange(1024)/1000.0
    expected_frequency = 15.625
    signal = np.sin(2*np.pi*expected_frequency*time)
    out = magnetic_mode_frequency(time, signal, frequency_band_Hz=(10.0, 20.0))
    assert out["dominant_frequency_Hz"] == pytest.approx(expected_frequency)
    assert out["peak_power"] > 0



def test_fft_known_bin_frequency_and_amplitude():
    from tokamak_analysis.fft_analysis import analyze_fft
    time = np.arange(1000)/1000.0
    frequency = 10.0
    signal = 3.0*np.sin(2*np.pi*frequency*time)
    out = analyze_fft(time, signal, window="rectangular")
    idx = int(np.argmin(np.abs(out["frequency_Hz"]-frequency)))
    assert out["frequency_Hz"][idx] == pytest.approx(frequency)
    assert out["amplitude"][idx] == pytest.approx(3.0, rel=1e-10)



def test_svd_known_orthogonal_signal_matrix():
    from tokamak_analysis.svd_analysis import analyze_svd
    signals = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    out = analyze_svd(signals, rank=2)
    np.testing.assert_allclose(out["singular_values"], [np.sqrt(2.0), np.sqrt(2.0)], atol=1e-12)
    np.testing.assert_allclose(out["explained_variance_fraction"], [0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(out["reconstruction"], signals, atol=1e-12)
    assert out["residual_rms"] == pytest.approx(0.0, abs=1e-12)



def test_wavelet_known_frequency_localized_power():
    from tokamak_analysis.wavelet_analysis import analyze_wavelet
    time = np.arange(1024)/1000.0
    frequency = 15.625
    signal = np.sin(2*np.pi*frequency*time)
    out = analyze_wavelet(time, signal, frequencies_Hz=[frequency])
    power = out["power"][0]
    assert power.shape == time.shape
    assert np.all(np.isfinite(power))
    assert np.max(power) > 0
    assert power[len(power)//2] > power[0]



def test_turbulence_known_rms_and_cross_covariance():
    from tokamak_analysis.turbulence import analyze_turbulence
    n = 1000
    time = np.arange(n)/1000.0
    fluctuation = np.sin(2*np.pi*10.0*time)
    a = 10.0 + fluctuation
    b = 2.0*fluctuation
    out = analyze_turbulence(time, a, b)
    assert out["mean"] == pytest.approx(10.0, abs=1e-12)
    assert out["rms_fluctuation"] == pytest.approx(np.sqrt(n/(2*(n-1))), rel=1e-10)
    assert out["relative_rms"] == pytest.approx(np.sqrt(n/(2*(n-1)))/10.0, rel=1e-10)
    assert out["cross_covariance"] == pytest.approx(1.0, abs=1e-12)



def test_radial_transport_known_covariance():
    from tokamak_analysis.transport import fluctuation_transport
    out = fluctuation_transport([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], direction="radial")
    assert out["direction"] == "radial"
    assert out["particle_flux_covariance_m-2_s-1"] == pytest.approx(2.5)
    assert out["density_fluctuation_rms"] == pytest.approx(np.sqrt(1.25))
    assert out["velocity_fluctuation_rms_m_s"] == pytest.approx(np.sqrt(5.0))



def test_poloidal_transport_known_anticorrelation():
    from tokamak_analysis.transport import fluctuation_transport
    out = fluctuation_transport([1.0, 2.0, 3.0, 4.0], [8.0, 6.0, 4.0, 2.0], direction="poloidal")
    assert out["direction"] == "poloidal"
    assert out["particle_flux_covariance_m-2_s-1"] == pytest.approx(-2.5)



def test_reynolds_stress_known_velocity_covariance():
    from tokamak_analysis.transport import reynolds_stress
    out = reynolds_stress([1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0], density_kg_m3=2.0)
    assert out["velocity_covariance_m2_s-2"] == pytest.approx(2.5)
    assert out["reynolds_stress_Pa"] == pytest.approx(5.0)



def test_energy_confinement_time_known_value_and_uncertainty():
    from tokamak_analysis.confinement import energy_confinement_time
    out = energy_confinement_time(1000.0, 500.0, uncertainty_energy_J=100.0, uncertainty_power_W=50.0)
    assert out["energy_confinement_time_s"] == pytest.approx(2.0)
    assert out["uncertainty_s"] == pytest.approx(2*np.sqrt(0.1**2+0.1**2), rel=1e-12)



def test_heat_flux_known_profile_integral_and_peak():
    from tokamak_analysis.heat_flux import analyze_heat_flux
    out = analyze_heat_flux([0.0, 1.0, 2.0], [0.0, 2.0, 0.0])
    assert out["peak_heat_flux_W_m2"] == pytest.approx(2.0)
    assert out["peak_position_m"] == pytest.approx(1.0)
    assert out["integrated_heat_flux_W_m"] == pytest.approx(2.0)



def test_electron_heating_known_power_integral():
    from tokamak_analysis.heating_analysis import analyze_heating
    out = analyze_heating([0.0, 1.0, 2.0], [0.0, 2.0, 0.0], stored_energy_J=[10.0, 12.0, 15.0], species="electron")
    assert out["species"] == "electron"
    assert out["peak_power_W"] == pytest.approx(2.0)
    assert out["injected_energy_J"] == pytest.approx(2.0)
    assert out["stored_energy_change_J"] == pytest.approx(5.0)



def test_ion_heating_known_power_integral():
    from tokamak_analysis.heating_analysis import analyze_heating
    out = analyze_heating([0.0, 1.0, 2.0], [0.0, 4.0, 0.0], species="ion")
    assert out["species"] == "ion"
    assert out["peak_power_W"] == pytest.approx(4.0)
    assert out["injected_energy_J"] == pytest.approx(4.0)



def test_ecrh_fundamental_resonance_field_known_frequency():
    from tokamak_analysis.ecrh_icrh import ecrh_resonance_field, ME, E
    frequency = 140e9
    out = ecrh_resonance_field(frequency, harmonic=1)
    expected = 2*np.pi*ME*frequency/E
    assert float(out["resonance_field_T"]) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(5.001, rel=2e-3)



def test_icrh_proton_fundamental_resonance_field():
    from tokamak_analysis.ecrh_icrh import icrh_resonance_field, U, E
    frequency = 40e6
    out = icrh_resonance_field(frequency, ion_mass_amu=1.0, charge_state=1, harmonic=1)
    expected = 2*np.pi*U*frequency/E
    assert float(out["resonance_field_T"]) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(2.605, rel=1e-3)



def test_rf_plasma_interaction_known_power_balance():
    from tokamak_analysis.rf_interaction import analyze_rf_interaction
    out = analyze_rf_interaction(1000.0, 100.0, 200.0, uncertainty_W=[10.0, 5.0, 5.0])
    assert out["unaccounted_power_W"] == pytest.approx(700.0)
    assert out["coupling_fraction"] == pytest.approx(0.7)
    assert out["unaccounted_power_uncertainty_W"] == pytest.approx(np.sqrt(150.0))



def test_divertor_heat_flux_known_profile_and_power_load():
    from tokamak_analysis.divertor_analysis import analyze_divertor_flux
    out = analyze_divertor_flux([0.0, 1.0, 2.0], [0.0, 2.0, 0.0], surface_area_m2=3.0)
    assert out["peak_heat_flux_W_m2"] == pytest.approx(2.0)
    assert out["integrated_heat_flux_W_m"] == pytest.approx(2.0)
    assert out["peak_power_load_W"] == pytest.approx(6.0)



def test_plasma_facing_component_known_heat_load():
    from tokamak_analysis.divertor_analysis import analyze_divertor_flux
    out = analyze_divertor_flux([0.0, 1.0], [1.0, 3.0], surface_area_m2=0.5)
    assert out["peak_heat_flux_W_m2"] == pytest.approx(3.0)
    assert out["peak_power_load_W"] == pytest.approx(1.5)
    assert out["integrated_heat_flux_W_m"] == pytest.approx(2.0)



def test_wall_sputtering_known_yield():
    from tokamak_analysis.divertor_analysis import sputtering_yield
    out = sputtering_yield(1e20, 2e19)
    assert out["sputtering_yield_atoms_per_ion"] == pytest.approx(0.2, rel=1e-12)



def test_divertor_particle_flux_known_profile_integral():
    from tokamak_analysis.divertor_analysis import analyze_divertor_flux
    out = analyze_divertor_flux([0.0, 1.0, 2.0], [0.0, 0.0, 0.0], particle_flux_m2_s=[0.0, 4.0, 0.0])
    np.testing.assert_allclose(out["particle_flux_m-2_s-1"], [0.0, 4.0, 0.0])
    assert out["integrated_particle_flux_m-1_s-1"] == pytest.approx(4.0)



def test_edge_field_known_linear_potential_profile():
    from tokamak_analysis.edge_field import analyze_edge_field
    out = analyze_edge_field([0.0, 1.0, 2.0], [10.0, 8.0, 6.0], floating_potential_V=[5.0, 4.0, 3.0])
    np.testing.assert_allclose(out["electric_field_V_m"], [2.0, 2.0, 2.0])
    np.testing.assert_allclose(out["plasma_minus_floating_V"], [5.0, 4.0, 3.0])



def test_lh_transition_known_persistent_step():
    from tokamak_analysis.lh_transition import detect_lh_transition
    time = np.arange(20, dtype=float)
    signal = np.zeros(20)
    signal[10:] = 1.0
    out = detect_lh_transition(time, signal, threshold_sigma=3.0, persistence=3)
    assert out["transition_candidate_time_s"] == pytest.approx(10.0)
    assert out["baseline_median"] == pytest.approx(0.0)
    assert out["baseline_robust_sigma"] == pytest.approx(0.0)



def test_signal_calibration_known_linear_transfer_function():
    from tokamak_analysis.signal_calibration import calibrate_signal, calibration_residuals
    out = calibrate_signal([0.0, 1.0, 2.0], [2.0, 1.0], uncertainty_raw=[0.1, 0.1, 0.1])
    np.testing.assert_allclose(out["calibrated"], [1.0, 3.0, 5.0])
    np.testing.assert_allclose(out["propagated_uncertainty"], [0.2, 0.2, 0.2])
    residual = calibration_residuals([1.0, 2.0, 3.0], [1.0, 2.5, 2.5])
    assert residual["bias"] == pytest.approx(0.0)
    assert residual["rmse"] == pytest.approx(np.sqrt(0.5/3))
    assert residual["max_absolute_error"] == pytest.approx(0.5)



def test_signal_denoising_median_removes_impulse():
    from tokamak_analysis.denoising import denoise_signal
    signal = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    out = denoise_signal(signal, method="median", window_length=3)
    np.testing.assert_allclose(out["filtered_signal"], np.zeros(5))
    np.testing.assert_allclose(out["residual"], signal)
    assert out["residual_rms"] == pytest.approx(np.sqrt(20.0))



def test_measurement_uncertainty_known_summary_and_product():
    from tokamak_analysis.measurement_uncertainty import uncertainty_summary, propagate_product_uncertainty
    summary = uncertainty_summary([10.0, 12.0, 14.0], [1.0, 2.0, 2.0])
    assert summary["mean"] == pytest.approx(12.0)
    assert summary["mean_standard_uncertainty"] == pytest.approx(1.0)
    assert summary["expanded_uncertainty_k2"] == pytest.approx(2.0)
    product = propagate_product_uncertainty([2.0, 3.0], [0.1, 0.2])
    assert product["value"] == pytest.approx(6.0)
    assert product["relative_uncertainty"] == pytest.approx(np.sqrt((0.1/2)**2+(0.2/3)**2))



def test_channel_synchronization_known_linear_interpolation():
    from tokamak_analysis.channel_synchronization import synchronize_channels
    out = synchronize_channels(
        [[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]],
        [[0.0, 10.0, 20.0], [5.0, 15.0, 25.0]],
    )
    np.testing.assert_allclose(out["time_s"], [1.0, 2.0])
    np.testing.assert_allclose(out["synchronized_signals"], [[10.0, 20.0], [10.0, 20.0]])
    assert out["channel_count"] == 2


def test_thomson_scattering_physical_density_relation_known_answer():
    from tokamak_analysis.thomson_scattering import (
        thomson_differential_cross_section_m2_sr,
        density_from_thomson_scattered_power,
    )
    ne = 2.0e19
    incident = 10.0
    length = 0.1
    theta = np.pi / 2
    solid_angle = 0.01
    efficiency = 0.5
    dsigma = thomson_differential_cross_section_m2_sr(theta)
    scattered = incident * ne * length * dsigma * solid_angle * efficiency
    out = density_from_thomson_scattered_power(
        scattered, incident, length, theta, solid_angle,
        optical_efficiency=efficiency,
    )
    assert out["electron_density_m3"] == pytest.approx(ne, rel=1e-12)
    assert out["model_scattered_power_W"] == pytest.approx(scattered, rel=1e-12)
    assert dsigma == pytest.approx(0.5 * (2.8179403262e-15)**2, rel=1e-12)


def test_grad_shafranov_physical_relation_known_quadratic_solution():
    from tokamak_analysis.equilibrium_reconstruction import grad_shafranov_physical_relation
    R = np.array([1.0, 2.0, 3.0])
    Z = np.array([-1.0, 0.0, 1.0])
    RR, ZZ = np.meshgrid(R, Z)
    psi = RR**2 + ZZ**2
    zeros = np.zeros_like(psi)
    out = grad_shafranov_physical_relation(R, Z, psi, zeros, zeros)
    np.testing.assert_allclose(out["grad_shafranov_operator_Wb_rad_m-2"], 2.0, atol=1e-10)
    np.testing.assert_allclose(out["physical_source_Wb_rad_m-2"], 0.0, atol=0.0)
    np.testing.assert_allclose(out["equation_residual_Wb_rad_m-2"], 2.0, atol=1e-10)


def test_miller_plasma_boundary_physical_relation_circle():
    from tokamak_analysis.shape_reconstruction import miller_boundary
    theta=np.linspace(0.0,2*np.pi,2001)
    out=miller_boundary(1.5,0.2,1.0,0.0,theta)
    np.testing.assert_allclose(out["R_m"],1.5+0.2*np.cos(theta),atol=1e-14)
    np.testing.assert_allclose(out["Z_m"],0.2*np.sin(theta),atol=1e-14)
    assert out["cross_section_area_m2"] == pytest.approx(np.pi*0.2**2,rel=2e-5)


def test_rogowski_voltage_current_physical_relation():
    from tokamak_analysis.plasma_current import current_from_rogowski_voltage
    t=np.linspace(0.0,1.0,101)
    M=2e-6; slope=3e5
    out=current_from_rogowski_voltage(t,np.full_like(t,M*slope),M,initial_current_A=10.0)
    np.testing.assert_allclose(out["current_A"],10.0+slope*t,rtol=1e-12,atol=1e-10)
    np.testing.assert_allclose(out["dIp_dt_A_s"],slope,rtol=1e-12)


def test_toroidal_vacuum_field_scaling_relation():
    from tokamak_analysis.magnetic_field import toroidal_field_vacuum_scaling
    r=np.array([1.0,2.0,4.0])
    out=toroidal_field_vacuum_scaling(3.0,2.0,r)
    np.testing.assert_allclose(out["B_toroidal_T"],[6.0,3.0,1.5],rtol=1e-14)
    np.testing.assert_allclose(out["Bt_times_R_T_m"],6.0,rtol=1e-14)


def test_thermal_beta_density_temperature_relation():
    from tokamak_analysis.beta import thermal_beta_from_density_temperature, MU0
    out=thermal_beta_from_density_temperature(1e19,100.0,1e19,100.0,2.0)
    expected=2*MU0*(2*1e19*100.0*1.602176634e-19)/(2.0**2)
    assert out["thermal_beta"] == pytest.approx(expected,rel=1e-14)


def test_safety_factor_field_line_pitch_integral():
    from tokamak_analysis.safety_factor import safety_factor_from_field_line_pitch
    theta=np.linspace(0.0,2*np.pi,1001)
    q=2.5
    out=safety_factor_from_field_line_pitch(theta,np.full_like(theta,q))
    assert out["safety_factor_q"] == pytest.approx(q,rel=1e-12)


def test_quadratic_magnetic_axis_refinement_known_offgrid_minimum():
    from tokamak_analysis.magnetic_axis import quadratic_magnetic_axis_refinement
    R=np.linspace(1.0,2.0,11); Z=np.linspace(-0.5,0.5,11)
    RR,ZZ=np.meshgrid(R,Z); psi=(RR-1.43)**2+2*(ZZ-0.13)**2+4.0
    out=quadratic_magnetic_axis_refinement(R,Z,psi)
    assert out["R_axis_m"] == pytest.approx(1.43,abs=1e-10)
    assert out["Z_axis_m"] == pytest.approx(0.13,abs=1e-10)
    assert out["psi_axis_Wb_rad"] == pytest.approx(4.0,abs=1e-10)


def test_flux_hessian_x_point_classification():
    from tokamak_analysis.x_point import classify_flux_critical_point
    out=classify_flux_critical_point(2.0,-2.0,0.0)
    assert out["hessian_determinant"] == pytest.approx(-4.0)
    assert out["classification"] == "saddle (X-point candidate)"


def test_large_aspect_ratio_shafranov_shift_relation():
    from tokamak_analysis.shafranov_shift import large_aspect_ratio_shafranov_shift
    out=large_aspect_ratio_shafranov_shift(0.5,1.5,1.0,1.0)
    assert out["predicted_shafranov_shift_m"] == pytest.approx(0.08333333333333333)
    assert out["normalized_shift"] == pytest.approx(0.16666666666666666)


def test_dedicated_plasma_area_and_volume_relations():
    from tokamak_analysis.plasma_geometry import cross_section_area_from_boundary, axisymmetric_volume_from_boundary
    theta=np.linspace(0,2*np.pi,2001)
    R=1.5+0.2*np.cos(theta); Z=0.2*np.sin(theta)
    area=cross_section_area_from_boundary(R,Z)
    volume=axisymmetric_volume_from_boundary(R,Z)
    assert area["cross_section_area_m2"] == pytest.approx(np.pi*0.2**2,rel=1e-5)
    assert volume["toroidal_volume_m3"] == pytest.approx(2*np.pi**2*1.5*0.2**2,rel=1e-5)


def test_cylindrical_volume_average_density_relation():
    from tokamak_analysis.average_density import cylindrical_volume_average_density
    r=np.linspace(0.0,1.0,1001)
    constant=cylindrical_volume_average_density(r,np.full_like(r,3e19))
    assert constant["volume_average_density_m3"] == pytest.approx(3e19,rel=1e-12)
    linear=cylindrical_volume_average_density(r,2e19*(1-r))
    assert linear["volume_average_density_m3"] == pytest.approx(2e19/3,rel=2e-6)


def test_electron_temperature_pressure_density_relation():
    from tokamak_analysis.electron_temperature import electron_temperature_from_pressure_density
    ne=2e19; te=100.0; e=1.602176634e-19
    out=electron_temperature_from_pressure_density(ne*te*e,ne)
    assert out["electron_temperature_eV"] == pytest.approx(te,rel=1e-14)


def test_ion_temperature_pressure_density_relation():
    from tokamak_analysis.ion_temperature import ion_temperature_from_pressure_density
    ni=3e19; ti=250.0; e=1.602176634e-19
    out=ion_temperature_from_pressure_density(ni*ti*e,ni)
    assert out["ion_temperature_eV"] == pytest.approx(ti,rel=1e-14)


def test_langmuir_electron_retarding_current_relation():
    from tokamak_analysis.probe_analysis import electron_retarding_current
    v=np.array([0.0,5.0,10.0]); out=electron_retarding_current(v,10.0,2.0,5.0)
    np.testing.assert_allclose(out["electron_current_A"],2*np.exp((v-10.0)/5.0),rtol=1e-14)


def test_ball_pen_calibrated_plasma_potential_relation():
    from tokamak_analysis.ball_pen_probe import plasma_potential_from_ball_pen
    out=plasma_potential_from_ball_pen(20.0,10.0,0.6)
    assert out["plasma_potential_V"] == pytest.approx(26.0)


def test_gundestrup_calibrated_mach_relation():
    from tokamak_analysis.gundestrup_probe import calibrated_mach_from_current_harmonics
    out=calibrated_mach_from_current_harmonics(2.0,0.4,0.5)
    assert out["first_harmonic_fraction"] == pytest.approx(0.2)
    assert out["projected_mach_number"] == pytest.approx(0.4)


def test_exb_drift_physical_relation():
    from tokamak_analysis.multipin_probe import exb_drift_velocity
    out=exb_drift_velocity(0.0,2.0,0.0,0.0,0.0,4.0)
    assert out["v_exb_x_m_s"] == pytest.approx(0.5)
    assert out["v_exb_y_m_s"] == pytest.approx(0.0)
    assert out["v_exb_z_m_s"] == pytest.approx(0.0)


def test_floating_potential_current_balance_relation():
    from tokamak_analysis.potential_analysis import floating_potential_from_saturation_currents
    out=floating_potential_from_saturation_currents(20.0,5.0,np.e**2,1.0)
    assert out["floating_potential_V"] == pytest.approx(10.0)


def test_relativistic_doppler_velocity_relation():
    from tokamak_analysis.doppler_spectroscopy import relativistic_doppler_velocity, C_M_S
    rest=500.0; obs=rest*np.sqrt((1+0.01)/(1-0.01))
    out=relativistic_doppler_velocity(obs,rest)
    assert out["beta_v_over_c"] == pytest.approx(0.01,rel=1e-13)
    assert out["line_of_sight_velocity_m_s"] == pytest.approx(0.01*C_M_S,rel=1e-13)


def test_forward_inverse_doppler_temperature_relation():
    from tokamak_analysis.doppler_temperature import thermal_doppler_fwhm_nm, ion_temperature_from_doppler_fwhm
    ti=80.0; lam=500.0; mass=40.0
    width=thermal_doppler_fwhm_nm(ti,lam,mass)["thermal_fwhm_nm"]
    out=ion_temperature_from_doppler_fwhm(width,lam,mass)
    assert out["ion_temperature_eV"] == pytest.approx(ti,rel=1e-8)


def test_forward_doppler_wavelength_velocity_relation():
    from tokamak_analysis.doppler_flow import wavelength_from_line_of_sight_velocity, doppler_flow_velocity
    rest=500.0; velocity=1e5
    wavelength=wavelength_from_line_of_sight_velocity(rest,velocity,relativistic=True)["observed_wavelength_nm"]
    recovered=doppler_flow_velocity(wavelength,rest,relativistic=True)["velocity_m_s"]
    assert recovered == pytest.approx(velocity,rel=1e-12)


def test_impurity_fraction_line_ratio_relation():
    from tokamak_analysis.impurity_concentration import impurity_fraction_from_line_ratio
    out=impurity_fraction_from_line_ratio(4.0,10.0,2.0,5.0)
    assert out["impurity_to_hydrogen_density_ratio"] == pytest.approx(1.0)


def test_voigt_fwhm_approximation_limits():
    from tokamak_analysis.spectral_broadening import voigt_fwhm_approximation
    assert voigt_fwhm_approximation(2.0,0.0)["voigt_fwhm_approx_nm"] == pytest.approx(2.0)
    assert voigt_fwhm_approximation(0.0,2.0)["voigt_fwhm_approx_nm"] == pytest.approx(2.0,rel=2e-4)


def test_hydrogen_rydberg_wavelength_relation():
    from tokamak_analysis.spectral_line_identification import hydrogen_rydberg_wavelength_nm
    out=hydrogen_rydberg_wavelength_nm(3,2)
    assert out["wavelength_nm"] == pytest.approx(656.47,rel=2e-4)


def test_boltzmann_line_ratio_temperature_relation():
    from tokamak_analysis.spectral_line_ratio import boltzmann_temperature_from_line_ratio
    te=10.0; delta=5.0; pref=4.0; ratio=pref*np.exp(-delta/te)
    out=boltzmann_temperature_from_line_ratio(ratio,pref,delta)
    assert out["excitation_temperature_eV"] == pytest.approx(te,rel=1e-14)


def test_interferometer_forward_inverse_phase_relation():
    from tokamak_analysis.interferometry import phase_shift_from_line_density, interferometry_density
    line_density=2e19; frequency=140e9
    phase=phase_shift_from_line_density(line_density,frequency)["phase_shift_rad"]
    recovered=interferometry_density(phase,frequency)["line_integrated_density_m-2"]
    assert recovered == pytest.approx(line_density,rel=1e-12)


def test_o_mode_reflectometry_dispersion_relation():
    from tokamak_analysis.reflectometry import density_from_o_mode_refractive_index, cutoff_density
    f=60e9; n=0.6
    out=density_from_o_mode_refractive_index(f,n)
    cutoff=cutoff_density(f,mode="O")["cutoff_density_m-3"]
    assert out["electron_density_m3"] == pytest.approx(cutoff*(1-n**2),rel=1e-14)


def test_density_wave_phase_velocity_relation():
    from tokamak_analysis.density_waves import wave_phase_velocity
    out=wave_phase_velocity(1000.0,0.25)
    assert out["phase_velocity_m_s"] == pytest.approx(250.0)


def test_langmuir_dispersion_zero_wavenumber():
    from tokamak_analysis.plasma_wave_spectrum import langmuir_wave_frequency
    from tokamak_analysis.interferometry import E, ME, EPS0
    ne=1e19
    out=langmuir_wave_frequency(ne,20.0,0.0)
    expected=np.sqrt(ne*E**2/(EPS0*ME))/(2*np.pi)
    assert out["frequency_Hz"] == pytest.approx(expected,rel=1e-14)


def test_compton_scattering_energy_relation():
    from tokamak_analysis.hard_xray import compton_scattered_photon_energy_keV
    out=compton_scattered_photon_energy_keV(511.0,np.pi)
    assert out["scattered_photon_energy_keV"] == pytest.approx(511.0/3.0,rel=1e-5)


def test_soft_xray_maxwellian_bremsstrahlung_shape():
    from tokamak_analysis.soft_xray import maxwellian_bremsstrahlung_spectral_shape
    out=maxwellian_bremsstrahlung_spectral_shape(np.array([0.0,10.0]),10.0)
    shape=out["relative_bremsstrahlung_spectral_shape"]
    assert shape[1]/shape[0] == pytest.approx(np.exp(-1.0),rel=1e-14)


def test_xray_continuum_temperature_from_exponential_spectrum():
    from tokamak_analysis.xray_spectrum import electron_temperature_from_xray_continuum
    energy=np.linspace(1.0,10.0,20); te=3.0; counts=100*np.exp(-energy/te)
    out=electron_temperature_from_xray_continuum(energy,counts)
    assert out["electron_temperature_keV"] == pytest.approx(te,rel=1e-12)


def test_axisymmetric_emissivity_radiated_power_relation():
    from tokamak_analysis.bolometry import radiated_power_from_axisymmetric_emissivity
    a=0.2; R0=1.5; eps=100.0; r=np.linspace(0.0,a,2001)
    out=radiated_power_from_axisymmetric_emissivity(r,np.full_like(r,eps),R0)
    assert out["total_radiated_power_W"] == pytest.approx(eps*2*np.pi**2*R0*a**2,rel=1e-6)


def test_abel_projection_constant_emissivity():
    from tokamak_analysis.radiation_tomography import abel_project_axisymmetric_emissivity
    r=np.linspace(0.0,1.0,10001); eps=3.0; p=np.array([0.2,0.5])
    out=abel_project_axisymmetric_emissivity(r,np.full_like(r,eps),p)
    expected=2*eps*np.sqrt(1-p**2)
    np.testing.assert_allclose(out["line_integrated_emissivity_W_m2"],expected,rtol=0.03)


def test_visible_imaging_pixel_to_object_relation():
    from tokamak_analysis.visible_imaging import pixel_to_object_coordinate
    out=pixel_to_object_coordinate(np.array([100,110]),100,5e-6,2.0,0.05)
    np.testing.assert_allclose(out["object_coordinate_m"],[0.0,0.002],rtol=1e-14,atol=0.0)


def test_photon_energy_wavelength_relation():
    from tokamak_analysis.line_radiation import photon_energy_from_wavelength_eV
    out=photon_energy_from_wavelength_eV(500.0)
    assert out["photon_energy_eV"] == pytest.approx(1239.841984/500.0,rel=1e-8)


def test_tomography_forward_projection_relation():
    from tokamak_analysis.radiation_tomography import forward_project_emissivity
    A=np.array([[1.0,2.0],[0.5,1.0]])
    out=forward_project_emissivity(A,[3.0,4.0])
    np.testing.assert_allclose(out["predicted_line_integrals"],[11.0,5.5])


def test_integrated_neutron_flux_poisson_relation():
    from tokamak_analysis.neutron_flux import neutron_flux_from_integrated_counts
    out=neutron_flux_from_integrated_counts(100.0,10.0,0.5,2.0)
    assert out["neutron_flux_m2_s"] == pytest.approx(10.0)
    assert out["poisson_uncertainty_m2_s"] == pytest.approx(1.0)


def test_neutron_time_of_flight_energy_relation():
    from tokamak_analysis.neutron_spectrum import neutron_energy_from_time_of_flight_keV
    mass=1.67492749804e-27; energy=2.45e6*1.602176634e-19
    speed=np.sqrt(2*energy/mass); tof=1.0/speed
    out=neutron_energy_from_time_of_flight_keV(1.0,tof)
    assert out["neutron_energy_keV"] == pytest.approx(2450.0,rel=1e-12)


def test_fusion_power_from_reaction_rate_relation():
    from tokamak_analysis.fusion_rate import fusion_power_from_reaction_rate
    out=fusion_power_from_reaction_rate(1e18,1.0,17.6)
    assert out["fusion_power_W"] == pytest.approx(1e18*17.6*1.602176634e-13,rel=1e-14)


def test_gamma_peak_activity_relation():
    from tokamak_analysis.nuclear_gamma import activity_from_gamma_peak_rate
    out=activity_from_gamma_peak_rate(100.0,0.2,0.5)
    assert out["activity_Bq"] == pytest.approx(1000.0)


def test_runaway_electron_field_energy_gain_relation():
    from tokamak_analysis.runaway_electrons import electric_field_energy_gain_eV
    out=electric_field_energy_gain_eV(5.0,100.0)
    assert out["energy_gain_eV"] == pytest.approx(500.0)


def test_relativistic_particle_kinetic_energy_relation():
    from tokamak_analysis.energetic_particles import relativistic_kinetic_energy_keV
    mass=9.1093837139e-31; c=299792458.0; energy=1000.0
    p=np.sqrt((energy*1.602176634e-16)*(energy*1.602176634e-16+2*mass*c*c))/c
    out=relativistic_kinetic_energy_keV(p,mass)
    assert out["kinetic_energy_keV"] == pytest.approx(energy,rel=1e-12)


def test_exponential_particle_loss_rate_relation():
    from tokamak_analysis.energetic_particles import exponential_confinement_loss_rate
    assert exponential_confinement_loss_rate(1e12,0.02)["particle_loss_rate_s-1"] == pytest.approx(5e13)


def test_relativistic_runaway_speed_relation():
    from tokamak_analysis.runaway_energy import runaway_speed_from_kinetic_energy
    out=runaway_speed_from_kinetic_energy(0.0)
    assert out["speed_m_s"] == pytest.approx(0.0)
    assert out["gamma_factor"] == pytest.approx(1.0)


def test_energetic_particle_flux_area_efficiency_relation():
    from tokamak_analysis.energetic_particles import energetic_particle_flux
    out=energetic_particle_flux([0.0,1.0,2.0],[0.0,10.0,20.0],effective_area_m2=2.0,efficiency=0.5,counts_are_cumulative=True,timestamp_semantics="counter_read_times")
    np.testing.assert_allclose(out["particle_flux_m-2_s-1"],[10.0,10.0,10.0])


def test_vacuum_pressure_gas_balance_relation():
    from tokamak_analysis.vacuum_pressure import pressure_rate_from_gas_load
    out=pressure_rate_from_gas_load(10.0,2.0,3.0,4.0)
    assert out["pressure_rate_Pa_s"] == pytest.approx(1.0)


def test_gas_puff_molar_throughput_relation():
    from tokamak_analysis.gas_puffing import molar_flow_to_gas_throughput_Pa_m3_s
    out=molar_flow_to_gas_throughput_Pa_m3_s(2.0,300.0)
    assert out["gas_throughput_Pa_m3_s"] == pytest.approx(2*8.314462618*300)


def test_fueling_efficiency_relation():
    from tokamak_analysis.fueling_rate import fueling_efficiency_from_particle_inventory
    assert fueling_efficiency_from_particle_inventory(2e19,5e19)["fueling_efficiency"] == pytest.approx(0.4)


def test_neutral_mean_free_path_relation():
    from tokamak_analysis.neutral_gas import neutral_mean_free_path_m, K_B
    out=neutral_mean_free_path_m(1.0,300.0,3e-10)
    assert out["mean_free_path_m"] == pytest.approx(K_B*300/(np.sqrt(2)*np.pi*(3e-10)**2))


def test_net_wall_particle_sink_relation():
    from tokamak_analysis.wall_recycling import net_wall_particle_sink
    assert net_wall_particle_sink(10.0,7.0)["net_wall_sink_flux_m-2_s-1"] == pytest.approx(3.0)


def test_mirnov_faraday_law_relation():
    from tokamak_analysis.mirnov_analysis import magnetic_perturbation_from_mirnov_voltage
    t=np.linspace(0.0,1.0,101); n=10; area=0.01; slope=2.0
    out=magnetic_perturbation_from_mirnov_voltage(t,np.full_like(t,-n*area*slope),n,area)
    np.testing.assert_allclose(out["delta_B_T"],slope*t,rtol=1e-12,atol=1e-12)


def test_helical_mhd_mode_phase_relation():
    from tokamak_analysis.mhd_modes import helical_mode_phase
    out=helical_mode_phase(2,1,np.pi,0.0,2*np.pi,0.5)
    assert out["helical_phase_rad"] == pytest.approx(2*np.pi- np.pi)


def test_mode_frequency_from_phase_slope():
    from tokamak_analysis.magnetic_mode_frequency import frequency_from_unwrapped_phase
    t=np.linspace(0,1,101); f=25.0
    out=frequency_from_unwrapped_phase(t,2*np.pi*f*t)
    assert out["mode_frequency_Hz"] == pytest.approx(f,rel=1e-12)


def test_fft_frequency_resolution_relation():
    from tokamak_analysis.fft_analysis import frequency_resolution_Hz
    t=np.linspace(0,2,201)
    assert frequency_resolution_Hz(t)["frequency_resolution_Hz"] == pytest.approx(1.0/(len(t)*np.median(np.diff(t))))


def test_svd_truncation_error_relation():
    from tokamak_analysis.svd_analysis import svd_truncation_frobenius_error
    out=svd_truncation_frobenius_error([5.0,3.0,1.0],1)
    assert out["minimum_frobenius_error"] == pytest.approx(np.sqrt(10.0))
    assert out["retained_energy_fraction"] == pytest.approx(25/35)


def test_morlet_time_frequency_scales_relation():
    from tokamak_analysis.wavelet_analysis import morlet_time_frequency_scales
    out=morlet_time_frequency_scales(100.0,6.0)
    assert out["time_frequency_product"] == pytest.approx(1/(2*np.pi),rel=1e-14)


def test_turbulence_mixing_length_relation():
    from tokamak_analysis.turbulence import mixing_length_diffusivity
    assert mixing_length_diffusivity(1000.0,10.0)["mixing_length_diffusivity_m2_s"] == pytest.approx(10.0)


def test_radial_and_poloidal_fick_law_relations():
    from tokamak_analysis.transport import radial_diffusive_particle_flux, poloidal_diffusive_particle_flux
    assert radial_diffusive_particle_flux(2.0,-3.0)["radial_particle_flux_m-2_s-1"] == pytest.approx(6.0)
    assert poloidal_diffusive_particle_flux(4.0,2.0)["poloidal_particle_flux_m-2_s-1"] == pytest.approx(-8.0)


def test_turbulent_momentum_flux_relation():
    from tokamak_analysis.transport import turbulent_momentum_flux
    out=turbulent_momentum_flux([1.0,2.0,3.0],[3.0,2.0,1.0],2.0)
    assert out["velocity_covariance_m2_s2"] == pytest.approx(-2/3)
    assert out["turbulent_momentum_flux_Pa"] == pytest.approx(4/3)


def test_transient_energy_decay_confinement_relation():
    from tokamak_analysis.confinement import confinement_time_from_energy_decay
    t=np.linspace(0.0,2.0,101); tau=0.4; W=100*np.exp(-t/tau)
    out=confinement_time_from_energy_decay(t,W)
    assert out["energy_decay_confinement_time_s"] == pytest.approx(tau,rel=1e-12)


def test_fourier_conductive_heat_flux_relation():
    from tokamak_analysis.heat_flux import conductive_heat_flux_from_gradient
    assert conductive_heat_flux_from_gradient(5.0,2.0)["conductive_heat_flux_W_m2"] == pytest.approx(-10.0)


def test_electron_ion_heating_partition_relations():
    from tokamak_analysis.heating_analysis import electron_heating_power_from_partition, ion_heating_power_from_partition
    assert electron_heating_power_from_partition(1000.0,0.3)["electron_heating_power_W"] == pytest.approx(300.0)
    assert ion_heating_power_from_partition(1000.0,0.3)["ion_heating_power_W"] == pytest.approx(700.0)


def test_relativistic_ecrh_resonance_relation():
    from tokamak_analysis.ecrh_icrh import relativistic_ecrh_resonance_field, ecrh_resonance_field
    base=ecrh_resonance_field(140e9)["resonance_field_T"]
    shifted=relativistic_ecrh_resonance_field(140e9,gamma_factor=2.0)["resonance_field_T"]
    assert shifted == pytest.approx(2*base,rel=1e-14)


def test_rf_power_balance_relation():
    from tokamak_analysis.ecrh_icrh import rf_power_balance
    out=rf_power_balance(1000.0,100.0,50.0)
    assert out["net_delivered_power_W"] == pytest.approx(850.0)
    assert out["reflection_fraction"] == pytest.approx(0.1)


def test_divertor_sheath_heat_flux_relation():
    from tokamak_analysis.divertor_analysis import sheath_heat_flux_to_divertor
    out=sheath_heat_flux_to_divertor(1e19,10.0,2.0,7.0,1.0,2.0)
    assert out["target_heat_flux_W_m2"] == pytest.approx(out["parallel_sheath_heat_flux_W_m2"]*0.5)


def test_pfc_surface_power_integral_relation():
    from tokamak_analysis.divertor_analysis import total_pfc_power_from_surface_profile
    s=np.linspace(0,1,101); q=np.full_like(s,10.0); R=np.full_like(s,2.0)
    out=total_pfc_power_from_surface_profile(s,q,R)
    assert out["total_pfc_power_W"] == pytest.approx(40*np.pi)


def test_debye_length_relation():
    from tokamak_analysis.edge_field import debye_length_m
    ne=1e19; te=10.0; e=1.602176634e-19; eps=8.8541878128e-12
    out=debye_length_m(ne,te)
    assert out["debye_length_m"] == pytest.approx(np.sqrt(eps*te/(ne*e)),rel=1e-14)


def test_martin_lh_threshold_scaling():
    from tokamak_analysis.lh_transition import martin_lh_threshold_power_MW
    out=martin_lh_threshold_power_MW(1e20,1.0,10.0)
    assert out["lh_threshold_power_MW"] == pytest.approx(0.0488*10**0.941,rel=1e-14)


def test_affine_signal_calibration_relation():
    from tokamak_analysis.signal_calibration import fit_linear_calibration
    out=fit_linear_calibration([0,1,2],[1,3,5])
    assert out["gain"] == pytest.approx(2.0)
    assert out["offset"] == pytest.approx(1.0)


def test_robust_noise_sigma_relation():
    from tokamak_analysis.denoising import robust_noise_sigma_from_residual
    out=robust_noise_sigma_from_residual([-1.0,0.0,1.0,-2.0,2.0])
    assert out["robust_noise_sigma"] == pytest.approx(1.4826)


def test_sum_uncertainty_propagation_relation():
    from tokamak_analysis.measurement_uncertainty import propagate_sum_uncertainty
    out=propagate_sum_uncertainty([2.0,3.0],[0.3,0.4])
    assert out["sum_value"] == pytest.approx(5.0)
    assert out["sum_standard_uncertainty"] == pytest.approx(0.5)


def test_channel_lag_cross_correlation_relation():
    from tokamak_analysis.channel_synchronization import estimate_channel_time_lag
    t=np.arange(0.0,10.0,0.01); ref=np.sin(2*np.pi*1.0*t); target=np.sin(2*np.pi*1.0*(t-0.2))
    out=estimate_channel_time_lag(t,ref,t,target)
    assert out["target_delay_relative_to_reference_s"] == pytest.approx(0.2,abs=0.02)


def test_divertor_sheath_particle_flux_relation():
    from tokamak_analysis.divertor_analysis import divertor_ion_particle_flux_from_sheath
    out=divertor_ion_particle_flux_from_sheath(1e19,10.0,2.0)
    assert out["divertor_ion_particle_flux_m-2_s-1"] == pytest.approx(1e19*out["sound_speed_m_s"])


def test_registered_shape_analysis_exposes_used_physical_relations():
    from tokamak_analysis.shape_reconstruction import analyze_plasma_shape
    R=np.linspace(1.0,3.0,101); Z=np.linspace(-1.0,1.0,101)
    RR,ZZ=np.meshgrid(R,Z); psi=((RR-2.0)/0.5)**2+(ZZ/0.5)**2
    out=analyze_plasma_shape(R,Z,psi,boundary_level=0.95)
    assert "cross_section_area" in out["physical_relations"]
    assert out["cross_section_area_m2"] > 0


def test_registered_current_analysis_exposes_physical_relations():
    from tokamak_analysis.plasma_current import analyze_plasma_current
    t=np.linspace(0,1,11); out=analyze_plasma_current(t,100*t)
    assert "dIp/dt" in out["physical_relations"]["current_ramp_rate"]


def test_registered_magnetic_field_analysis_exposes_physical_relations():
    from tokamak_analysis.magnetic_field import analyze_magnetic_field
    out=analyze_magnetic_field(3.0,4.0)
    assert out["B_total_T"] == pytest.approx(5.0)
    assert "sqrt(Bt^2+Bp^2)" in out["physical_relations"]["field_magnitude"]

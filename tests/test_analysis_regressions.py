import numpy as np
import pytest

from tokamak_analysis.safety_factor import estimate_safety_factor
from tokamak_analysis.neutron_flux import analyze_neutron_flux
from tokamak_analysis.spectral_line_ratio import analyze_line_ratio
from tokamak_analysis.wall_recycling import recycling_coefficient
from tokamak_analysis.measurement_uncertainty import propagate_product_uncertainty
from tokamak_analysis.channel_synchronization import synchronize_channels
from tokamak_analysis.plasma_geometry import geometry_from_boundary


def test_cylindrical_safety_factor_uses_minor_over_major_radius():
    result = estimate_safety_factor(2.0, 0.5, 1.0, 0.1)
    assert result["q_cylindrical_estimate"] == pytest.approx(2.5)


def test_neutron_flux_treats_input_as_binned_counts_by_default():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(result["count_rate_s-1"], [10.0, 20.0, 30.0])
    assert result["total_counts"] == 60.0


def test_neutron_flux_supports_cumulative_counters_explicitly():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [0.0, 10.0, 30.0], counts_are_cumulative=True)
    assert result["total_counts"] == 30.0
    np.testing.assert_allclose(result["count_rate_s-1"], [10.0, 15.0, 20.0])


def test_line_ratio_uncertainty_is_finite_for_zero_numerator():
    result = analyze_line_ratio(0.0, 2.0, uncertainty_numerator=0.1, uncertainty_denominator=0.2)
    assert result["line_ratio"] == 0.0
    assert result["line_ratio_uncertainty"] == pytest.approx(0.05)


def test_recycling_uncertainty_is_finite_for_zero_returned_flux():
    result = recycling_coefficient(2.0, 0.0, uncertainties=(0.1, 0.2))
    assert result["recycling_coefficient"] == 0.0
    assert result["uncertainty"] == pytest.approx(0.1)


def test_product_uncertainty_handles_zero_factor():
    result = propagate_product_uncertainty([0.0, 3.0], [2.0, 0.1])
    assert result["value"] == 0.0
    assert result["standard_uncertainty"] == pytest.approx(6.0)
    assert np.isnan(result["relative_uncertainty"])


def test_synchronization_rejects_extrapolation():
    with pytest.raises(ValueError, match="inside"):
        synchronize_channels([[0.0, 1.0, 2.0], [0.5, 1.0, 1.5]],
                              [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                              target_time=[0.0, 1.0])


def test_geometry_rejects_boundary_crossing_symmetry_axis():
    with pytest.raises(ValueError, match="symmetry axis"):
        geometry_from_boundary([-0.1, 0.5, 0.5, -0.1, -0.1],
                               [0.0, 0.0, 1.0, 1.0, 0.0])


def test_fft_one_sided_amplitude_does_not_double_nyquist_bin():
    from tokamak_analysis.fft_analysis import analyze_fft

    time = np.arange(8, dtype=float) / 8.0
    signal = (-1.0) ** np.arange(8)
    result = analyze_fft(time, signal, detrend=False, window="rectangular")
    assert result["amplitude"][-1] == pytest.approx(1.0)


def test_fft_frequency_resolution_uses_n_times_sample_interval():
    from tokamak_analysis.fft_analysis import frequency_resolution_Hz

    time = np.arange(8, dtype=float) * 0.25
    result = frequency_resolution_Hz(time)
    assert result["frequency_resolution_Hz"] == pytest.approx(0.5)


def test_confinement_uncertainty_handles_zero_stored_energy():
    from tokamak_analysis.confinement import energy_confinement_time

    result = energy_confinement_time(0.0, 10.0, uncertainty_energy_J=2.0, uncertainty_power_W=1.0)
    assert result["energy_confinement_time_s"] == 0.0
    assert result["uncertainty_s"] == pytest.approx(0.2)


def test_fft_rejects_degenerate_two_sample_hann_window():
    from tokamak_analysis.fft_analysis import analyze_fft

    with pytest.raises(ValueError, match="at least 3"):
        analyze_fft([0.0, 1.0], [0.0, 1.0], window="hann")


def test_radial_and_poloidal_registry_entries_use_distinct_directions():
    from tokamak_analysis.analysis_registry import resolve_analysis

    radial = resolve_analysis(61)([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    poloidal = resolve_analysis(62)([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
    assert radial["direction"] == "radial"
    assert poloidal["direction"] == "poloidal"


def test_electron_and_ion_heating_registry_entries_are_species_specific():
    from tokamak_analysis.analysis_registry import resolve_analysis

    electron = resolve_analysis(66)([0.0, 1.0], [10.0, 10.0])
    ion = resolve_analysis(67)([0.0, 1.0], [10.0, 10.0])
    assert electron["species"] == "electron"
    assert ion["species"] == "ion"


def test_divertor_particle_flux_entry_does_not_require_heat_flux():
    from tokamak_analysis.analysis_registry import resolve_analysis

    result = resolve_analysis(74)([0.0, 1.0, 2.0], [1.0, 2.0, 1.0])
    assert result["peak_particle_flux_m-2_s-1"] == 2.0
    assert result["integrated_particle_flux_m-1_s-1"] == pytest.approx(3.0)


def test_pfc_heat_load_entry_requires_area_and_reports_peak_load():
    from tokamak_analysis.analysis_registry import resolve_analysis

    result = resolve_analysis(72)([0.0, 1.0], [10.0, 20.0], 3.0)
    assert result["peak_power_load_W"] == pytest.approx(60.0)
    assert result["surface_area_m2"] == 3.0


def test_energetic_particle_flux_uses_binned_counts_per_bin_width():
    from tokamak_analysis.energetic_particles import energetic_particle_flux

    result = energetic_particle_flux([0.0, 1.0, 2.0], [10.0, 20.0, 30.0], 2.0, efficiency=0.5)
    np.testing.assert_allclose(result["particle_flux_m-2_s-1"], [10.0, 20.0, 30.0])


def test_plasma_current_distinguishes_signed_peak_from_absolute_peak():
    from tokamak_analysis.plasma_current import analyze_plasma_current

    result = analyze_plasma_current([0.0, 1.0, 2.0], [-5.0, 2.0, 3.0])
    assert result["peak_signed_current_A"] == pytest.approx(3.0)
    assert result["peak_absolute_current_A"] == pytest.approx(5.0)
    assert result["current_at_peak_absolute_magnitude_A"] == pytest.approx(-5.0)


def test_mhd_mode_estimate_sorts_unsorted_sensor_angles():
    from tokamak_analysis.mhd_modes import estimate_mode_number

    angles = np.array([2 * np.pi / 3, 0.0, 4 * np.pi / 3])
    amplitudes = np.exp(1j * angles)
    result = estimate_mode_number(angles, amplitudes)
    assert result["mode_number_estimate"] == pytest.approx(1.0)


def test_divertor_bohm_flux_uses_electron_temperature():
    from tokamak_analysis.divertor_analysis import divertor_ion_particle_flux_from_sheath

    result = divertor_ion_particle_flux_from_sheath(1e19, 10.0, 2.0)
    expected_sound_speed = np.sqrt(10.0 * 1.602176634e-19 / (2.0 * 1.66053906660e-27))
    assert result["sound_speed_m_s"] == pytest.approx(expected_sound_speed)


def test_large_aspect_ratio_shafranov_shift_includes_geometric_half_term():
    from tokamak_analysis.shafranov_shift import large_aspect_ratio_shafranov_shift

    result = large_aspect_ratio_shafranov_shift(0.5, 1.0, 1.0, 1.0)
    assert result["predicted_shafranov_shift_m"] == pytest.approx(0.125)


def test_langmuir_temperature_fit_subtracts_ion_saturation_baseline():
    from tokamak_analysis.probe_analysis import analyze_probe_voltage_current

    voltage = np.arange(-5.0, 5.0)
    current = np.full_like(voltage, -1.0)
    mask = voltage >= 0
    current[mask] = -1.0 + np.exp((voltage[mask] - 4.0) / 2.0)
    result = analyze_probe_voltage_current(voltage, current, electron_fit_range=(0.0, 4.0))
    assert result["electron_temperature_estimate_eV"] == pytest.approx(2.0)


def test_volume_and_cross_section_registry_entries_are_distinct():
    from tokamak_analysis.analysis_registry import resolve_analysis

    R = [1.5, 2.5, 2.5, 1.5, 1.5]
    Z = [-0.5, -0.5, 0.5, 0.5, -0.5]
    volume = resolve_analysis(11)(R, Z)
    area = resolve_analysis(12)(R, Z)
    assert volume["toroidal_volume_m3"] == pytest.approx(4 * np.pi)
    assert area["cross_section_area_m2"] == pytest.approx(1.0)


def test_bolometric_tomography_has_diagnostic_specific_entry_point():
    from tokamak_analysis.analysis_registry import resolve_analysis
    result = resolve_analysis(39)([[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0], regularization=0.0)
    assert result["diagnostic_type"] == "bolometric tomography"
    np.testing.assert_allclose(result["emissivity"], [2.0, 3.0], atol=1e-6)


def test_x_point_detection_does_not_duplicate_a_single_saddle():
    from tokamak_analysis.x_point import find_x_point_candidates

    R = np.linspace(1.0, 3.0, 9)
    Z = np.linspace(-1.0, 1.0, 9)
    RR, ZZ = np.meshgrid(R, Z)
    psi = (RR - 2.0) ** 2 - ZZ ** 2
    result = find_x_point_candidates(R, Z, psi, gradient_tolerance=1e-10)
    assert result["count"] == 1
    assert result["candidates"][0]["R_m"] == pytest.approx(2.0)
    assert result["candidates"][0]["Z_m"] == pytest.approx(0.0)


def test_interferometry_rejects_phase_sign_that_implies_negative_density():
    from tokamak_analysis.interferometry import interferometry_density
    with pytest.raises(ValueError, match="negative line density"):
        interferometry_density(-1.0, 10e9, sign=1)


def test_reflectometry_zero_spectrum_has_no_defined_peak():
    from tokamak_analysis.reflectometry import analyze_reflectometry_spectrum
    result = analyze_reflectometry_spectrum([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert result["peak_frequency_Hz"] is None


def test_plasma_wave_spectrum_zero_signal_has_no_dominant_frequency():
    from tokamak_analysis.plasma_wave_spectrum import analyze_wave_signal
    result = analyze_wave_signal(np.arange(16, dtype=float) * 0.01, np.zeros(16))
    assert result["dominant_frequency_Hz"] is None


def test_empty_xray_and_neutron_spectra_have_no_peak_energy():
    from tokamak_analysis.hard_xray import analyze_hard_xray
    from tokamak_analysis.xray_spectrum import analyze_xray_spectrum
    from tokamak_analysis.neutron_spectrum import analyze_neutron_spectrum
    assert analyze_hard_xray([1.0, 2.0], [0.0, 0.0])["peak_energy_keV"] is None
    assert analyze_xray_spectrum([1.0, 2.0], [0.0, 0.0])["peak_energy_keV"] is None
    assert analyze_neutron_spectrum([1.0, 2.0], [0.0, 0.0])["peak_energy_keV"] is None


def test_abel_projection_on_axis_has_finite_endpoint():
    from tokamak_analysis.radiation_tomography import abel_project_axisymmetric_emissivity
    result = abel_project_axisymmetric_emissivity([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], [0.0])
    assert np.all(np.isfinite(result["line_integrated_emissivity_W_m2"]))


def test_tomography_uses_stable_least_squares_for_identity_response():
    from tokamak_analysis.radiation_tomography import reconstruct_emissivity
    result = reconstruct_emissivity([[1.0, 0.0], [0.0, 1.0]], [2.0, 3.0], regularization=0.0, nonnegative=False)
    np.testing.assert_allclose(result["emissivity"], [2.0, 3.0])


def test_empty_energetic_particle_distribution_has_no_peak_energy():
    from tokamak_analysis.energetic_particles import analyze_energy_distribution
    result = analyze_energy_distribution([1.0, 2.0], [0.0, 0.0])
    assert result["peak_energy_keV"] is None


def test_runaway_energy_fit_rejects_non_decaying_spectrum():
    from tokamak_analysis.runaway_energy import estimate_runaway_energy
    with pytest.raises(ValueError, match="non-negative log-slope"):
        estimate_runaway_energy([10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0])


def test_zero_magnetic_signal_has_no_dominant_mode_frequency():
    from tokamak_analysis.magnetic_mode_frequency import magnetic_mode_frequency
    result = magnetic_mode_frequency(np.arange(16, dtype=float), np.zeros(16))
    assert result["dominant_frequency_Hz"] is None


def test_channel_lag_rejects_constant_signals():
    from tokamak_analysis.channel_synchronization import estimate_channel_time_lag
    t = np.arange(8, dtype=float)
    with pytest.raises(ValueError, match="constant or zero-variance"):
        estimate_channel_time_lag(t, np.ones(8), t, np.ones(8))


def test_lh_transition_rejects_invalid_baseline_window():
    from tokamak_analysis.lh_transition import detect_lh_transition
    t = np.arange(12, dtype=float)
    with pytest.raises(ValueError, match="finite increasing bounds"):
        detect_lh_transition(t, np.zeros(12), baseline_window=(8.0, 2.0))


def test_channel_lag_sign_is_positive_when_target_is_delayed():
    from tokamak_analysis.channel_synchronization import estimate_channel_time_lag
    t = np.arange(20, dtype=float)
    reference = np.zeros(20); reference[5] = 1.0
    target = np.zeros(20); target[7] = 1.0
    result = estimate_channel_time_lag(t, reference, t, target)
    assert result["target_delay_relative_to_reference_s"] == pytest.approx(2.0)

"""Independent analytic/reference checks for the remaining mapped formula primitives."""
import math

import numpy as np
import pytest

from tokamak_analysis.formula_library import (
    apparent_feature_speed,
    debye_parameter,
    doppler_line_of_sight_velocity,
    electron_plasma_angular_frequency,
    energy_confinement_time,
    faraday_induction,
    gas_pumping_speed,
    gaussian_fwhm_from_sigma,
    greenwald_density_reference,
    injected_particle_number,
    integrated_surface_power,
    ion_saturation_particle_flux_proxy,
    ion_temperature_from_doppler_sigma,
    magnitude_squared_coherence,
    magnetic_energy_density,
    nyquist_frequency,
    ohmic_power_approximation,
    particle_balance_rate,
    particle_confinement_time_proxy,
    plasma_frequency_from_density_cm3,
    plasma_loop_resistance_proxy,
    radiated_power_fraction,
    relative_fluctuation_level,
    rms_signal,
    safety_factor_large_aspect_ratio,
    toroidal_beta,
)

E = 1.602176634e-19
EPS0 = 8.8541878128e-12
ME = 9.1093837139e-31
MU0 = 1.25663706212e-6
C = 299792458.0
AMU = 1.66053906660e-27


def test_plasma_frequency_cm3_unit_conversion_reference():
    density_cm3 = 1.0e12
    expected = math.sqrt(density_cm3 * 1e6 * E**2 / (EPS0 * ME)) / (2 * math.pi)
    assert plasma_frequency_from_density_cm3(density_cm3)["frequency_pe_Hz"] == pytest.approx(expected, rel=1e-12)


def test_doppler_velocity_sign_and_unit_invariance():
    result = doppler_line_of_sight_velocity(0.01, 500.0)
    assert result["velocity_los_m_s"] == pytest.approx(C * 0.01 / 500.0, rel=1e-12)
    assert doppler_line_of_sight_velocity(-0.01, 500.0)["velocity_los_m_s"] < 0


def test_doppler_temperature_uses_corrected_gaussian_sigma_relation():
    sigma, wavelength, mass = 0.1, 500.0, 40 * AMU
    expected = mass * C**2 / E * (sigma / wavelength)**2
    assert ion_temperature_from_doppler_sigma(sigma, wavelength, mass)["ion_temperature_eV"] == pytest.approx(expected, rel=1e-12)


def test_gaussian_fwhm_conversion():
    factor = 2 * math.sqrt(2 * math.log(2))
    result = gaussian_fwhm_from_sigma(2.0, sigma_sigma_width=0.1)
    assert result["fwhm_width"] == pytest.approx(2 * factor)
    assert result["uncertainty_fwhm_width"] == pytest.approx(0.1 * factor)


def test_magnetic_energy_density_and_toroidal_beta_references():
    assert magnetic_energy_density(2.0)["magnetic_energy_density_J_m3"] == pytest.approx(4.0 / (2 * MU0))
    assert toroidal_beta(1000.0, 2.0)["toroidal_beta"] == pytest.approx(2 * MU0 * 1000.0 / 4.0)


def test_large_aspect_ratio_safety_factor_estimate():
    result = safety_factor_large_aspect_ratio(0.3, 1.5, 2.0, 0.2)
    assert result["safety_factor_estimate"] == pytest.approx(2.0)


def test_faraday_induction_sign_and_magnitude():
    assert faraday_induction(2.0, 5)["coil_voltage_V"] == pytest.approx(-10.0)


def test_loop_resistance_and_ohmic_power_proxy():
    assert plasma_loop_resistance_proxy(4.0, 2.0)["loop_resistance_proxy_ohm"] == pytest.approx(2.0)
    assert ohmic_power_approximation(4.0, 2.0)["ohmic_power_approximation_W"] == pytest.approx(8.0)


def test_energy_confinement_time():
    assert energy_confinement_time(1000.0, 500.0)["energy_confinement_time_s"] == pytest.approx(2.0)


def test_greenwald_density_reference_units():
    result = greenwald_density_reference(1.0, 0.5)
    assert result["greenwald_density_1e20_m3"] == pytest.approx(1.0 / (math.pi * 0.5**2))
    assert result["greenwald_density_m3"] == pytest.approx(result["greenwald_density_1e20_m3"] * 1e20)


def test_debye_parameter_reference():
    expected = (4 * math.pi / 3) * 1e18 * (1e-4)**3
    assert debye_parameter(1e18, 1e-4)["debye_parameter_particles"] == pytest.approx(expected)


def test_electron_plasma_angular_frequency_view():
    expected = math.sqrt(1e18 * E**2 / (EPS0 * ME))
    assert electron_plasma_angular_frequency(1e18)["omega_pe_rad_s"] == pytest.approx(expected, rel=1e-12)


def test_ion_saturation_particle_flux_proxy_reference():
    result = ion_saturation_particle_flux_proxy(1e-3, 1e-4)
    assert result["ion_saturation_particle_flux_m2_s"] == pytest.approx(1e-3 / (E * 1e-4), rel=1e-12)


def test_radiated_power_fraction_and_quality_flag():
    assert radiated_power_fraction(20.0, 100.0)["radiated_power_fraction"] == pytest.approx(0.2)
    assert "quality_flag" in radiated_power_fraction(120.0, 100.0)


def test_integrated_surface_power_weighted_sum():
    assert integrated_surface_power([10.0, 20.0], [0.5, 0.5])["integrated_surface_power_W"] == pytest.approx(15.0)


def test_apparent_feature_speed_preserves_displacement_sign():
    assert apparent_feature_speed(-0.2, 0.1)["apparent_feature_speed_m_s"] == pytest.approx(-2.0)


def test_gas_pumping_speed_units():
    assert gas_pumping_speed(2.0, 4.0)["gas_pumping_speed_m3_s"] == pytest.approx(0.5)


def test_injected_particle_number_trapezoid():
    assert injected_particle_number([2.0, 2.0, 2.0], [0.0, 1.0, 2.0])["injected_particle_number"] == pytest.approx(4.0)


def test_particle_confinement_time_proxy():
    assert particle_confinement_time_proxy(100.0, 20.0)["particle_confinement_time_s"] == pytest.approx(5.0)


def test_relative_fluctuation_level_population_rms():
    result = relative_fluctuation_level([9.0, 10.0, 11.0])
    assert result["mean_signal"] == pytest.approx(10.0)
    assert result["fluctuation_rms"] == pytest.approx(math.sqrt(2.0 / 3.0))
    assert result["relative_fluctuation_level"] == pytest.approx(math.sqrt(2.0 / 3.0) / 10.0)


def test_rms_signal_includes_dc_component():
    assert rms_signal([3.0, 4.0])["rms_signal"] == pytest.approx(math.sqrt(12.5))


def test_nyquist_frequency_is_half_sample_rate():
    assert nyquist_frequency(100.0)["nyquist_frequency_Hz"] == pytest.approx(50.0)


def test_cross_correlation_lag_positive_when_y_lags_x():
    x = np.zeros(32)
    y = np.zeros(32)
    x[5] = 1.0
    y[8] = 1.0
    result = __import__("tokamak_analysis.formula_library", fromlist=["cross_correlation_lag"]).cross_correlation_lag(x, y, 0.01)
    assert result["cross_correlation_lag_s"] == pytest.approx(0.03)
    assert result["normalized_peak_correlation"] > 0.0


def test_particle_balance_signed_net_rate():
    result = particle_balance_rate(10.0, 7.0, 4.0, 2.0)
    assert result["particle_balance_rate_s"] == pytest.approx(5.0)


def test_uncertainty_inputs_must_be_paired_or_complete():
    with pytest.raises(ValueError):
        energy_confinement_time(10.0, 2.0, sigma_stored_energy_J=1.0)
    with pytest.raises(ValueError):
        safety_factor_large_aspect_ratio(0.3, 1.5, 2.0, 0.2, sigma_r_m=0.01)

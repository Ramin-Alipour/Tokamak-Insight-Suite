"""Regression tests for scientific invariants and defensive input validation.

These are software-level checks using analytical/synthetic values, not validation
against a particular tokamak or diagnostic calibration.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tis.core import (
    AMU, C_LIGHT, E_CHARGE, EPS0, M_E, MU0,
    corrected_line_ratio, derive_parameters, doppler_temperature_eV,
    doppler_velocity_m_s, electron_cyclotron_resonance_field, fluctuation_flux,
    exponential_heat_flux_fit, floating_potential, integrate_profile,
    linear_phase_density, neutron_rate_from_counts, reflectometry_cutoff_density,
    spectrum_moments, uncertainty_linear, xray_spectrum_summary,
)
from tis.catalog import DIAGNOSTICS, FORMULAS


def test_doppler_temperature_matches_thermal_gaussian_sigma_relation():
    sigma_lambda, lambda0, mass_amu = 0.01, 500.0, 40.0
    expected = mass_amu * AMU * C_LIGHT**2 / E_CHARGE * (sigma_lambda / lambda0)**2
    assert doppler_temperature_eV(sigma_lambda, lambda0, mass_amu) == pytest.approx(expected)


def test_doppler_temperature_zero_width_is_zero():
    assert doppler_temperature_eV(0.0, 500.0, 40.0) == 0.0


@pytest.mark.parametrize("args", [(-1, 500, 40), (1, 0, 40), (1, 500, 0)])
def test_doppler_temperature_rejects_invalid_inputs(args):
    with pytest.raises(ValueError):
        doppler_temperature_eV(*args)


def test_doppler_velocity_uses_small_shift_approximation():
    assert doppler_velocity_m_s(0.1, 500.0) == pytest.approx(C_LIGHT * 0.1 / 500.0)


def test_derived_plasma_pressure_uses_electron_volts_as_energy_per_charge():
    ne, te = 2.0e19, 50.0
    result = derive_parameters(ne, te, 0, 1e5, 1, 1.5, 0.5, 2.0)
    assert result["electron_pressure_Pa"] == pytest.approx(ne * te * E_CHARGE)
    assert result["debye_length_m"] == pytest.approx(np.sqrt(EPS0 * te / (ne * E_CHARGE)))


def test_energy_confinement_time_requires_positive_loss_power():
    assert derive_parameters(1e19, 10, 10, 1e5, 1, 1, .3, 2, 1000, 500)["energy_confinement_time_s"] == 2
    assert derive_parameters(1e19, 10, 10, 1e5, 1, 1, .3, 2, 1000, 0)["energy_confinement_time_s"] is None


def test_invalid_plasma_radius_is_rejected():
    with pytest.raises(ValueError):
        derive_parameters(1e19, 10, 10, 1e5, 1, 0, .3, 2)


def test_floating_potential_interpolates_zero_crossing():
    assert floating_potential([-2, -1, 0, 1, 2], [-2, -1, 1, 2, 3]) == pytest.approx(-0.5)


def test_profile_integral_is_coordinate_aware():
    result = integrate_profile([0, 1, 2], [0, 1, 2])
    assert result["integral_trapezoid"] == pytest.approx(2.0)


def test_profile_integral_rejects_duplicate_coordinates():
    with pytest.raises(ValueError):
        integrate_profile([0, 1, 1], [1, 2, 3])


def test_reflectometry_cutoff_density_scales_with_frequency_squared():
    a = reflectometry_cutoff_density(60e9, mode="O")
    b = reflectometry_cutoff_density(120e9, mode="O")
    assert b["cutoff_density_m-3"] == pytest.approx(4 * a["cutoff_density_m-3"], rel=1e-10)


def test_electron_cyclotron_resonance_field_scales_with_frequency():
    a = electron_cyclotron_resonance_field(60e9, harmonic=1)
    b = electron_cyclotron_resonance_field(120e9, harmonic=1)
    assert b["resonance_field_T"] == pytest.approx(2 * a["resonance_field_T"], rel=1e-10)


def test_neutron_count_rate_efficiency_correction():
    r = neutron_rate_from_counts(1000, efficiency=0.2, geometric_factor=0.5, calibration_factor=2)
    assert r["estimated_source_rate_s-1"] == pytest.approx(20000)


def test_line_ratio_response_correction():
    r = corrected_line_ratio(20, 10, response_1=2, response_2=1)
    assert r["instrument_response_corrected_ratio"] == pytest.approx(1)


def test_spectral_moment_centroid_for_symmetric_peak():
    r = spectrum_moments([-2, -1, 0, 1, 2], [0, 1, 2, 1, 0])
    assert r["mean_energy_eV"] == pytest.approx(0.0)


def test_xray_spectrum_summary_reports_positive_total_counts():
    r = xray_spectrum_summary([1, 2, 3], [2, 5, 2])
    assert r["integrated_counts_keV"] == pytest.approx(7.0)


def test_catalog_and_formula_registry_are_nonempty_and_structured():
    assert len(DIAGNOSTICS) >= 80
    assert len(FORMULAS) >= 35
    assert all(d.get("diagnostic") and d.get("family") and d.get("caveat") for d in DIAGNOSTICS)
    assert all(isinstance(f, dict) and f for f in FORMULAS)


def test_registry_json_files_are_valid_json():
    root = Path(__file__).parents[1]
    for name in ("DIAGNOSTIC_REGISTRY.json", "FORMULA_REGISTRY.json"):
        with (root / name).open(encoding="utf-8") as stream:
            parsed = json.load(stream)
        assert isinstance(parsed, (list, dict))
        assert parsed



def test_doppler_temperature_rejects_nonfinite_inputs():
    with pytest.raises(ValueError):
        doppler_temperature_eV(float("nan"), 500.0, 40.0)
    with pytest.raises(ValueError):
        doppler_temperature_eV(0.01, float("inf"), 40.0)


def test_derived_parameters_reject_negative_ion_density():
    with pytest.raises(ValueError):
        derive_parameters(1e19, 10, 10, 1e5, 1, 1, .3, 2, ni_m3=-1)


def test_neutron_rate_rejects_nonfinite_factors():
    with pytest.raises(ValueError):
        neutron_rate_from_counts(1000, float("nan"))
    with pytest.raises(ValueError):
        neutron_rate_from_counts(1000, 0.2, geometric_factor=float("inf"))


def test_ecr_harmonic_must_be_a_positive_integer():
    with pytest.raises(ValueError):
        electron_cyclotron_resonance_field(60e9, harmonic=1.5)


def test_uncertainty_calibration_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        uncertainty_linear([1, 2], 1, 0, sigma_y=-0.1)
    with pytest.raises(ValueError):
        uncertainty_linear([1, float("nan")], 1, 0)


def test_tomography_rejects_nonfinite_matrix_and_measurements():
    from tis.core import regularized_tomography
    with pytest.raises(ValueError):
        regularized_tomography([[1, float("nan")]], [1])


def test_fluctuation_flux_uses_common_finite_mask():
    result = fluctuation_flux([1, 2, 3, 4], [-1, 0, 1, 2], [10, 20, float("nan"), 40])
    assert np.isfinite(result["particle_flux_covariance_m-2_s-1"])
    assert np.isfinite(result["electron_heat_flux_proxy_W_m-2"])
    assert result["mean_density_m-3"] == pytest.approx(7 / 3)


def test_fft_rejects_nonfinite_signal_and_sampling_rate():
    from tis.core import fft_spectrum
    with pytest.raises(ValueError, match="finite"):
        fft_spectrum([0, 1, np.nan, 2], 100.0)
    with pytest.raises(ValueError, match="sampling rate"):
        fft_spectrum(np.arange(8), float("nan"))


def test_welch_rejects_invalid_segment_length_and_nonfinite_samples():
    from tis.core import welch_psd
    with pytest.raises(ValueError, match="nperseg"):
        welch_psd(np.arange(32, dtype=float), 100.0, nperseg=0)
    with pytest.raises(ValueError, match="finite"):
        welch_psd([0, 1, 2, 3, 4, 5, np.inf, 7], 100.0)


def test_cross_analysis_validates_sampling_and_segment_length():
    from tis.core import cross_analysis
    x = np.arange(32, dtype=float)
    with pytest.raises(ValueError, match="(?i)sampling frequency"):
        cross_analysis(x, x, 0.0)
    with pytest.raises(ValueError, match="nperseg"):
        cross_analysis(x, x, 100.0, nperseg=-1)


def test_cross_correlation_rejects_nonfinite_signal_and_invalid_lag():
    from tis.core import cross_correlation_lag
    x = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="finite"):
        cross_correlation_lag(x, np.r_[x[:-1], np.nan], 100.0)
    with pytest.raises(ValueError, match="Maximum lag"):
        cross_correlation_lag(x, x, 100.0, max_lag_s=-0.1)


def test_filter_rejects_nonfinite_signal_and_noninteger_order():
    from tis.core import butterworth_filter
    x = np.arange(100, dtype=float)
    with pytest.raises(ValueError, match="finite"):
        butterworth_filter(np.r_[x[:-1], np.nan], 1000.0, lowpass=100.0)
    with pytest.raises(ValueError, match="order"):
        butterworth_filter(x, 1000.0, lowpass=100.0, order=0)


@pytest.mark.parametrize("field,value", [("stored_energy_J", -1.0), ("loss_power_W", float("nan"))])
def test_derived_parameters_reject_invalid_confinement_inputs(field, value):
    kwargs = {"stored_energy_J": 1000.0, "loss_power_W": 500.0}
    kwargs[field] = value
    with pytest.raises(ValueError, match="energy|power"):
        derive_parameters(1e19, 10, 10, 1e5, 1, 1.5, .3, 2.0, **kwargs)


def test_mode_number_rejects_zero_or_nonfinite_weights():
    from tis.core import mode_number_from_phase
    theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
    phase = 2*theta
    with pytest.raises(ValueError, match="Weights"):
        mode_number_from_phase(theta, phase, np.zeros_like(theta))
    weights = np.ones_like(theta)
    weights[0] = np.nan
    with pytest.raises(ValueError, match="Weights"):
        mode_number_from_phase(theta, phase, weights)


def test_equilibrium_grid_derivatives_rejects_nonfinite_flux_and_field():
    from tis.core import equilibrium_grid_derivatives
    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    psi = np.add.outer(Z, R)
    with pytest.raises(ValueError, match="finite"):
        equilibrium_grid_derivatives(R, Z, psi * np.nan, 2.0)
    with pytest.raises(ValueError, match="field"):
        equilibrium_grid_derivatives(R, Z, psi, float("inf"))


def test_spectral_moments_reject_duplicate_energy_bins():
    with pytest.raises(ValueError, match="unique"):
        spectrum_moments([1.0, 2.0, 2.0, 3.0], [1.0, 2.0, 3.0, 1.0])


def test_abel_reconstruction_rejects_nonfinite_radial_edges():
    from tis.core import abel_reconstruction
    with pytest.raises(ValueError, match="finite"):
        abel_reconstruction([0.0, 0.1, 0.2], [1.0, 0.8, 0.4], [0.0, 0.1, np.nan, 0.3])


def test_radiated_power_rejects_nonfinite_coordinates():
    from tis.core import axisymmetric_radiated_power
    R = np.array([1.0, 1.5, np.nan, 2.0])
    Z = np.linspace(-0.5, 0.5, 4)
    with pytest.raises(ValueError, match="coordinates"):
        axisymmetric_radiated_power(R, Z, np.ones((4, 4)))


def test_mode_number_requires_distinct_sensor_angles():
    from tis.core import mode_number_from_phase
    with pytest.raises(ValueError, match="distinct"):
        mode_number_from_phase([1, 1, 1], [0, 1, 2])


# Additional signal-analysis invariants added during the v0.4.x audit.
def test_fft_recovers_frequency_of_synthetic_single_tone():
    from tis.core import fft_spectrum
    fs = 2048.0
    n = 4096
    frequency_hz = 128.0
    t = np.arange(n) / fs
    signal = 2.5 * np.sin(2 * np.pi * frequency_hz * t)
    frequencies, amplitudes = fft_spectrum(signal, fs, window="boxcar")
    peak = int(np.argmax(amplitudes[1:]) + 1)
    assert frequencies[peak] == pytest.approx(frequency_hz)
    assert amplitudes[peak] == pytest.approx(2.5, rel=1e-10)


def test_welch_psd_peak_matches_synthetic_mhd_tone():
    from tis.core import welch_psd
    fs = 100_000.0
    n = 16_384
    frequency_hz = 12_207.03125
    t = np.arange(n) / fs
    signal = np.sin(2 * np.pi * frequency_hz * t)
    frequencies, psd = welch_psd(signal, fs, nperseg=4096, noverlap=2048)
    peak = int(np.argmax(psd[1:]) + 1)
    assert frequencies[peak] == pytest.approx(frequency_hz, abs=fs / 4096)
    assert np.all(np.isfinite(psd))
    assert np.all(psd >= 0)


def test_fluctuation_flux_covariance_is_zero_for_constant_radial_velocity():
    density = np.array([1.0, 2.0, 3.0, 4.0, 5.0]) * 1e19
    radial_velocity = np.full(5, 12.0)
    result = fluctuation_flux(density, radial_velocity)
    assert result["particle_flux_covariance_m-2_s-1"] == pytest.approx(0.0, abs=1e-12)


def test_fluctuation_flux_uses_common_finite_sample_mask():
    density = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    radial_velocity = np.array([5.0, 4.0, 100.0, 2.0, 1.0])
    result = fluctuation_flux(density, radial_velocity)
    expected = np.mean((np.array([1.0, 2.0, 4.0, 5.0]) - 3.0) *
                       (np.array([5.0, 4.0, 2.0, 1.0]) - 3.0))
    assert result["particle_flux_covariance_m-2_s-1"] == pytest.approx(expected)



def test_fft_recovers_even_length_nyquist_amplitude_without_factor_of_two_error():
    from tis.core import fft_spectrum
    n = 1024
    fs = 1024.0
    # At Nyquist, a cosine is sampled as an alternating sequence.
    x = 3.25 * (-1.0) ** np.arange(n)
    frequencies, amplitude = fft_spectrum(x, fs, window="boxcar", detrend=False)
    nyquist = int(np.argmax(amplitude))
    assert frequencies[nyquist] == pytest.approx(fs / 2)
    assert amplitude[nyquist] == pytest.approx(3.25, rel=1e-12)


def test_welch_psd_integral_is_consistent_with_white_noise_variance():
    from tis.core import welch_psd
    rng = np.random.default_rng(20261014)
    x = rng.normal(loc=0.0, scale=2.0, size=32768)
    frequencies, psd = welch_psd(x, fs=2000.0, nperseg=2048, noverlap=1024)
    from scipy.integrate import trapezoid
    integrated_power = float(trapezoid(psd, frequencies))
    # For a sufficiently long white-noise realization, integrated one-sided
    # density should recover the variance to statistical accuracy.
    assert integrated_power == pytest.approx(np.var(x), rel=0.08)
    assert np.all(np.isfinite(psd))
    assert np.all(psd >= 0)



def test_cross_correlation_lag_sign_for_known_delayed_signal():
    from tis.core import cross_correlation_lag
    rng = np.random.default_rng(20261014)
    x = rng.normal(size=4096)
    delay_samples = 17
    y = np.r_[np.zeros(delay_samples), x[:-delay_samples]]
    _, _, lag_s = cross_correlation_lag(x, y, fs=2000.0)
    # This API defines negative lag when y occurs later than x.
    assert lag_s == pytest.approx(-delay_samples / 2000.0, abs=1e-12)


def test_cross_spectral_mode_number_recovers_known_positive_phase_winding():
    from tis.core import mode_number_from_channels
    sensor_count = 8
    theta = np.arange(sensor_count) * (2 * np.pi / sensor_count)
    fs = 1024.0
    n = 2048
    t = np.arange(n) / fs
    frequency_hz = 64.0
    true_mode = 2
    signals = np.array([
        np.sin(2 * np.pi * frequency_hz * t + true_mode * angle)
        for angle in theta
    ])
    result = mode_number_from_channels(
        theta, signals, fs,
        frequency_range_hz=(60.0, 68.0),
        min_coherence=0.2,
    )
    assert result["dominant_frequency_Hz"] == pytest.approx(frequency_hz, abs=fs / 512)
    assert result["mode_number_fit"] == pytest.approx(true_mode)
    assert result["phase_rms_residual_rad"] < 1e-8

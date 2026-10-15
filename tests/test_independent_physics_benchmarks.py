"""Independent physics benchmarks with hand-computed references.

These checks validate equations and numerical conventions, not any machine's
calibration. Values are computed independently from the implementation under test.
"""
import math

import numpy as np
import pytest

from tis.core import (
    AMU, C_LIGHT, E_CHARGE, MU0,
    doppler_temperature_eV, doppler_velocity_m_s,
    fft_spectrum, welch_psd, fluctuation_flux,
)
from tis.plasma import greenwald, beta_metrics, ipb98y2_confinement_time, lh_threshold_martin08


def test_doppler_temperature_from_thermal_gaussian_sigma_independent_reference():
    # Thermal 1-D velocity sigma^2 = kT/m; wavelength sigma/lambda = sigma_v/c.
    mass_amu, lambda0_nm, sigma_nm = 40.0, 500.0, 0.01
    expected_eV = mass_amu * AMU * C_LIGHT**2 / E_CHARGE * (sigma_nm/lambda0_nm)**2
    assert doppler_temperature_eV(sigma_nm, lambda0_nm, mass_amu) == pytest.approx(expected_eV, rel=1e-12)


def test_doppler_velocity_nonrelativistic_reference_and_sign():
    # v/c ~= (lambda-lambda0)/lambda0; the routine accepts the signed shift.
    assert doppler_velocity_m_s(0.1, 500.0) == pytest.approx(C_LIGHT*0.1/500.0, rel=1e-12)
    assert doppler_velocity_m_s(-0.1, 500.0) == pytest.approx(-C_LIGHT*0.1/500.0, rel=1e-12)


def test_greenwald_current_and_minor_radius_units():
    # n_G[10^20 m^-3] = I_p[MA]/(pi*a^2); 1 MA and a=0.5 m.
    r = greenwald(0.5e20, 1.0e6, 0.5)
    expected_ng = 1.0/(math.pi*0.5**2)
    assert r["greenwald_density_1e20_m-3"] == pytest.approx(expected_ng, rel=1e-12)
    assert r["greenwald_fraction"] == pytest.approx(0.5/expected_ng, rel=1e-12)


def test_beta_metrics_pressure_and_toroidal_beta_independent_units():
    ne, ni, te, ti, bt = 2e19, 1.5e19, 100.0, 80.0, 2.0
    r = beta_metrics(ne, te, ti, 1e6, bt, 0.5, kappa=1.5, ni_m3=ni)
    p = E_CHARGE*(ne*te + ni*ti)
    assert r["pressure_Pa"] == pytest.approx(p, rel=1e-12)
    assert r["beta_t"] == pytest.approx(2*MU0*p/bt**2, rel=1e-12)
    assert 0 < r["beta_t"] < 1


def test_ipb98y2_reference_equation_and_unit_convention():
    # Explicit substitution into the published empirical expression.
    ip, bt, ploss, n19, mass, R, a, kappa = 1.0, 2.0, 10.0, 5.0, 2.0, 1.7, 0.5, 1.5
    eps = a/R
    expected = (0.0562 * ip**0.93 * bt**0.15 * ploss**(-0.69) *
                n19**0.41 * mass**0.19 * R**1.97 * eps**0.58 * kappa**0.78)
    got = ipb98y2_confinement_time(ip, bt, ploss, n19, mass, R, a, kappa)
    assert got["tau_E_IPB98y2_s"] == pytest.approx(expected, rel=1e-12)


def test_martin_lh_threshold_reference_substitution():
    n20, bt, surface = 0.5, 2.0, 10.0
    expected = 0.0488*n20**0.717*bt**0.803*surface**0.941
    got = lh_threshold_martin08(n20*1e20, bt, surface, ion_mass_amu=2.0)
    assert got["P_LH_MW"] == pytest.approx(expected, rel=1e-12)


def test_fft_single_sided_amplitude_and_frequency_reference():
    fs, n, f0, amplitude = 4096.0, 4096, 256.0, 2.5
    t = np.arange(n)/fs
    f, a = fft_spectrum(amplitude*np.sin(2*np.pi*f0*t), fs, window="boxcar")
    idx = int(np.argmax(a))
    assert f[idx] == pytest.approx(f0, abs=fs/n/2)
    assert a[idx] == pytest.approx(amplitude, rel=1e-12)


def test_welch_psd_integral_recovers_variance_with_tolerance():
    # For a stationary white-noise record, integrated one-sided density estimates
    # the sample variance. Statistical tolerance is intentionally explicit.
    rng = np.random.default_rng(741)
    x = rng.normal(0, 2.0, 131072)
    f, pxx = welch_psd(x, fs=10_000.0, nperseg=4096)
    recovered = float(np.sum(0.5 * (pxx[:-1] + pxx[1:]) * np.diff(f)))
    assert recovered == pytest.approx(np.var(x), rel=0.08)


def test_fluctuation_flux_covariance_includes_common_finite_samples_only():
    n = np.array([1., 2., 1000., 4.])
    v = np.array([4., 3., -1000., 1.])
    t = np.array([10., 20., np.nan, 40.])
    r = fluctuation_flux(n, v, t)
    mask = np.isfinite(n) & np.isfinite(v) & np.isfinite(t)
    expected = np.mean((n[mask]-n[mask].mean())*(v[mask]-v[mask].mean()))
    assert r["particle_flux_covariance_m-2_s-1"] == pytest.approx(expected, rel=1e-12)



def test_gaussian_spectral_fit_recovers_known_line_center_and_width():
    # Independent synthetic spectrum with exact Gaussian parameters and flat baseline.
    center_nm, sigma_nm, amplitude, baseline = 500.0, 0.025, 120.0, 7.0
    wavelength = np.linspace(499.85, 500.15, 601)
    intensity = baseline + amplitude*np.exp(-0.5*((wavelength-center_nm)/sigma_nm)**2)
    fit = __import__("tis.core", fromlist=["fit_spectral_line"]).fit_spectral_line(
        wavelength, intensity, model="Gaussian"
    )
    assert fit["center"] == pytest.approx(center_nm, abs=1e-8)
    assert fit["sigma"] == pytest.approx(sigma_nm, rel=1e-6)
    assert fit["FWHM"] == pytest.approx(2*np.sqrt(2*np.log(2))*sigma_nm, rel=1e-6)


def test_langmuir_electron_retarding_branch_recovers_known_temperature():
    # Idealized Maxwellian electron-retarding branch: I_e ∝ exp(V/Te), with Te in eV.
    from tis.core import langmuir_iv_analysis
    te_eV = 20.0
    voltage = np.linspace(-20.0, 30.0, 501)
    current = np.full_like(voltage, -1e-3)
    electron_branch = voltage >= 0
    current[electron_branch] += 2e-5*np.exp(voltage[electron_branch]/te_eV)
    result = langmuir_iv_analysis(
        voltage, current,
        ion_fit_range=(-20.0, -2.0),
        electron_fit_range=(0.0, 30.0),
    )
    assert result["electron_temperature_eV"] == pytest.approx(te_eV, rel=1e-6)
    assert result["ion_saturation_current_A"] == pytest.approx(-1e-3, abs=1e-14)


def test_ir_t1_published_doppler_widths_reproduce_reported_ion_temperatures():
    """Reproduce Tables VI–VII in Alipour et al., AIP Advances 7, 115303 (2017).

    DOI: https://doi.org/10.1063/1.4990869
    The paper reports measured and resolution FWHM values, their quadrature-subtracted
    Doppler FWHM, and ion temperatures. The raw CCD spectra are not publicly included;
    this is an independent reproduction of published tabulated values, not a refit
    of the original spectra.
    """
    from tis.core import doppler_temperature_eV

    # Species, lambda0 [nm], mass [amu], measured FWHM [nm],
    # resolution FWHM [nm], published Doppler FWHM [nm], published Ti [eV].
    rows = [
        ("N III", 409.731, 14.0, 0.0637, 0.04003, 0.04955, 34.44),
        ("O V", 412.390, 16.0, 0.0838, 0.04005, 0.07361, 85.75),
        ("Cr I", 425.434, 52.0, 0.0413, 0.04015, 0.00968, 4.53),
        ("C II", 426.700, 12.0, 0.0645, 0.04016, 0.05047, 28.24),
        ("Mo I", 427.726, 96.0, 0.0408, 0.04017, 0.00714, 4.50),
        ("O II", 441.490, 16.0, 0.0631, 0.04027, 0.04858, 32.59),
        ("Ar II", 443.018, 40.0, 0.0483, 0.04028, 0.02665, 24.35),
        ("N II", 444.703, 14.0, 0.0644, 0.04029, 0.05024, 30.06),
        ("Ni I", 446.264, 58.7, 0.0419, 0.04031, 0.01144, 6.48),
        ("Ti I", 453.323, 47.9, 0.0415, 0.04036, 0.00966, 3.66),
        ("C III", 464.742, 12.0, 0.0789, 0.04044, 0.06775, 42.90),
        ("Mn I", 476.237, 55.0, 0.0423, 0.04053, 0.01211, 5.98),
        ("O IV", 479.825, 16.0, 0.0861, 0.04058, 0.07594, 67.41),
        ("W I", 484.382, 183.8, 0.0410, 0.04059, 0.00578, 4.41),
        ("Fe I", 537.149, 55.8, 0.0426, 0.04098, 0.01164, 4.40),
    ]
    fwhm_to_sigma = 2.0 * np.sqrt(2.0 * np.log(2.0))
    for species, lambda0, mass, measured, resolution, published_doppler, published_ti in rows:
        reconstructed_doppler = np.sqrt(measured**2 - resolution**2)
        assert reconstructed_doppler == pytest.approx(published_doppler, abs=2.0e-5), species
        calculated_ti = doppler_temperature_eV(
            published_doppler / fwhm_to_sigma, lambda0, mass
        )
        # Table values are rounded; tolerance is fixed at 0.5% before comparison.
        assert calculated_ti == pytest.approx(published_ti, rel=0.005), species


def test_ir_t1_published_probe_flow_is_consistent_with_reported_mach_number():
    """Cross-check the paper's Te≈14 eV, M_parallel≈0.5 and flow≈17 km/s.

    DOI: https://doi.org/10.1063/1.4994037. This is a reported-value consistency
    check, not a validation of I–V fitting because the original sweep data are unavailable.
    """
    from tis.core import derive_parameters

    result = derive_parameters(
        ne_m3=1.0e19, te_eV=14.0, ti_eV=0.0, ip_A=1.0e4,
        vloop_V=1.0, major_radius_m=0.45, minor_radius_m=0.125,
        bt_T=0.8, ion_mass_amu=1.0,
    )
    velocity_from_reported_mach = 0.5 * result["ion_sound_speed_approx_m_s"]
    assert velocity_from_reported_mach == pytest.approx(18.3e3, rel=0.02)
    # Published mean flow is about 17 km/s; allow 10% for rounded mean Mach/parameters.
    assert velocity_from_reported_mach == pytest.approx(17.0e3, rel=0.10)

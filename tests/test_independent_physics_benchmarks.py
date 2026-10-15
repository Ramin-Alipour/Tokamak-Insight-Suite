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

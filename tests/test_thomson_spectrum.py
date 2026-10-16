import numpy as np
import pytest

from tokamak_analysis.thomson_spectrum import (
    ELECTRON_REST_ENERGY_EV,
    fit_thomson_spectrum,
)


def _synthetic_spectrum(temperature_ev=100.0, instrument_sigma_nm=0.05):
    laser_nm = 532.0
    angle = np.pi / 2
    sigma_physical = laser_nm * np.sqrt(
        2.0 * temperature_ev / ELECTRON_REST_ENERGY_EV
    ) * np.sin(angle / 2)
    sigma_measured = np.sqrt(sigma_physical**2 + instrument_sigma_nm**2)
    wavelength = np.linspace(laser_nm - 30.0, laser_nm + 30.0, 801)
    signal = (
        1200.0 * np.exp(-0.5 * ((wavelength - laser_nm) / sigma_measured) ** 2)
        + 30.0
        + 0.4 * (wavelength - laser_nm)
    )
    rng = np.random.default_rng(20261016)
    signal = signal + rng.normal(0.0, 0.8, wavelength.size)
    return wavelength, signal, laser_nm, angle, sigma_physical


def test_raw_thomson_spectrum_recovers_temperature_with_instrument_deconvolution():
    wavelength, signal, laser_nm, angle, sigma_physical = _synthetic_spectrum()
    out = fit_thomson_spectrum(
        wavelength,
        signal,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
        signal_uncertainty=np.full(signal.size, 0.8),
    )
    assert out["electron_temperature_eV"] == pytest.approx(100.0, rel=0.03)
    assert out["physical_sigma_nm"] == pytest.approx(sigma_physical, rel=0.02)
    assert out["fitted_spectrum"].shape == signal.shape
    assert out["residual"].shape == signal.shape
    assert out["degrees_of_freedom"] == signal.size - 5
    assert np.isfinite(out["electron_temperature_uncertainty_eV"])
    assert "Does not infer electron density" in out["limitations"][0]


def test_raw_thomson_spectrum_rejects_nonmonotonic_wavelength():
    with pytest.raises(ValueError, match="strictly increasing"):
        fit_thomson_spectrum(
            [531, 532, 531.5, 533, 534, 535, 536],
            [1, 2, 4, 8, 4, 2, 1],
            laser_wavelength_nm=532,
            scattering_angle_rad=np.pi / 2,
            instrument_sigma_nm=0.05,
        )


def test_raw_thomson_spectrum_rejects_unresolved_instrument_width():
    wavelength = np.linspace(531.8, 532.2, 51)
    signal = 100 * np.exp(-0.5 * ((wavelength - 532.0) / 0.01) ** 2) + 2
    with pytest.raises(ValueError, match="not larger than the instrumental width"):
        fit_thomson_spectrum(
            wavelength,
            signal,
            laser_wavelength_nm=532,
            scattering_angle_rad=np.pi / 2,
            instrument_sigma_nm=0.05,
        )


def test_raw_thomson_spectrum_rejects_invalid_uncertainty():
    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    uncertainty = np.ones_like(signal)
    uncertainty[3] = 0
    with pytest.raises(ValueError, match="strictly positive"):
        fit_thomson_spectrum(
            wavelength,
            signal,
            laser_wavelength_nm=laser_nm,
            scattering_angle_rad=angle,
            instrument_sigma_nm=0.05,
            signal_uncertainty=uncertainty,
        )


def test_raw_thomson_spectrum_does_not_claim_chi_square_without_uncertainties():
    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    out = fit_thomson_spectrum(
        wavelength,
        signal,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
    )
    assert out["chi_square"] is None
    assert out["reduced_chi_square"] is None
    assert out["residual_variance"] >= 0

import numpy as np
import pytest

from tokamak_analysis.thomson_spectrum import (
    ELECTRON_REST_ENERGY_EV,
    fit_thomson_spectrum,
)


def _synthetic_spectrum(temperature_ev=100.0, instrument_sigma_nm=0.05):
    laser_nm = 532.0
    angle = np.pi / 2
    sigma_physical = 2.0 * laser_nm * np.sqrt(
        temperature_ev / ELECTRON_REST_ENERGY_EV
    ) * np.sin(angle / 2)
    sigma_measured = np.sqrt(sigma_physical**2 + instrument_sigma_nm**2)
    wavelength = np.linspace(laser_nm - 40.0, laser_nm + 40.0, 1001)
    signal = (
        1200.0 * np.exp(-0.5 * ((wavelength - laser_nm) / sigma_measured) ** 2)
        + 30.0
        + 0.4 * (wavelength - laser_nm)
    )
    rng = np.random.default_rng(20261016)
    signal = signal + rng.normal(0.0, 0.8, wavelength.size)
    return wavelength, signal, laser_nm, angle, sigma_physical


def _measured_response(wavelength, sigma_nm=0.05):
    x = np.asarray(wavelength, dtype=float)
    delta = x[:, None] - x[None, :]
    response = np.exp(-0.5 * (delta / sigma_nm) ** 2)
    return response / response.sum(axis=0, keepdims=True)


def test_raw_thomson_spectrum_recovers_temperature_with_instrument_deconvolution():
    wavelength, signal, laser_nm, angle, sigma_physical = _synthetic_spectrum()
    out = fit_thomson_spectrum(
        wavelength,
        signal,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
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
            instrument_response_matrix=np.eye(7),
        )


def test_raw_thomson_spectrum_rejects_unresolved_instrument_width():
    wavelength = np.linspace(531.8, 532.2, 51)
    signal = 100 * np.exp(-0.5 * ((wavelength - 532.0) / 0.01) ** 2) + 2
    with pytest.raises(ValueError, match="not resolved relative to its fit uncertainty"):
        fit_thomson_spectrum(
            wavelength,
            signal,
            laser_wavelength_nm=532,
            scattering_angle_rad=np.pi / 2,
            instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
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
            instrument_response_matrix=_measured_response(wavelength),
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
            instrument_response_matrix=_measured_response(wavelength),
    )
    assert out["chi_square"] is None
    assert out["reduced_chi_square"] is None
    assert out["residual_variance"] >= 0



def test_calibrated_thomson_spectral_power_integrates_fitted_gaussian_area():
    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    calibrated_signal = signal * 1.0e-18
    out = fit_thomson_spectrum(
        wavelength,
        calibrated_signal,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
        signal_uncertainty=np.full(signal.size, 0.8e-18),
        spectral_power_density_calibrated=True,
    )
    expected_power = out["amplitude"] * out["fitted_sigma_nm"] * np.sqrt(2.0 * np.pi)
    assert out["integrated_scattered_power_W"] == pytest.approx(expected_power, rel=1e-12)
    assert out["integrated_scattered_power_uncertainty_W"] > 0
    assert out["spectral_power_density_calibrated"] is True
    assert out["electron_temperature_eV"] == pytest.approx(100.0, rel=0.03)



def test_thomson_temperature_fit_is_stable_under_signal_unit_scaling():
    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    counts_fit = fit_thomson_spectrum(
        wavelength, signal,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
    )
    power_density_fit = fit_thomson_spectrum(
        wavelength, signal * 1.0e-18,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
        signal_uncertainty=np.full(signal.size, 0.8e-18),
        spectral_power_density_calibrated=True,
    )
    assert power_density_fit["electron_temperature_eV"] == pytest.approx(
        counts_fit["electron_temperature_eV"], rel=1e-8
    )



def test_calibrated_thomson_spectrum_to_temperature_and_density_pipeline():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    calibrated_signal = signal * 1.0e-18
    fit = fit_thomson_spectrum(
        wavelength,
        calibrated_signal,
        laser_wavelength_nm=laser_nm,
        scattering_angle_rad=angle,
        instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
        signal_uncertainty=np.full(signal.size, 0.8e-18),
        spectral_power_density_calibrated=True,
    )
    density = density_from_thomson_scattered_power(
        fit["integrated_scattered_power_W"],
        incident_power_W=2.0,
        interaction_length_m=0.15,
        scattering_angle_rad=angle,
        collection_solid_angle_sr=1.0e-3,
        optical_efficiency=0.7,
        scattered_power_uncertainty_W=fit["integrated_scattered_power_uncertainty_W"],
        incident_power_uncertainty_W=0.02,
        interaction_length_uncertainty_m=0.001,
        scattering_angle_uncertainty_rad=np.deg2rad(0.5),
        collection_solid_angle_uncertainty_sr=2.0e-5,
        optical_efficiency_uncertainty=0.01,
    )
    assert fit["electron_temperature_eV"] == pytest.approx(100.0, rel=0.03)
    density_value = float(np.asarray(density["electron_density_m3"]).item())
    density_uncertainty = float(np.asarray(density["electron_density_uncertainty_m3"]).item())
    assert 1.0e19 < density_value < 1.0e20
    assert 0 < density_uncertainty < density_value



def test_raw_thomson_spectrum_rejects_insufficient_wavelength_window():
    wavelength = np.linspace(531.8, 532.2, 101)
    signal = 100.0 * np.exp(-0.5 * ((wavelength - 532.0) / 0.3) ** 2) + 2.0
    with pytest.raises(ValueError, match="Wavelength window must cover"):
        fit_thomson_spectrum(
            wavelength,
            signal,
            laser_wavelength_nm=532.0,
            scattering_angle_rad=np.pi / 2,
            instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
        )



def test_raw_thomson_spectrum_requires_boolean_power_calibration_flag():
    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    with pytest.raises(ValueError, match="must be boolean"):
        fit_thomson_spectrum(
            wavelength,
            signal,
            laser_wavelength_nm=laser_nm,
            scattering_angle_rad=angle,
            instrument_sigma_nm=0.05,
            instrument_response_matrix=_measured_response(wavelength),
            spectral_power_density_calibrated="yes",
        )

def test_thomson_density_propagates_scattering_angle_uncertainty():
    from tokamak_analysis.thomson_scattering import density_from_thomson_scattered_power

    angle = np.pi / 3.0
    angle_sigma = 2.0e-3
    result = density_from_thomson_scattered_power(
        scattered_power_W=1.0e-10,
        incident_power_W=2.0,
        interaction_length_m=0.15,
        scattering_angle_rad=angle,
        collection_solid_angle_sr=1.0e-3,
        optical_efficiency=0.7,
        scattering_angle_uncertainty_rad=angle_sigma,
    )
    density = float(np.asarray(result["electron_density_m3"]).item())
    actual_sigma = float(np.asarray(result["electron_density_uncertainty_m3"]).item())
    relative_sensitivity = 2.0 * np.cos(angle) * np.sin(angle) / (1.0 + np.cos(angle) ** 2)
    expected_sigma = density * abs(relative_sensitivity) * angle_sigma
    assert actual_sigma == pytest.approx(expected_sigma, rel=1e-12)
    assert actual_sigma > 0


def test_legacy_adapter_rejects_synthetic_response_from_scalar_width():
    wavelength, signal, laser_nm, angle, _ = _synthetic_spectrum()
    with pytest.raises(ValueError, match="measured instrument_response_matrix is required"):
        fit_thomson_spectrum(
            wavelength, signal, laser_wavelength_nm=laser_nm,
            scattering_angle_rad=angle, instrument_sigma_nm=0.05,
        )

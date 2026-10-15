import numpy as np
import pytest

from tis.thomson import analyze_thomson_spectrum


def test_thomson_spectrum_recovers_temperature_and_optional_density():
    laser_nm = 1064.0
    theta_deg = 90.0
    expected_te_ev = 250.0
    sigma_nm = laser_nm * np.sqrt(2 * expected_te_ev / 510998.95069) * np.sin(np.deg2rad(theta_deg) / 2)
    wavelength = np.linspace(laser_nm - 8 * sigma_nm, laser_nm + 8 * sigma_nm, 501)
    signal = 12.0 + 0.03 * (wavelength - laser_nm) + 100.0 * np.exp(
        -0.5 * ((wavelength - laser_nm) / sigma_nm) ** 2
    )
    result = analyze_thomson_spectrum(
        wavelength, signal, laser_wavelength_nm=laser_nm,
        scattering_angle_deg=theta_deg, density_per_integrated_signal=2.5e17,
    )
    assert result["electron_temperature_eV"] == pytest.approx(expected_te_ev, rel=1e-3)
    assert result["electron_density_m3"] == pytest.approx(
        result["fit_parameters"]["integrated_signal_signal_nm"] * 2.5e17
    )
    assert result["fit_quality"]["r_squared"] > 0.99999


@pytest.mark.parametrize(
    "kwargs",
    [
        {"laser_wavelength_nm": 0, "scattering_angle_deg": 90},
        {"laser_wavelength_nm": 1064, "scattering_angle_deg": 0},
        {"laser_wavelength_nm": 1064, "scattering_angle_deg": 181},
    ],
)
def test_thomson_spectrum_rejects_invalid_geometry(kwargs):
    x = np.linspace(1050, 1078, 20)
    y = np.exp(-((x - 1064) / 3) ** 2)
    with pytest.raises(ValueError):
        analyze_thomson_spectrum(x, y, **kwargs)


def test_thomson_spectrum_requires_monotonic_finite_axis():
    with pytest.raises(ValueError, match="strictly increasing"):
        analyze_thomson_spectrum(
            [1, 2, 2, 4, 5, 6, 7, 8], np.arange(8),
            laser_wavelength_nm=4, scattering_angle_deg=90,
        )

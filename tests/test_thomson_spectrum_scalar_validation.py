"""Regression tests for Thomson-spectrum scalar input validation."""
import numpy as np
import pytest

from tokamak_analysis.thomson_spectrum import fit_thomson_spectrum


@pytest.mark.parametrize(
    "parameter,bad_value",
    [
        ("laser_wavelength_nm", True),
        ("laser_wavelength_nm", np.bool_(True)),
        ("laser_wavelength_nm", "1064"),
        ("laser_wavelength_nm", np.nan),
        ("laser_wavelength_nm", np.inf),
        ("scattering_angle_rad", True),
        ("scattering_angle_rad", np.bool_(True)),
        ("scattering_angle_rad", "1.2"),
        ("scattering_angle_rad", np.nan),
        ("scattering_angle_rad", np.inf),
        ("instrument_sigma_nm", True),
        ("instrument_sigma_nm", np.bool_(True)),
        ("instrument_sigma_nm", "0.02"),
        ("instrument_sigma_nm", np.nan),
        ("instrument_sigma_nm", np.inf),
    ],
)
def test_thomson_spectrum_rejects_invalid_scalar_physics_parameters(parameter, bad_value):
    wavelength = np.linspace(1058.0, 1070.0, 121)
    signal = 100.0 * np.exp(-0.5 * ((wavelength - 1064.0) / 0.8) ** 2) + 2.0
    kwargs = {
        "laser_wavelength_nm": 1064.0,
        "scattering_angle_rad": 1.2,
        "instrument_sigma_nm": 0.1,
    }
    kwargs[parameter] = bad_value
    with pytest.raises(ValueError):
        fit_thomson_spectrum(wavelength, signal, **kwargs)

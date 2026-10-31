"""Input-contract tests for measured Thomson spectral data."""

import numpy as np
import pytest

@pytest.mark.parametrize(
    ("wavelength", "signal"),
    [
        ([False, True, False, True, False, True, False], np.arange(7.0)),
        (np.arange(7.0), ["0", "1", "2", "3", "4", "5", "6"]),
    ],
)
def test_thomson_spectrum_rejects_boolean_or_string_vectors(wavelength, signal):
    from tokamak_analysis.thomson_spectrum import fit_thomson_spectrum

    with pytest.raises(ValueError, match="1-D array|real numeric"):
        fit_thomson_spectrum(
            wavelength, signal, laser_wavelength_nm=1064.0,
            scattering_angle_rad=1.0, instrument_sigma_nm=0.1,
        )


def test_thomson_spectrum_rejects_boolean_uncertainties():
    from tokamak_analysis.thomson_spectrum import fit_thomson_spectrum

    wavelength = np.linspace(1050.0, 1078.0, 9)
    signal = np.exp(-0.5 * ((wavelength - 1064.0) / 3.0) ** 2)
    with pytest.raises(ValueError, match="real numeric"):
        fit_thomson_spectrum(
            wavelength, signal, laser_wavelength_nm=1064.0,
            scattering_angle_rad=1.0, instrument_sigma_nm=0.1,
            signal_uncertainty=[True] * 9,
        )


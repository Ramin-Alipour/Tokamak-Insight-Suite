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


@pytest.mark.parametrize("invalid_value", ["false", 0, 1, None])
def test_primary_thomson_rejects_non_boolean_evdf_switch(invalid_value):
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra

    with pytest.raises(ValueError, match="enable_evdf_reconstruction must be boolean"):
        analyze_thomson_spectra(
            np.linspace(1050.0, 1078.0, 9), np.ones((1, 9)), [0.5],
            laser_wavelength_nm=1064.0, scattering_angle_rad=1.0,
            instrument_response=np.eye(9), model="non_collective_maxwellian",
            enable_evdf_reconstruction=invalid_value,
        )


@pytest.mark.parametrize("invalid_points", [True, 40, 2002, 121.0, "121"])
def test_primary_thomson_rejects_evdf_grid_coercion(invalid_points):
    from tokamak_analysis.thomson_inference import analyze_thomson_spectra

    with pytest.raises(ValueError, match="evdf_velocity_points must be an integer from 41 to 2001"):
        analyze_thomson_spectra(
            np.linspace(1050.0, 1078.0, 9), np.ones((1, 9)), [0.5],
            laser_wavelength_nm=1064.0, scattering_angle_rad=1.0,
            instrument_response=np.eye(9), model="non_collective_maxwellian",
            evdf_velocity_points=invalid_points,
        )


def test_analysis1_does_not_expose_raw_count_poisson_input():
    import tokamak_analysis.thomson_inference as inference

    assert not hasattr(inference, "fit_optical_thomson_poisson_counts")

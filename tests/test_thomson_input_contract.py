import numpy as np
import pytest

from tokamak_analysis.thomson_scattering import build_profile


@pytest.mark.parametrize(
    ("radius", "density", "temperature"),
    [
        ([False, True], [1.0, 2.0], [3.0, 4.0]),
        ([0.0, 1.0], ["1", "2"], [3.0, 4.0]),
        ([0.0, 1.0], [1.0, 2.0], [True, False]),
        ([[0.0], [1.0]], [1.0, 2.0], [3.0, 4.0]),
    ],
)
def test_thomson_profile_rejects_non_numeric_or_non_1d_measurements(
    radius, density, temperature
):
    with pytest.raises(ValueError, match="one-dimensional|real numeric"):
        build_profile(radius, density, temperature)


@pytest.mark.parametrize(
    "uncertainty",
    [
        [True, False],
        ["0.1", "0.2"],
        [[0.1], [0.2]],
    ],
)
def test_thomson_profile_rejects_invalid_uncertainty_types_and_shapes(uncertainty):
    with pytest.raises(ValueError, match="one-dimensional|real numeric|same length"):
        build_profile([0.0, 1.0], [1e19, 2e19], [10.0, 20.0],
                      density_uncertainty_m3=uncertainty)


def test_thomson_profile_accepts_valid_numeric_measurements_and_uncertainties():
    profile = build_profile(
        [1.0, 0.0], [2e19, 1e19], [20.0, 10.0],
        density_uncertainty_m3=[2e18, 1e18],
    )
    np.testing.assert_allclose(profile.radius_m, [0.0, 1.0])
    np.testing.assert_allclose(profile.density_m3, [1e19, 2e19])
    np.testing.assert_allclose(profile.density_uncertainty_m3, [1e18, 2e18])

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


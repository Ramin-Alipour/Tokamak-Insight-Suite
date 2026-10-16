import numpy as np
import pytest

from tis.cxrs import analyze_cxrs_line


def test_cxrs_recovers_line_of_sight_velocity_and_temperature():
    wavelength0 = 529.1
    velocity = 25_000.0
    center = wavelength0 * (1 + velocity / 299792458.0)
    te = 80.0
    mass_amu = 12.0
    from tis.core import doppler_temperature_from_fwhm_eV
    # Generate an observed Gaussian FWHM from a known thermal temperature.
    thermal_fwhm = np.sqrt(
        8 * np.log(2) * wavelength0**2 * te * 1.602176634e-19
        / (mass_amu * 1.66053906660e-27 * 299792458.0**2)
    )
    wavelength = np.linspace(center - 0.3, center + 0.3, 401)
    intensity = 2 + 100 * np.exp(-0.5 * ((wavelength - center) / (thermal_fwhm / 2.354820045)) ** 2)
    result = analyze_cxrs_line(
        wavelength, intensity, rest_wavelength_nm=wavelength0,
        ion_mass_amu=mass_amu, viewing_projection=0.5,
    )
    assert result["line_of_sight_velocity_m_s"] == pytest.approx(velocity, abs=100)
    assert result["ion_temperature"]["temperature_eV"] == pytest.approx(te, rel=0.03)
    assert result["projected_flow_component_m_s"] == pytest.approx(2 * velocity, abs=200)


def test_cxrs_rejects_unresolved_instrumental_line():
    wavelength = np.linspace(500, 501, 101)
    intensity = 1 + 10 * np.exp(-0.5 * ((wavelength - 500.5) / 0.02) ** 2)
    with pytest.raises(ValueError, match="unresolved"):
        analyze_cxrs_line(
            wavelength, intensity, rest_wavelength_nm=500.5,
            ion_mass_amu=12, instrumental_fwhm_nm=0.2,
        )


def test_cxrs_rejects_invalid_viewing_projection():
    with pytest.raises(ValueError, match="viewing_projection"):
        analyze_cxrs_line(
            np.linspace(500, 501, 101), 1 + np.exp(-((np.linspace(500, 501, 101)-500.5)/.1)**2),
            rest_wavelength_nm=500.5, ion_mass_amu=12, viewing_projection=0,
        )

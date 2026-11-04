import numpy as np
import pytest

from tokamak_analysis.cts_em_background import (
    integrate_thermal_microwave_background,
    thermal_bremsstrahlung_emissivity_si,
)


def _vacuum_case(**overrides):
    case = {
        "frequency_Hz": np.array([90e9, 100e9, 110e9]),
        "path_length_m": np.array([0.1, 0.2]),
        "electron_temperature_eV": np.array([100.0, 150.0]),
        "electron_density_m3": np.zeros(2),
        "ion_densities_m3": np.zeros((2, 1)),
        "ion_charge_states": np.array([1.0]),
    }
    case.update(overrides)
    return case


def test_zero_density_free_free_is_exactly_zero():
    result = thermal_bremsstrahlung_emissivity_si(
        np.array([90e9, 100e9]), 0.0, np.array([0.0]), np.array([1.0]), 100.0
    )
    np.testing.assert_array_equal(result, np.zeros(2))


def test_zero_plasma_and_zero_incident_radiance_has_explicit_zero_background_status():
    result = integrate_thermal_microwave_background(**_vacuum_case())
    np.testing.assert_array_equal(
        result["spectral_radiance_W_m2_sr_Hz"], np.zeros(3)
    )
    assert result["background_status"] == "zero_background_exact_zero_radiance"
    assert result["zero_background"] is True
    assert result["ece_model_status"].startswith("ECE omitted")


def test_zero_plasma_with_incident_radiance_is_not_misreported_as_zero_background():
    incident = np.array([1e-18, 2e-18, 3e-18])
    result = integrate_thermal_microwave_background(
        **_vacuum_case(incident_radiance_W_m2_sr_Hz=incident)
    )
    np.testing.assert_allclose(
        result["spectral_radiance_W_m2_sr_Hz"], incident, rtol=0.0, atol=0.0
    )
    assert result["background_status"] == "incident_radiance_only_no_plasma_background"
    assert result["zero_background"] is False


def test_zero_electron_density_rejects_nonzero_positive_ion_density():
    with pytest.raises(ValueError, match="zero electron density requires zero positive-ion densities"):
        integrate_thermal_microwave_background(
            **_vacuum_case(ion_densities_m3=np.array([[1e18], [0.0]]))
        )


def test_zero_density_path_rejects_nonzero_ece_absorption():
    alpha = np.zeros((2, 3))
    alpha[0, 1] = 1e-4
    with pytest.raises(ValueError, match="ECE absorption must be zero"):
        integrate_thermal_microwave_background(
            **_vacuum_case(ece_absorption_coefficient_m_inverse=alpha)
        )


def test_free_free_emissivity_uses_per_steradian_si_normalization():
    # Rybicki-Lightman thermal free-free coefficient: 6.8e-38 in
    # erg s^-1 cm^-3 Hz^-1, integrated over 4*pi sr.  Converting to
    # W m^-3 and dividing by 4*pi gives 5.41e-40 per steradian.
    frequency = np.array([80e9, 120e9])
    ne = 2.0e19
    ni = np.array([1.0e19, 2.0e18])
    charge = np.array([1.0, 2.0])
    temperature_eV = 2.0e3
    gaunt = np.array([1.2, 1.4])
    temperature_K = temperature_eV * 1.602176634e-19 / 1.380649e-23
    expected = (
        5.44e-40 * ne * np.sum(charge**2 * ni)
        * temperature_K**-0.5
        * np.exp(-6.62607015e-34 * frequency / (1.380649e-23 * temperature_K))
        * gaunt
    )
    actual = thermal_bremsstrahlung_emissivity_si(
        frequency, ne, ni, charge, temperature_eV, gaunt_factor=gaunt
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)
    assert np.all(actual > 1e11 * 5.44e-52 * ne * np.sum(charge**2 * ni) * temperature_K**-0.5 * 1e-12)


def test_free_free_optical_depth_uses_corrected_emissivity_via_kirchhoff():
    case = _vacuum_case(
        electron_density_m3=np.array([1.0e19, 1.0e19]),
        ion_densities_m3=np.array([[1.0e19], [1.0e19]]),
        electron_temperature_eV=np.array([2.0e3, 2.0e3]),
        path_length_m=np.array([0.1, 0.2]),
    )
    result = integrate_thermal_microwave_background(**case)
    assert np.all(result["thermal_bremsstrahlung_optical_depth"] > 0.0)
    assert np.all(result["spectral_radiance_W_m2_sr_Hz"] > 0.0)
    assert result["background_status"] == "plasma_emission_or_absorption_present"

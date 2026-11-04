import numpy as np
import pytest

from tokamak_analysis.cts_em_background import (
    integrate_thermal_microwave_background,
    thermal_bremsstrahlung_emissivity_si,
    thermal_bremsstrahlung_absorption_si,
    ece_lte_emissivity_si,
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
        (6.8e-38 * 0.1 / (4.0 * np.pi)) * ne * np.sum(charge**2 * ni)
        * temperature_K**-0.5
        * np.exp(-6.62607015e-34 * frequency / (1.380649e-23 * temperature_K))
        * gaunt
    )
    actual = thermal_bremsstrahlung_emissivity_si(
        frequency, ne, ni, charge, temperature_eV, gaunt_factor=gaunt
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-15, atol=0.0)
    assert np.all(np.isfinite(actual))


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



@pytest.mark.parametrize(
    "arguments",
    [
        (True, np.array([1.0]), np.array([1.0]), 100.0),
        (1.0e19, np.array([1.0]), np.array([1.5]), 100.0),
        (1.0e19, np.array([True]), np.array([1.0]), 100.0),
        (1.0e19, np.array([1.0]), np.array([True]), 100.0),
    ],
)
def test_free_free_rejects_boolean_or_non_integer_physical_inputs(arguments):
    ne, ni, charge, temperature = arguments
    with pytest.raises(ValueError):
        thermal_bremsstrahlung_emissivity_si(
            np.array([100e9]), ne, ni, charge, temperature
        )


def test_free_free_rejects_string_scalar_and_boolean_gaunt_factor():
    with pytest.raises(ValueError, match="electron_temperature_eV must be a real numeric scalar"):
        thermal_bremsstrahlung_emissivity_si(
            np.array([100e9]), 1e19, np.array([1e19]), np.array([1.0]), "1000"
        )
    with pytest.raises(ValueError, match="gaunt_factor must contain real numeric values"):
        thermal_bremsstrahlung_emissivity_si(
            np.array([100e9]), 1e19, np.array([1e19]), np.array([1.0]), 1000.0,
            gaunt_factor=True,
        )


def test_background_transfer_rejects_boolean_physical_arrays():
    case = _vacuum_case(electron_density_m3=np.array([False, False]))
    with pytest.raises(ValueError, match="electron_density_m3 must contain real numeric values"):
        integrate_thermal_microwave_background(**case)



def test_free_free_absorption_obeys_kirchhoff_with_exact_planck_source():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance

    frequency = np.array([20e9, 100e9, 400e9])
    temperature_eV = 2500.0
    emissivity = thermal_bremsstrahlung_emissivity_si(
        frequency, 1.2e19, np.array([1.2e19]), np.array([1.0]),
        temperature_eV, gaunt_factor=np.array([1.1, 1.2, 1.3]),
    )
    alpha = thermal_bremsstrahlung_absorption_si(
        frequency, 1.2e19, np.array([1.2e19]), np.array([1.0]),
        temperature_eV, gaunt_factor=np.array([1.1, 1.2, 1.3]),
    )
    source = planck_spectral_radiance(
        frequency, temperature_eV * 1.602176634e-19 / 1.380649e-23
    )
    np.testing.assert_allclose(alpha * source, emissivity, rtol=2e-15, atol=0.0)
    assert np.all(alpha > 0.0)


def test_mode_resolved_lte_ece_emissivity_is_kirchhoff_consistent():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance

    frequency = np.array([80e9, 100e9, 120e9])
    alpha = np.array([[1e-4, 2e-4], [3e-4, 4e-4], [5e-4, 6e-4]])
    temperature_eV = 1800.0
    emissivity = ece_lte_emissivity_si(frequency, alpha, temperature_eV)
    source = planck_spectral_radiance(
        frequency, temperature_eV * 1.602176634e-19 / 1.380649e-23
    )
    assert emissivity.shape == alpha.shape
    np.testing.assert_allclose(emissivity, alpha * source[:, None], rtol=2e-15, atol=0.0)


@pytest.mark.parametrize(
    "alpha",
    [np.array([-1e-4, 1e-4]), np.ones((2, 2, 2)), np.array([1e-4])],
)
def test_lte_ece_emissivity_rejects_unphysical_or_mismatched_absorption(alpha):
    with pytest.raises(ValueError):
        ece_lte_emissivity_si(np.array([90e9, 100e9]), alpha, 1000.0)



def test_uniform_lte_ece_slab_matches_formal_solution_of_radiative_transfer():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance

    frequency = np.array([90e9, 100e9, 110e9])
    temperature_eV = 2200.0
    alpha = np.array([[0.02, 0.05, 0.08]])
    path = np.array([0.7])
    incident = np.array([1e-18, 2e-18, 3e-18])
    result = integrate_thermal_microwave_background(
        frequency_Hz=frequency,
        path_length_m=path,
        electron_temperature_eV=np.array([temperature_eV]),
        electron_density_m3=np.array([1e19]),
        ion_densities_m3=np.array([[1e19]]),
        ion_charge_states=np.array([1.0]),
        ece_absorption_coefficient_m_inverse=alpha,
        ece_lte=True,
        incident_radiance_W_m2_sr_Hz=incident,
    )
    source = planck_spectral_radiance(
        frequency, temperature_eV * 1.602176634e-19 / 1.380649e-23
    )
    # The integrated output is the exact formal solution I=I0 exp(-tau)+B(1-exp(-tau)).
    # Free-free opacity is also present in this plasma, so compare against the
    # returned total optical depth and the same LTE source function.
    tau = result["total_optical_depth"]
    expected = incident * np.exp(-tau) + source * (1.0 - np.exp(-tau))
    np.testing.assert_allclose(
        result["spectral_radiance_W_m2_sr_Hz"], expected, rtol=2e-12, atol=1e-30
    )


def test_multimode_total_optical_depth_is_exact_for_incoherent_receiver_mixture():
    frequency = np.array([100e9])
    weights = np.array([0.25, 0.75])
    alpha = np.array([[[0.2, 1.1]]])
    result = integrate_thermal_microwave_background(
        frequency_Hz=frequency,
        path_length_m=np.array([1.0]),
        electron_temperature_eV=np.array([1000.0]),
        electron_density_m3=np.array([0.0]),
        ion_densities_m3=np.array([[0.0]]),
        ion_charge_states=np.array([1.0]),
        ece_absorption_coefficient_m_inverse=alpha,
        ece_mode_weights=weights,
        ece_emissivity_W_m3_sr_Hz=np.zeros((1, 1, 2)),
        incident_radiance_W_m2_sr_Hz=np.array([1.0]),
    )
    expected = -np.log(weights @ np.exp(-np.array([0.2, 1.1])))
    assert result["total_optical_depth"][0] == pytest.approx(expected, rel=2e-14)
    assert result["mean_total_optical_depth"][0] == pytest.approx(
        weights @ np.array([0.2, 1.1]), rel=2e-14
    )
    assert result["total_optical_depth"][0] != pytest.approx(
        result["mean_total_optical_depth"][0], rel=1e-3
    )
    np.testing.assert_allclose(
        result["spectral_radiance_W_m2_sr_Hz"], [weights @ np.exp(-np.array([0.2, 1.1]))]
    )

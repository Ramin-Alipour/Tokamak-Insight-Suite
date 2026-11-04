import numpy as np
import pytest

from tokamak_analysis.cts_em_background import (
    integrate_thermal_microwave_background,
    thermal_bremsstrahlung_emissivity_si,
    thermal_bremsstrahlung_absorption_si,
    ece_lte_emissivity_si,
    radiance_to_receiver_spectral_power,
    propagate_receiver_spectral_power_covariance,
    integrate_receiver_spectral_power,
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
    # erg s^-1 cm^-3 Hz^-1, integrated over 4*pi sr, with densities in cm^-3.
    # For SI densities and emissivity, conversion is
    # 1e-7 (J/erg) * 1e6 (cm^3/m^3) * 1e-12 (two density conversions)
    # = 1e-13; divide by 4*pi for per-steradian emissivity.
    frequency = np.array([80e9, 120e9])
    ne = 2.0e19
    ni = np.array([1.0e19, 2.0e18])
    charge = np.array([1.0, 2.0])
    temperature_eV = 2.0e3
    gaunt = np.array([1.2, 1.4])
    temperature_K = temperature_eV * 1.602176634e-19 / 1.380649e-23
    expected = (
        (6.8e-38 * 1.0e-13 / (4.0 * np.pi)) * ne * np.sum(charge**2 * ni)
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
        electron_density_m3=np.array([1.0e-20]),
        ion_densities_m3=np.array([[1.0e-20]]),
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


def test_planck_and_free_free_opacity_remain_finite_in_wien_tail():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance
    frequency = np.array([1.0e22, 2.0e22])
    temperature_eV = 1.0e3
    temperature_K = temperature_eV * 1.602176634e-19 / 1.380649e-23
    planck = planck_spectral_radiance(frequency, temperature_K)
    emissivity = thermal_bremsstrahlung_emissivity_si(
        frequency, 1.0e19, [1.0e19], [1.0], temperature_eV
    )
    opacity = thermal_bremsstrahlung_absorption_si(
        frequency, 1.0e19, [1.0e19], [1.0], temperature_eV
    )
    assert np.all(np.isfinite(planck))
    assert np.all(planck == 0.0)  # representational underflow, not a clipped Wien tail
    assert np.all(emissivity == 0.0)
    assert np.all(np.isfinite(opacity))
    assert np.all(opacity > 0.0)


def test_free_free_absorption_is_algebraically_consistent_with_kirchhoff_at_moderate_x():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance
    frequency = np.array([80e9, 140e9, 220e9])
    temperature_eV = 1.5e3
    temperature_K = temperature_eV * 1.602176634e-19 / 1.380649e-23
    emissivity = thermal_bremsstrahlung_emissivity_si(
        frequency, 2.0e19, [1.0e19, 2.0e18], [1.0, 2.0],
        temperature_eV, gaunt_factor=[1.1, 1.2, 1.3],
    )
    planck = planck_spectral_radiance(frequency, temperature_K)
    opacity = thermal_bremsstrahlung_absorption_si(
        frequency, 2.0e19, [1.0e19, 2.0e18], [1.0, 2.0],
        temperature_eV, gaunt_factor=[1.1, 1.2, 1.3],
    )
    np.testing.assert_allclose(opacity, emissivity / planck, rtol=3e-14, atol=0.0)


def test_charge_state_moments_report_zeff_and_quasineutrality_without_repair():
    from tokamak_analysis.cts_em_background import plasma_charge_state_moments
    result = plasma_charge_state_moments(1.4e19, [1.0e19, 2.0e18], [1.0, 2.0])
    assert result["ion_charge_density_equivalent_m3"] == pytest.approx(1.4e19)
    assert result["ion_charge_squared_density_m3"] == pytest.approx(1.8e19)
    assert result["effective_charge_Zeff"] == pytest.approx(1.8e19 / 1.4e19)
    assert result["net_charge_density_C_m3"] == pytest.approx(0.0, abs=1e-35)
    assert result["quasineutrality_relative_residual"] == pytest.approx(0.0, abs=1e-15)
    assert result["quasineutrality_assumed_or_enforced"] is False


def test_charge_state_moments_report_non_neutrality_instead_of_silently_correcting():
    from tokamak_analysis.cts_em_background import plasma_charge_state_moments
    result = plasma_charge_state_moments(1.0e19, [0.8e19], [1.0])
    assert result["quasineutrality_relative_residual"] == pytest.approx(-0.2)
    assert result["net_charge_density_C_m3"] < 0.0
    assert result["effective_charge_Zeff"] == pytest.approx(0.8)


def test_charge_state_moments_has_undefined_zeff_for_vacuum():
    from tokamak_analysis.cts_em_background import plasma_charge_state_moments
    result = plasma_charge_state_moments(0.0, [0.0], [1.0])
    assert result["effective_charge_Zeff"] is None
    assert result["quasineutrality_relative_residual"] == 0.0



def test_charge_state_moments_are_reachable_through_advanced_physics_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "plasma_charge_state_moments",
        electron_density_m3=1.4e19,
        ion_densities_m3=[1.0e19, 2.0e18],
        ion_charge_states=[1.0, 2.0],
    )
    assert result["public_physics_entrypoint"] == "plasma_charge_state_moments"
    assert result["effective_charge_Zeff"] == pytest.approx(1.8e19 / 1.4e19)



def test_microwave_background_carries_path_resolved_charge_state_moments():
    case = _vacuum_case(
        electron_density_m3=np.array([1.0e19, 1.4e19]),
        ion_densities_m3=np.array([[1.0e19, 0.0], [1.0e19, 2.0e18]]),
        ion_charge_states=np.array([1.0, 2.0]),
    )
    result = integrate_thermal_microwave_background(**case)
    moments = result["plasma_charge_state_moments_by_path"]
    assert len(moments) == 2
    assert moments[0]["effective_charge_Zeff"] == pytest.approx(1.0)
    assert moments[1]["effective_charge_Zeff"] == pytest.approx(1.8e19 / 1.4e19)
    assert moments[1]["quasineutrality_relative_residual"] == pytest.approx(0.0, abs=1e-15)


def test_free_free_si_coefficient_uses_cgs_density_and_volume_conversions():
    # The tabulated cgs coefficient uses n in cm^-3 and emissivity per cm^3.
    # For n in m^-3 and output per m^3: 1e-7 * 1e6 * (1e-6)^2 = 1e-13.
    from tokamak_analysis.cts_em_background import _FREE_FREE_EMISSIVITY_COEFFICIENT_SI

    assert _FREE_FREE_EMISSIVITY_COEFFICIENT_SI == pytest.approx(
        6.8e-51 / (4.0 * np.pi), rel=0.0, abs=0.0
    )


def test_free_free_only_uniform_slab_obeys_planck_kirchhoff_solution():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance

    frequency = np.array([80e9, 100e9, 130e9])
    temperature_eV = 1800.0
    ne = 1.0e20
    ni = np.array([1.0e20])
    path = np.array([0.4])
    incident = np.array([2e-18, 3e-18, 4e-18])
    result = integrate_thermal_microwave_background(
        frequency_Hz=frequency,
        path_length_m=path,
        electron_temperature_eV=np.array([temperature_eV]),
        electron_density_m3=np.array([ne]),
        ion_densities_m3=ni[None, :],
        ion_charge_states=np.array([1.0]),
        incident_radiance_W_m2_sr_Hz=incident,
    )
    temperature_K = temperature_eV * 1.602176634e-19 / 1.380649e-23
    source = planck_spectral_radiance(frequency, temperature_K)
    tau = result["thermal_bremsstrahlung_optical_depth"]
    expected = incident * np.exp(-tau) + source * (-np.expm1(-tau))
    np.testing.assert_allclose(
        result["spectral_radiance_W_m2_sr_Hz"], expected,
        rtol=2e-12, atol=1e-30,
    )



def test_effective_mode_optical_depth_ignores_zero_weight_modes_without_underflow():
    # The zero-weight mode has the smallest optical depth. It must not become
    # the log-sum-exp reference: doing so underflows the only active mode.
    result = integrate_thermal_microwave_background(
        frequency_Hz=np.array([100e9]),
        path_length_m=np.array([1.0]),
        electron_temperature_eV=np.array([1000.0]),
        electron_density_m3=np.array([1.0e-20]),
        ion_densities_m3=np.array([[1.0e-20]]),
        ion_charge_states=np.array([1.0]),
        ece_absorption_coefficient_m_inverse=np.array([[[0.0, 1000.0]]]),
        ece_mode_weights=np.array([0.0, 1.0]),
        ece_emissivity_W_m3_sr_Hz=np.zeros((1, 1, 2)),
        incident_radiance_W_m2_sr_Hz=np.array([1.0]),
    )
    assert np.isfinite(result["total_optical_depth"][0])
    assert result["total_optical_depth"][0] == pytest.approx(1000.0)
    assert result["total_optical_depth_by_mode"][0, 0] == pytest.approx(0.0)
    assert result["total_optical_depth_by_mode"][0, 1] == pytest.approx(1000.0)


def test_mode_weights_reject_boolean_and_non_numeric_values():
    base = dict(
        frequency_Hz=np.array([100e9]),
        path_length_m=np.array([1.0]),
        electron_temperature_eV=np.array([1000.0]),
        electron_density_m3=np.array([1.0e-20]),
        ion_densities_m3=np.array([[1.0e-20]]),
        ion_charge_states=np.array([1.0]),
        ece_absorption_coefficient_m_inverse=np.array([[[0.2, 0.8]]]),
        ece_emissivity_W_m3_sr_Hz=np.zeros((1, 1, 2)),
    )
    with pytest.raises(ValueError, match="ece_mode_weights must contain real numeric values"):
        integrate_thermal_microwave_background(
            **base, ece_mode_weights=np.array([True, False])
        )



def test_zero_electron_density_rejects_nonzero_ece_emissivity():
    with pytest.raises(ValueError, match="ECE emissivity must be zero"):
        integrate_thermal_microwave_background(
            **_vacuum_case(
                ece_absorption_coefficient_m_inverse=np.zeros((2, 3)),
                ece_emissivity_W_m3_sr_Hz=np.ones((2, 3)),
            )
        )



def test_effective_mode_optical_depth_uses_logsumexp_for_extreme_mode_contrast():
    result = integrate_thermal_microwave_background(
        frequency_Hz=np.array([100e9]),
        path_length_m=np.array([1.0]),
        electron_temperature_eV=np.array([1000.0]),
        electron_density_m3=np.array([1.0e-20]),
        ion_densities_m3=np.array([[1.0e-20]]),
        ion_charge_states=np.array([1.0]),
        ece_absorption_coefficient_m_inverse=np.array([[[0.0, 1000.0]]]),
        ece_mode_weights=np.array([1.0e-300, 1.0]),
        ece_emissivity_W_m3_sr_Hz=np.zeros((1, 1, 2)),
        incident_radiance_W_m2_sr_Hz=np.array([1.0]),
    )
    expected = -np.log(1.0e-300 + np.exp(-1000.0))
    assert result["total_optical_depth"][0] == pytest.approx(expected, rel=2e-14)


@pytest.mark.parametrize("bad_gaunt", [np.nan, np.inf, -np.inf])
def test_free_free_absorption_rejects_nonfinite_gaunt_factor(bad_gaunt):
    with pytest.raises(ValueError, match="gaunt_factor must be positive and finite"):
        thermal_bremsstrahlung_absorption_si(
            np.array([100e9]), 1.0e19, np.array([1.0e19]),
            np.array([1.0]), 1000.0, gaunt_factor=bad_gaunt,
        )



def test_receiver_spectral_power_uses_etendue_and_efficiency_once():
    radiance = np.array([2.0e-18, 3.0e-18, 4.0e-18])
    etendue = 2.5e-4
    efficiency = 0.4
    actual = radiance_to_receiver_spectral_power(
        radiance,
        receiver_etendue_m2_sr=etendue,
        end_to_end_efficiency=efficiency,
    )
    np.testing.assert_allclose(actual, radiance * etendue * efficiency)
    assert np.all(actual < radiance * etendue)


def test_receiver_spectral_power_covariance_propagates_cross_correlations():
    radiance = np.array([2.0e-18, 3.0e-18])
    etendue, efficiency = 2.0e-4, 0.5
    standard = np.array([1.0e-20, 1.5e-20, 1.0e-6, 0.01])
    correlation = np.eye(4)
    correlation[0, 1] = correlation[1, 0] = 0.25
    correlation[0, 2] = correlation[2, 0] = 0.1
    covariance = np.outer(standard, standard) * correlation
    result = propagate_receiver_spectral_power_covariance(
        radiance,
        receiver_etendue_m2_sr=etendue,
        end_to_end_efficiency=efficiency,
        input_covariance=covariance,
    )
    jacobian = result["propagation_jacobian"]
    expected = jacobian @ covariance @ jacobian.T
    np.testing.assert_allclose(
        result["receiver_spectral_power_covariance_W2_Hz2"], expected,
        rtol=1e-12, atol=0.0,
    )
    assert result["receiver_spectral_power_uncertainty_W_Hz"].shape == (2,)
    assert result["uncertainty_method"] == "first_order_full_joint_covariance_J_C_JT"


def test_receiver_spectral_power_covariance_rejects_non_psd_joint_input():
    radiance = np.array([2.0e-18, 3.0e-18])
    covariance = np.eye(4)
    covariance[0, 1] = covariance[1, 0] = 2.0
    with pytest.raises(ValueError, match="positive semidefinite"):
        propagate_receiver_spectral_power_covariance(
            radiance, receiver_etendue_m2_sr=1.0e-4,
            end_to_end_efficiency=0.5, input_covariance=covariance,
        )


def test_integrated_receiver_power_uses_nonuniform_frequency_quadrature_and_covariance():
    frequency = np.array([80.0e9, 100.0e9, 150.0e9])
    spectral_power = np.array([1.0e-20, 2.0e-20, 1.0e-20])
    covariance = np.diag(np.array([1.0e-42, 4.0e-42, 1.0e-42]))
    result = integrate_receiver_spectral_power(
        frequency, spectral_power,
        spectral_power_covariance_W2_Hz2=covariance,
    )
    weights = np.array([10.0e9, 35.0e9, 25.0e9])
    np.testing.assert_allclose(result["frequency_integration_weights_Hz"], weights)
    np.testing.assert_allclose(
        result["integrated_receiver_power_W"], weights @ spectral_power
    )
    np.testing.assert_allclose(
        result["integrated_receiver_power_uncertainty_W"],
        np.sqrt(weights @ covariance @ weights),
    )


def test_microwave_background_api_exposes_receiver_power_and_integrated_band_power():
    from tokamak_analysis.cts_full_physics import analyze_cts_microwave_background

    result = analyze_cts_microwave_background(
        np.array([90.0e9, 100.0e9, 110.0e9]),
        np.array([0.1]),
        np.array([1000.0]),
        electron_density_m3=np.array([1.0e19]),
        ion_densities_m3=np.array([[1.0e19]]),
        ion_charge_states=np.array([1]),
        receiver_etendue_m2_sr=1.0e-6,
        end_to_end_efficiency=0.5,
    )
    assert result["receiver_conversion_applied_once"] is True
    assert result["receiver_spectral_power_W_Hz"].shape == (3,)
    assert result["integrated_receiver_power_W"] > 0.0
    assert result["receiver_power_uncertainty_status"].startswith("not_propagated")


def test_microwave_background_api_requires_receiver_factors_as_a_pair():
    from tokamak_analysis.cts_full_physics import analyze_cts_microwave_background

    with pytest.raises(ValueError, match="must be supplied together"):
        analyze_cts_microwave_background(
            np.array([90.0e9, 100.0e9]),
            np.array([0.1]),
            np.array([1000.0]),
            electron_density_m3=np.array([1.0e19]),
            ion_densities_m3=np.array([[1.0e19]]),
            ion_charge_states=np.array([1]),
            receiver_etendue_m2_sr=1.0e-6,
        )

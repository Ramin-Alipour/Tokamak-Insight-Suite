"""Physics-level regression tests for calibrated CTS thermal backgrounds.

These are deterministic numerical/software checks, not experimental validation.
"""
import numpy as np

from tokamak_analysis.cts_em_background import (
    integrate_thermal_microwave_background,
    planck_spectral_radiance,
    radiance_to_receiver_spectral_power,
    thermal_bremsstrahlung_emissivity_si,
)


def test_free_free_emissivity_scales_linearly_with_electron_density():
    frequency = np.array([90.0e9, 140.0e9, 220.0e9])
    ions = np.array([1.0e19, 0.5e19])
    charges = np.array([1.0, 2.0])
    base = thermal_bremsstrahlung_emissivity_si(
        frequency, 2.0e19, ions, charges, 1500.0, gaunt_factor=1.15
    )
    doubled = thermal_bremsstrahlung_emissivity_si(
        frequency, 4.0e19, ions, charges, 1500.0, gaunt_factor=1.15
    )
    np.testing.assert_allclose(doubled, 2.0 * base, rtol=1e-13, atol=0.0)


def test_free_free_emissivity_uses_charge_squared_ion_weighting():
    frequency = np.array([110.0e9])
    ne = 2.0e19
    te = 1800.0
    one = thermal_bremsstrahlung_emissivity_si(
        frequency, ne, np.array([1.0e19]), np.array([1.0]), te
    )
    two = thermal_bremsstrahlung_emissivity_si(
        frequency, ne, np.array([1.0e19]), np.array([2.0]), te
    )
    np.testing.assert_allclose(two, 4.0 * one, rtol=1e-13, atol=0.0)


def test_lte_transfer_preserves_planck_radiance_at_thermal_equilibrium():
    frequency = np.array([80.0e9, 120.0e9, 180.0e9])
    temperature_eV = 2000.0
    from tokamak_analysis.cts_em_background import E_CHARGE, K_B

    temperature_K = temperature_eV * E_CHARGE / K_B
    incident = planck_spectral_radiance(frequency, temperature_K)
    result = integrate_thermal_microwave_background(
        frequency,
        np.array([0.1, 0.2]),
        np.array([temperature_eV, temperature_eV]),
        electron_density_m3=np.array([2.0e19, 1.8e19]),
        ion_densities_m3=np.array([[2.0e19], [1.8e19]]),
        ion_charge_states=np.array([1.0]),
        ece_absorption_coefficient_m_inverse=np.full((2, frequency.size), 0.02),
        gaunt_factor=1.1,
        incident_radiance_W_m2_sr_Hz=incident,
    )
    np.testing.assert_allclose(
        result["spectral_radiance_W_m2_sr_Hz"], incident, rtol=2e-13, atol=0.0
    )


def test_mode_resolved_ece_opacity_changes_transfer_and_is_reported():
    frequency = np.array([100.0e9, 150.0e9])
    common = dict(
        frequency_Hz=frequency,
        path_length_m=np.array([0.2]),
        electron_temperature_eV=np.array([1200.0]),
        electron_density_m3=np.array([2.0e19]),
        ion_densities_m3=np.array([[2.0e19]]),
        ion_charge_states=np.array([1.0]),
        gaunt_factor=1.0,
        incident_radiance_W_m2_sr_Hz=0.0,
    )
    without_ece = integrate_thermal_microwave_background(
        **common, ece_absorption_coefficient_m_inverse=None
    )
    with_ece = integrate_thermal_microwave_background(
        **common,
        ece_absorption_coefficient_m_inverse=np.full((1, frequency.size), 0.5),
    )
    assert np.all(with_ece["spectral_radiance_W_m2_sr_Hz"] >
                  without_ece["spectral_radiance_W_m2_sr_Hz"])
    assert with_ece["ece_model_status"].startswith("mode-resolved")
    assert np.all(with_ece["ece_optical_depth"] > 0.0)


def test_receiver_conversion_applies_etendue_and_efficiency_once():
    radiance = np.array([2.0e-12, 4.0e-12])
    etendue = 3.0e-5
    efficiency = 0.4
    power = radiance_to_receiver_spectral_power(
        radiance,
        receiver_etendue_m2_sr=etendue,
        end_to_end_efficiency=efficiency,
    )
    np.testing.assert_allclose(power, radiance * etendue * efficiency, rtol=0.0, atol=0.0)

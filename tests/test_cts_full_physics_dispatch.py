import numpy as np
import pytest

from tokamak_analysis.cts_full_physics import (
    analyze_cts_microwave_background,
    run_advanced_thomson_physics,
)


def _vacuum_background_inputs():
    return {
        "frequency_Hz": np.array([90e9, 100e9, 110e9]),
        "path_length_m": np.array([0.1, 0.2]),
        "electron_temperature_eV": np.array([100.0, 150.0]),
        "electron_density_m3": np.zeros(2),
        "ion_densities_m3": np.zeros((2, 1)),
        "ion_charge_states": np.array([1.0]),
    }


def test_microwave_background_is_reachable_through_public_dispatcher():
    result = run_advanced_thomson_physics(
        "microwave_background", **_vacuum_background_inputs()
    )
    np.testing.assert_array_equal(
        result["spectral_radiance_W_m2_sr_Hz"], np.zeros(3)
    )
    assert result["public_physics_entrypoint"] == "microwave_background"
    assert result["background_status"] == "zero_background_exact_zero_radiance"
    assert result["automatic_model_substitution"] is False


def test_microwave_background_direct_entrypoint_matches_dispatcher():
    direct = analyze_cts_microwave_background(**_vacuum_background_inputs())
    dispatched = run_advanced_thomson_physics(
        "microwave_background", **_vacuum_background_inputs()
    )
    np.testing.assert_array_equal(
        direct["spectral_radiance_W_m2_sr_Hz"],
        dispatched["spectral_radiance_W_m2_sr_Hz"],
    )


def test_dispatcher_rejects_unknown_physics_without_fallback():
    with pytest.raises(ValueError, match="no automatic model substitution"):
        run_advanced_thomson_physics("unimplemented_relativistic_ece")


def test_relativistic_kinetic_ece_is_reachable_through_public_dispatcher():
    from tokamak_analysis.ece_kinetic import relativistic_kinetic_ece_absorption_coefficient

    inputs = {
        "frequency_Hz": 100e9,
        "wavevector_rad_m": [0.0, 0.0, 2*np.pi*100e9/299792458.0],
        "magnetic_field_T": [0.0, 0.0, 2.0],
        "electron_density_m3": 1e19,
        "electron_distribution": {
            "kind": "relativistic_maxwell_juttner",
            "temperature_eV": 1000.0,
        },
        "polarization": [1.0, 0.0, 0.0],
        "quadrature_order": 4,
        "gyrophase_points": 16,
        "resonance_width_rad_s": 2e7,
    }
    direct = relativistic_kinetic_ece_absorption_coefficient(**inputs)
    dispatched = run_advanced_thomson_physics("relativistic_kinetic_ece", **inputs)
    assert np.isfinite(direct["absorption_coefficient_m_inverse"])
    assert direct["conductivity_S_m"].shape == (3, 3)
    assert direct["relativistic_kinematics"] is True
    assert direct["distribution_is_external"] is True
    np.testing.assert_allclose(
        dispatched["absorption_coefficient_m_inverse"],
        direct["absorption_coefficient_m_inverse"],
    )
    assert dispatched["public_physics_entrypoint"] == "relativistic_kinetic_ece"


def test_relativistic_kinetic_ece_rejects_missing_distribution():
    from tokamak_analysis.ece_kinetic import relativistic_kinetic_ece_absorption_coefficient

    with pytest.raises((ValueError, TypeError)):
        relativistic_kinetic_ece_absorption_coefficient(
            100e9, [0.0, 0.0, 2.1e3], [0.0, 0.0, 2.0],
            electron_density_m3=1e19, electron_distribution=None,
            polarization=[1.0, 0.0, 0.0],
            quadrature_order=4, gyrophase_points=16,
        )


def test_isotropic_kappa_electrostatic_susceptibility_is_publicly_dispatched():
    from tokamak_analysis.cts_kappa import kappa_electron_susceptibility

    inputs = {
        "v": np.linspace(-2.0e7, 2.0e7, 1001),
        "phase_velocity_m_s": 1.0e6,
        "electron_density_m3": 2.0e18,
        "electron_temperature_eV": 250.0,
        "kappa": 4.0,
        "wavenumber_m_inverse": 120.0,
        "pole_imaginary_part_m_s": 2.0e3,
    }
    direct = kappa_electron_susceptibility(**inputs)
    dispatched = run_advanced_thomson_physics(
        "kappa_electrostatic_susceptibility", **inputs
    )
    assert dispatched["public_physics_entrypoint"] == "kappa_electrostatic_susceptibility"
    assert dispatched["automatic_model_substitution"] is False
    np.testing.assert_allclose(
        dispatched["electron_susceptibility"],
        direct["electron_susceptibility"],
        rtol=0.0,
        atol=0.0,
    )
    assert dispatched["distribution_model"] == "exact_1D_marginal_of_isotropic_3D_Kappa"
    assert "No conserving collisions" in " ".join(dispatched["limitations"])

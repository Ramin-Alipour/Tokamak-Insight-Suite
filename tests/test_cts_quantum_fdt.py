"""Physics regression tests for the symmetrized quantum CTS FDT closure."""
import numpy as np
import pytest
from scipy.constants import hbar

from tokamak_analysis.cts_fdt import (
    classical_fdt_dynamic_structure_factor,
    quantum_fdt_dynamic_structure_factor,
)

def _passive_response(omega):
    amplitude = 3.0e19
    tau = 2.0e-6
    return (-amplitude / (1.0 - 1j * omega * tau))[:, None, None]


def test_quantum_fdt_recovers_classical_fdt_when_hbar_omega_over_2T_is_small():
    temperature = 100.0 * 1.602176634e-19
    omega = np.array([-2.0e5, 2.0e5])
    response = _passive_response(omega)
    classical = classical_fdt_dynamic_structure_factor(
        omega, response, temperature_J=temperature
    )["dynamic_structure_factor_m3_s"]
    quantum = quantum_fdt_dynamic_structure_factor(
        omega, response, temperature_J=temperature
    )["dynamic_structure_factor_m3_s"]
    np.testing.assert_allclose(quantum, classical, rtol=2e-12, atol=0.0)


def test_quantum_fdt_uses_coth_factor_outside_classical_limit():
    temperature = 1.0e-22
    omega = np.array([2.0e13])
    response = _passive_response(omega)
    result = quantum_fdt_dynamic_structure_factor(
        omega, response, temperature_J=temperature
    )
    classical = classical_fdt_dynamic_structure_factor(
        omega, response, temperature_J=temperature
    )["dynamic_structure_factor_m3_s"]
    y = hbar * omega[0] / (2.0 * temperature)
    expected_ratio = y / np.tanh(y)
    np.testing.assert_allclose(
        result["dynamic_structure_factor_m3_s"],
        classical * expected_ratio,
        rtol=2e-13,
        atol=0.0,
    )
    assert result["detailed_balance_representation"] == "symmetrized_spectrum_only"


def test_quantum_fdt_is_positive_semidefinite_for_positive_and_negative_frequency():
    omega = np.array([-3.0e12, 3.0e12])
    result = quantum_fdt_dynamic_structure_factor(
        omega, _passive_response(omega), temperature_J=2.0e-21
    )
    assert result["positive_semidefinite"] is True
    assert np.all(
        np.linalg.eigvalsh(result["dynamic_structure_factor_m3_s"]) >= 0.0
    )


@pytest.mark.parametrize("temperature", [0.0, -1.0, np.nan, True])
def test_quantum_fdt_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature_J"):
        quantum_fdt_dynamic_structure_factor(
            [1.0], _passive_response(np.array([1.0])),
            temperature_J=temperature,
        )


def test_quantum_fdt_rejects_zero_frequency_without_static_extrapolation():
    with pytest.raises(ValueError, match="zero frequency"):
        quantum_fdt_dynamic_structure_factor(
            [0.0], np.zeros((1, 1, 1), dtype=complex), temperature_J=1.0
        )


def test_quantum_fdt_is_available_from_analysis1_physics_dispatcher():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    omega = np.array([2.0e12])
    result = run_advanced_thomson_physics(
        "quantum_fdt",
        angular_frequency_rad_s=omega,
        retarded_density_response_m3_per_J=_passive_response(omega),
        temperature_J=1.0e-20,
    )
    assert result["public_physics_entrypoint"] == "quantum_fdt"
    assert result["closure"] == "symmetrized_quantum_matrix_fluctuation_dissipation_theorem"
    assert result["automatic_model_substitution"] is False

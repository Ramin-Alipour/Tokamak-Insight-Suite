import numpy as np
import pytest

from tokamak_analysis.cts_fdt import classical_fdt_dynamic_structure_factor
from tokamak_analysis.full_vector_maxwell import propagate_full_vector_stratified_maxwell


def test_classical_fdt_scalar_passive_response_is_nonnegative():
    # chi = -1/(1 - i*w*tau) has negative Im for positive omega
    w = np.array([0.5, 1.0, 2.0])
    tau = 1.0
    chi = (-1.0 / (1.0 - 1j*w*tau))[:, None, None]
    result = classical_fdt_dynamic_structure_factor(w, chi, temperature_J=2.0)
    assert np.all(np.real(result["dynamic_structure_factor_m3_s"][:, 0, 0]) >= 0)
    assert result["positive_semidefinite"]


def test_fdt_rejects_zero_frequency_and_nonpassive_response():
    with pytest.raises(ValueError, match="zero frequency"):
        classical_fdt_dynamic_structure_factor([0.0], np.zeros((1, 1, 1)), temperature_J=1.0)
    with pytest.raises(ValueError, match="positive semidefinite"):
        classical_fdt_dynamic_structure_factor([1.0], np.array([[[1j]]]), temperature_J=1.0)


def test_full_vector_maxwell_vacuum_transverse_plane_wave():
    from scipy.constants import c, mu_0, epsilon_0
    f = 10e9
    omega = 2*np.pi*f
    kz = omega/c
    eta = np.sqrt(mu_0/epsilon_0)
    eps = np.eye(3, dtype=complex)
    initial = np.array([1.0, 0.0, 0.0, 1.0/eta], dtype=complex)
    out = propagate_full_vector_stratified_maxwell(
        [0.0, 0.02], [0.0, 0.0], omega, 1.0,
        lambda z: eps, initial, samples=21,
    )
    expected = np.exp(1j*kz*out["z_m"])
    np.testing.assert_allclose(out["electric_field_V_m"][:, 0], expected, rtol=2e-6, atol=2e-6)
    assert np.max(np.abs(out["electric_field_V_m"][:, 2])) < 1e-9
    assert np.ptp(out["poynting_flux_z_W_m2"]) < 1e-8


def test_full_vector_solver_retains_off_diagonal_coupling():
    omega = 2*np.pi*20e9
    eps = np.array([[2.0, 0.15j, 0.0], [-0.15j, 2.0, 0.0], [0.0, 0.0, 2.0]], dtype=complex)
    initial = np.array([1.0, 0.0, 0.0, 1e-3], dtype=complex)
    out = propagate_full_vector_stratified_maxwell(
        [0.0, 0.01], [0.0, 0.0], omega, 1.0,
        lambda z: eps, initial, samples=31,
    )
    assert np.max(np.abs(out["electric_field_V_m"][:, 1])) > 0.0
    assert out["polarization_coupling_retained"]


def test_full_vector_solver_rejects_longitudinal_singularity():
    eps = np.diag([1.0, 1.0, 0.0]).astype(complex)
    with pytest.raises(ValueError, match="epsilon_zz is singular"):
        propagate_full_vector_stratified_maxwell(
            [0.0, 1.0], [0.0, 0.0], 1e9, 1.0, lambda z: eps,
            np.ones(4, dtype=complex),
        )

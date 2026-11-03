import numpy as np
import pytest

from tokamak_analysis.thomson_advanced_solvers import (
    _berreman_matrix,
    propagate_stratified_vector_maxwell,
)


def test_berreman_vacuum_generator_has_correct_plane_wave_sign():
    from scipy.constants import c, mu_0, epsilon_0
    frequency = 2.0e9
    omega = 2 * np.pi * frequency
    z0 = np.sqrt(mu_0 / epsilon_0)
    state = np.array([1.0, 0.0, 0.0, 1.0 / z0], dtype=complex)
    derivative = _berreman_matrix(np.eye(3), omega) @ state
    expected = 1j * omega / c * state
    np.testing.assert_allclose(derivative, expected, rtol=1e-12, atol=1e-12)


def test_stratified_vector_solver_propagates_vacuum_phase():
    from scipy.constants import c, mu_0, epsilon_0
    frequency = 1.0e9
    omega = 2 * np.pi * frequency
    z = np.linspace(0.0, 0.2, 21)
    z0 = np.sqrt(mu_0 / epsilon_0)
    initial = np.array([1.0, 0.0, 0.0, 1.0 / z0], dtype=complex)
    result = propagate_stratified_vector_maxwell(
        z, lambda _position: np.eye(3), frequency, initial,
        rtol=1e-10, atol=1e-12,
    )
    expected = np.exp(1j * omega * z / c)
    np.testing.assert_allclose(result["electric_field_V_m"][:, 0], expected,
                               rtol=2e-8, atol=2e-8)
    np.testing.assert_allclose(result["electric_field_V_m"][:, 1], 0.0,
                               atol=1e-12)
    assert result["nonparaxial"] is True
    assert result["integration"]["success"] is True


def test_berreman_rejects_singular_epsilon_zz():
    eps = np.eye(3, dtype=complex)
    eps[2, 2] = 0.0
    with pytest.raises(ValueError, match="epsilon_zz"):
        _berreman_matrix(eps, 1.0e9)

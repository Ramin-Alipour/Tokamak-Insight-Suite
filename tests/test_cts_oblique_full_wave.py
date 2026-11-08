import numpy as np
import pytest

from tokamak_analysis.cts_full_wave import (
    solve_stratified_full_wave_oblique,
    berreman_matrix_oblique,
)


def test_oblique_vacuum_slab_is_lossless_and_reflectionless():
    z = np.linspace(0.0, 0.02, 31)
    eps = np.repeat(np.eye(3, dtype=complex)[None, :, :], z.size, axis=0)
    omega = 2*np.pi*10e9
    kx = 0.35*omega/299792458.0
    out = solve_stratified_full_wave_oblique(
        omega, z, eps, kx_rad_m=kx, incident_amplitude_V_m=(1.0, 0.2j)
    )
    assert out["reflectance"] < 1e-7
    assert out["transmittance"] == pytest.approx(1.0, rel=1e-6)
    assert abs(out["absorptance"]) < 1e-6


def test_oblique_berreman_matrix_reduces_to_normal_incidence_matrix():
    from tokamak_analysis.cts_full_wave import berreman_matrix
    eps = np.array([[2.0, 0.1j, 0.0], [-0.1j, 2.2, 0.0], [0.0, 0.0, 1.7]], complex)
    omega = 2*np.pi*20e9
    np.testing.assert_allclose(
        berreman_matrix_oblique(omega, eps, 0.0, 0.0),
        berreman_matrix(omega, eps), rtol=1e-13, atol=1e-13
    )


def test_oblique_solver_rejects_evanescent_vacuum_ports():
    z = np.array([0.0, 0.01])
    eps = np.repeat(np.eye(3, dtype=complex)[None, :, :], 2, axis=0)
    with pytest.raises(ValueError, match="propagating vacuum ports"):
        solve_stratified_full_wave_oblique(
            2*np.pi*10e9, z, eps, kx_rad_m=2*np.pi*10e9/299792458.0
        )

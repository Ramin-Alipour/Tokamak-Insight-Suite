import numpy as np
from tokamak_analysis.cts_full_wave import solve_stratified_mode_conversion


def test_mode_conversion_matrix_is_identity_in_vacuum():
    z = np.linspace(0, 0.02, 21)
    eps = np.repeat(np.eye(3, dtype=complex)[None, :, :], z.size, axis=0)
    out = solve_stratified_mode_conversion(
        2*np.pi*10e9, z, eps, kx_rad_m=0.2*2*np.pi*10e9/299792458.0)
    np.testing.assert_allclose(out["reflected_jones_matrix"], 0, atol=1e-7)
    np.testing.assert_allclose(out["transmitted_jones_matrix"], np.eye(2), atol=1e-6)


def test_mode_conversion_matrix_has_two_complex_polarization_channels():
    z = np.linspace(0, 0.01, 15)
    eps = np.repeat(np.diag([2.0, 2.5, 1.8])[None, :, :], z.size, axis=0).astype(complex)
    out = solve_stratified_mode_conversion(
        2*np.pi*8e9, z, eps, kx_rad_m=0.1*2*np.pi*8e9/299792458.0)
    assert out["reflected_jones_matrix"].shape == (2, 2)
    assert out["transmitted_jones_matrix"].shape == (2, 2)
    assert np.all(np.isfinite(out["transmitted_jones_matrix"]))

import numpy as np
import pytest
from scipy.constants import c
from tokamak_analysis.cts_full_wave import propagate_vectorial_angular_spectrum


def test_vector_angular_spectrum_propagates_uniform_plane_wave_exactly():
    omega = 2*np.pi*30e9
    distance = 0.012
    field = np.zeros((8, 16, 2), dtype=complex)
    field[..., 0] = 1.0
    out = propagate_vectorial_angular_spectrum(
        omega, field, dx_m=1e-3, dy_m=1e-3,
        propagation_distance_m=distance,
    )
    expected = np.exp(1j*omega*distance/c)
    np.testing.assert_allclose(out["electric_field_xyz_V_m"][..., 0], expected, atol=1e-10)
    np.testing.assert_allclose(out["electric_field_xyz_V_m"][..., 2], 0.0, atol=1e-10)
    assert out["solver"] == "exact_nonparaxial_vector_angular_spectrum"


def test_vector_angular_spectrum_retains_longitudinal_field_for_oblique_component():
    omega = 2*np.pi*20e9
    k0 = omega/c
    kx = 0.4*k0
    nx, ny = 32, 8
    length_x = 2*np.pi/kx
    dx = length_x/nx
    x = np.arange(nx)*dx
    field = np.zeros((ny, nx, 2), dtype=complex)
    field[..., 0] = np.exp(1j*kx*x)[None, :]
    out = propagate_vectorial_angular_spectrum(
        omega, field, dx_m=dx, dy_m=1e-3,
        propagation_distance_m=0.003,
    )
    ex = out["electric_field_xyz_V_m"][0, :, 0]
    ez = out["electric_field_xyz_V_m"][0, :, 2]
    kz = np.sqrt(k0*k0-kx*kx)
    np.testing.assert_allclose(ez/ex, -kx/kz, rtol=1e-10, atol=1e-10)


def test_vector_angular_spectrum_rejects_gain_and_negative_distance():
    field = np.ones((4, 4, 2), dtype=complex)
    with pytest.raises(ValueError, match="passive"):
        propagate_vectorial_angular_spectrum(
            1e10, field, dx_m=1, dy_m=1,
            propagation_distance_m=0.1, relative_permittivity=1-0.1j,
        )
    with pytest.raises(ValueError, match="non-negative"):
        propagate_vectorial_angular_spectrum(
            1e10, field, dx_m=1, dy_m=1, propagation_distance_m=-1,
        )

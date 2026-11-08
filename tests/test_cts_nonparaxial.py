import numpy as np
import pytest

from tokamak_analysis.cts_nonparaxial import propagate_nonparaxial_vector_field


def test_nonparaxial_vector_propagator_preserves_transverse_plane_wave():
    nx = ny = 32
    dx = dy = 2.0e-3
    x = (np.arange(nx) - nx // 2) * dx
    y = (np.arange(ny) - ny // 2) * dy
    f = 30e9
    n = 1.0
    k = 2 * np.pi * f * n / 299792458.0
    kx = 2 * np.pi / (nx * dx) * 2
    ky = 2 * np.pi / (ny * dy)
    xx, yy = np.meshgrid(x, y)
    phase = np.exp(1j * (kx * xx + ky * yy))
    # E is chosen perpendicular to k=(kx,ky,kz).
    kz = np.sqrt(k*k-kx*kx-ky*ky)
    e = np.array([1.0, 0.0, -kx/kz], dtype=complex)
    e /= np.linalg.norm(e)
    field = phase[..., None] * e
    result = propagate_nonparaxial_vector_field(
        x, y, field, f, n, 0.01
    )
    expected = field * np.exp(1j * kz * 0.01)
    np.testing.assert_allclose(result["electric_field_V_m"], expected, atol=1e-10)
    assert result["paraxial_approximation_used"] is False
    assert result["transversality_relative_residual"] < 1e-12


def test_nonparaxial_vector_propagator_rejects_bad_grid_and_inputs():
    x = np.arange(8, dtype=float)
    y = np.arange(8, dtype=float)
    field = np.zeros((8, 8, 3), dtype=complex)
    with pytest.raises(ValueError, match="uniformly spaced"):
        propagate_nonparaxial_vector_field(
            [0, 1, 2, 4, 5, 6, 7, 8], y, field, 1e9, 1.0, 0.1
        )
    with pytest.raises(ValueError, match="non-negative"):
        propagate_nonparaxial_vector_field(x, y, field, 1e9, 1.0, -0.1)


def test_evanescent_components_decay_and_can_be_discarded():
    nx = ny = 16
    x = np.arange(nx) * 1e-4
    y = np.arange(ny) * 1e-4
    xx, yy = np.meshgrid(x, y)
    # High transverse spatial frequency, above the free-space light cone.
    field = np.zeros((ny, nx, 3), dtype=complex)
    field[..., 0] = np.exp(2j * np.pi * 7 * np.arange(nx)[None, :] / nx)
    decayed = propagate_nonparaxial_vector_field(x, y, field, 1e9, 1.0, 1e-3)
    discarded = propagate_nonparaxial_vector_field(
        x, y, field, 1e9, 1.0, 1e-3, evanescent="discard"
    )
    assert decayed["evanescent_component_count"] > 0
    assert np.linalg.norm(decayed["electric_field_V_m"]) < np.linalg.norm(field)
    assert np.linalg.norm(discarded["electric_field_V_m"]) == pytest.approx(0.0, abs=1e-12)

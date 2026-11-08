import numpy as np
import pytest

from tokamak_analysis.cts_full_wave import solve_3d_periodic_full_wave


def _grid():
    x = np.arange(4, dtype=float) * 0.01
    y = np.arange(4, dtype=float) * 0.01
    z = np.arange(4, dtype=float) * 0.01
    return x, y, z


def test_3d_periodic_tensor_maxwell_solver_returns_vector_fields_and_passive_absorption():
    x, y, z = _grid()
    shape = (z.size, y.size, x.size)
    epsilon = np.zeros(shape + (3, 3), dtype=complex)
    epsilon[..., 0, 0] = 2.0 + 0.08j
    epsilon[..., 1, 1] = 2.2 + 0.06j
    epsilon[..., 2, 2] = 1.8 + 0.05j
    # Spatially varying Hermitian off-diagonal coupling exercises 3-D
    # polarization mixing rather than three independent scalar equations.
    epsilon[..., 0, 1] = 0.03j
    epsilon[..., 1, 0] = -0.03j
    current = np.zeros(shape + (3,), dtype=complex)
    current[1, 1, 1, 0] = 1.0
    current[2, 1, 2, 1] = 0.5j

    result = solve_3d_periodic_full_wave(
        2*np.pi*3e9, x, y, z, epsilon,
        impressed_current_density_A_m2=current,
    )

    assert result["electric_field_V_m"].shape == shape + (3,)
    assert result["magnetic_field_A_m"].shape == shape + (3,)
    assert result["relative_linear_system_residual"] < 1e-7
    assert np.all(result["local_absorbed_power_density_W_m3"] >= -1e-10)
    assert result["boundary_condition"] == "three_dimensional_bloch_periodic"
    np.testing.assert_array_equal(result["bloch_wavevector_rad_m"], np.zeros(3))
    assert "full_coherent_3D_vector_coupling" in result["mode_conversion"]


def test_3d_periodic_solver_rejects_active_dielectric_tensor():
    x, y, z = _grid()
    current = np.zeros((z.size, y.size, x.size, 3), dtype=complex)
    epsilon = np.eye(3, dtype=complex)
    epsilon[0, 0] = 1.0 - 0.1j
    with pytest.raises(ValueError, match="passive"):
        solve_3d_periodic_full_wave(
            2*np.pi*3e9, x, y, z, epsilon,
            impressed_current_density_A_m2=current,
        )


def test_3d_periodic_solver_requires_full_three_component_current():
    x, y, z = _grid()
    with pytest.raises(ValueError, match="impressed_current_density"):
        solve_3d_periodic_full_wave(
            2*np.pi*3e9, x, y, z, np.eye(3),
            impressed_current_density_A_m2=np.zeros((z.size, y.size, x.size)),
        )



def test_3d_solver_accepts_nonzero_bloch_wavevector():
    x, y, z = _grid()
    shape = (z.size, y.size, x.size)
    current = np.zeros(shape + (3,), dtype=complex)
    current[1, 1, 1, 0] = 1.0
    epsilon = np.eye(3, dtype=complex) * (2.0 + 0.1j)
    result = solve_3d_periodic_full_wave(
        2*np.pi*3e9, x, y, z, epsilon,
        impressed_current_density_A_m2=current,
        bloch_wavevector_rad_m=(10.0, -5.0, 2.0),
    )
    np.testing.assert_allclose(result["bloch_wavevector_rad_m"], [10.0, -5.0, 2.0])
    assert result["relative_linear_system_residual"] < 1e-7

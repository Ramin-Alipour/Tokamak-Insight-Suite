import numpy as np
import pytest

from tis.equilibrium import solve_grad_shafranov


def test_grad_shafranov_solver_reproduces_harmonic_r_squared_solution():
    R = np.linspace(1.0, 2.0, 21)
    Z = np.linspace(-0.5, 0.5, 17)
    boundary = np.broadcast_to(R[None, :] ** 2, (len(Z), len(R))).copy()
    result = solve_grad_shafranov(R, Z, np.zeros_like(boundary), boundary)
    np.testing.assert_allclose(result["poloidal_flux_Wb"], boundary, atol=1e-9)
    assert result["interior_residual_rms"] < 1e-8


def test_grad_shafranov_rejects_nonuniform_grid():
    R = np.array([1.0, 1.1, 1.3, 1.6])
    Z = np.linspace(-1, 1, 4)
    field = np.zeros((4, 4))
    with pytest.raises(ValueError, match="uniform"):
        solve_grad_shafranov(R, Z, field, field)

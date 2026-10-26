import numpy as np
import pytest

from tokamak_analysis.equilibrium_reconstruction import (
    grad_shafranov_physical_relation,
    reconstruct_equilibrium,
)


def grid():
    return np.linspace(1.0, 2.0,  nine := 9), np.linspace(-0.5, 0.5, nine)


def test_solver_preserves_boundary_and_recovers_harmonic_flux():
    R, Z = grid()
    boundary = np.broadcast_to(R[None, :] ** 2, (Z.size, R.size)).copy()
    result = reconstruct_equilibrium(
        R, Z, boundary, [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
        tolerance=1e-9,
    )
    np.testing.assert_allclose(result["psi_Wb_rad"][[0, -1], :], boundary[[0, -1], :])
    np.testing.assert_allclose(result["psi_Wb_rad"][:, [0, -1]], boundary[:, [0, -1]])
    np.testing.assert_allclose(result["psi_Wb_rad"], boundary, atol=2e-6)
    assert result["converged"] is True
    assert result["iterations"] > 0
    assert np.all(np.isfinite(result["B_R_T"]))
    assert np.all(np.isfinite(result["B_Z_T"]))
    np.testing.assert_allclose(result["B_R_T"], 0.0, atol=1e-5)
    np.testing.assert_allclose(result["B_Z_T"], 2.0, atol=1e-5)


def test_physical_relation_returns_small_residual_for_exact_quadratic_solution():
    R, Z = grid()
    psi = R[None, :] ** 2 + Z[:, None] ** 2
    dp = np.zeros_like(psi)
    ffprime = np.full_like(psi, -2.0)
    result = grad_shafranov_physical_relation(R, Z, psi, dp, ffprime)
    np.testing.assert_allclose(result["grad_shafranov_operator_Wb_rad_m-2"], 2.0, atol=1e-10)
    np.testing.assert_allclose(result["physical_source_Wb_rad_m-2"], 2.0, atol=1e-12)
    np.testing.assert_allclose(result["equation_residual_Wb_rad_m-2"], 0.0, atol=1e-10)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"R_m": [1, 1, 2]}, "strictly increasing"),
        ({"boundary_flux_Wb_rad": np.zeros((2, 3))}, "Boundary flux"),
        ({"psi_profile": [0.5, 0.5]}, "increase strictly"),
        ({"max_iterations": 0}, "Solver controls"),
    ],
)
def test_solver_rejects_invalid_inputs(kwargs, message):
    R, Z = grid()
    boundary = np.zeros((Z.size, R.size))
    args = dict(
        R_m=R, Z_m=Z, boundary_flux_Wb_rad=boundary,
        psi_profile=[0.0, 1.0], pressure_gradient_Pa_per_Wb=[0.0, 0.0],
        ffprime_T2=[0.0, 0.0],
    )
    args.update(kwargs)
    with pytest.raises(ValueError, match=message):
        reconstruct_equilibrium(**args)


def test_iteration_cap_is_reported_without_false_convergence_claim():
    R, Z = grid()
    boundary = np.zeros((Z.size, R.size))
    boundary[0, :] = 1.0
    result = reconstruct_equilibrium(
        R, Z, boundary, [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
        max_iterations=1, tolerance=1e-14,
    )
    assert result["iterations"] == 1
    assert result["converged"] is False

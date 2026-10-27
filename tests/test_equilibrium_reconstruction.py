import numpy as np
import pytest

from tokamak_analysis.equilibrium_reconstruction import (
    grad_shafranov_physical_relation,
    reconstruct_equilibrium,
    infer_grad_shafranov_profiles_from_flux,
    reconstruct_equilibrium_from_reference_flux,
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
    assert result["equation_residual_relative_rms"] <= 1e-8
    assert result["iterations"] > 0
    assert np.all(np.isfinite(result["B_R_T"]))
    assert np.all(np.isfinite(result["B_Z_T"]))
    np.testing.assert_allclose(result["B_R_T"], 0.0, atol=1e-5)
    np.testing.assert_allclose(result["B_Z_T"], 2.0, atol=1e-5)



def test_solver_residual_scaling_is_invariant_for_small_flux_magnitudes():
    R, Z = grid()
    boundary = 1e-12 * np.broadcast_to(R[None, :] ** 2, (Z.size, R.size)).copy()
    result = reconstruct_equilibrium(
        R, Z, boundary, [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
        tolerance=1e-9,
    )
    assert result["converged"] is True
    assert result["equation_residual_relative_rms"] <= 1e-8
    np.testing.assert_allclose(result["psi_Wb_rad"], boundary, atol=2e-18, rtol=1e-8)


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
        ({"max_iterations": True}, "Solver controls"),
        ({"tolerance": True}, "Solver controls"),
        ({"relaxation": False}, "Solver controls"),
        ({"tolerance": "1e-8"}, "Solver controls"),
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



@pytest.mark.parametrize(
    "R, Z, message",
    [
        ([1.0, 1.4, 2.0], np.linspace(-0.5, 0.5, 3), "Uniform grids"),
        ([1.0, np.nan, 2.0], np.linspace(-0.5, 0.5, 3), "finite"),
        ([0.0, 0.5, 1.0], np.linspace(-0.5, 0.5, 3), "R must be positive"),
    ],
)
def test_solver_rejects_nonphysical_or_nonuniform_coordinate_grids(R, Z, message):
    boundary = np.zeros((len(Z), len(R)))
    with pytest.raises(ValueError, match=message):
        reconstruct_equilibrium(R, Z, boundary, [0.0, 1.0], [0.0, 0.0], [0.0, 0.0])


@pytest.mark.parametrize(
    "dp, ff, message",
    [
        (np.zeros((3, 3)), np.zeros((4, 3)), "shape"),
        (np.full((3, 3), np.nan), np.zeros((3, 3)), "finite"),
        (np.zeros((3, 3)), np.full((3, 3), np.inf), "finite"),
    ],
)
def test_physical_relation_rejects_mismatched_or_nonfinite_profiles(dp, ff, message):
    R = np.linspace(1.0, 2.0, 3)
    Z = np.linspace(-0.5, 0.5, 3)
    psi = np.zeros((3, 3))
    with pytest.raises(ValueError, match=message):
        grad_shafranov_physical_relation(R, Z, psi, dp, ff)


def test_solver_reports_finite_interior_equation_residual_and_fixed_perimeter():
    R, Z = grid()
    boundary = np.zeros((Z.size, R.size))
    boundary[0, :] = np.linspace(0.0, 1.0, R.size)
    boundary[-1, :] = np.linspace(0.0, 1.0, R.size)
    boundary[:, 0] = 0.0
    boundary[:, -1] = 1.0
    result = reconstruct_equilibrium(
        R, Z, boundary, [0.0, 1.0], [0.0, 0.0], [0.0, 0.0],
        max_iterations=20000, tolerance=1e-9,
    )
    np.testing.assert_array_equal(result["psi_Wb_rad"][[0, -1], :], boundary[[0, -1], :])
    np.testing.assert_array_equal(result["psi_Wb_rad"][:, [0, -1]], boundary[:, [0, -1]])
    assert np.isfinite(result["equation_residual_rms_Wb_rad_m-2"])
    assert np.isfinite(result["equation_residual_max_abs_Wb_rad_m-2"])
    assert result["equation_residual_rms_Wb_rad_m-2"] >= 0
    assert result["equation_residual_max_abs_Wb_rad_m-2"] >= result["equation_residual_rms_Wb_rad_m-2"]
    assert result["psi_Wb_rad"].shape == (Z.size, R.size)
    assert result["B_poloidal_T"].shape == (Z.size, R.size)



def test_inverse_reconstruction_recovers_coefficients_and_removes_known_coil_response():
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_inverse

    R = np.linspace(1.0, 2.0, 9)
    Z = np.linspace(-0.5, 0.5, 9)
    rr, zz = np.meshgrid(R, Z)
    basis = np.stack((-(rr - 1.5) ** 2 - zz**2,
                      0.2 * (rr - 1.5) ** 2 + 0.1 * zz**2))
    expected = np.array([2.0, 0.5])
    probe_response = np.array([[1.0, 0.2], [0.1, 1.2], [0.7, -0.4]])
    probe_coil = np.array([[0.01], [0.02], [-0.01]])
    currents = np.array([10.0])
    loop_response = np.array([[0.4, 0.3], [-0.2, 0.8]])
    loop_coil = np.array([[0.002], [0.004]])
    result = reconstruct_equilibrium_inverse(
        R, Z, basis,
        magnetic_probe_response_T_per_coefficient=probe_response,
        magnetic_probe_values_T=probe_response @ expected + probe_coil @ currents,
        magnetic_probe_sigma_T=[0.01, 0.02, 0.01],
        coil_probe_response_T_per_A=probe_coil,
        coil_probe_currents_A=currents,
        flux_loop_response_Wb_rad_per_coefficient=loop_response,
        flux_loop_values_Wb_rad=loop_response @ expected + loop_coil @ currents,
        flux_loop_sigma_Wb_rad=[0.001, 0.002],
        coil_flux_response_Wb_rad_per_A=loop_coil,
        coil_flux_currents_A=currents,
        regularization=0.0,
    )
    np.testing.assert_allclose(result["basis_coefficients"], expected, rtol=1e-10, atol=1e-10)
    assert result["measurement_count"] == 5
    assert result["measurement_diagnostics"]["magnetic probes"]["normalized_residual_rms"] < 1e-9
    assert result["measurement_diagnostics"]["flux loops"]["normalized_residual_rms"] < 1e-9
    assert result["psi_Wb_rad"].shape == (len(Z), len(R))
    assert result["boundary_contours_RZ_m"]


@pytest.mark.parametrize("regularization", [True, False, "1e-8", np.nan, np.inf, -1.0])
def test_inverse_reconstruction_rejects_invalid_regularization(regularization):
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_inverse

    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    basis = np.stack((np.broadcast_to(R[None, :] ** 2, (len(Z), len(R))),
                      np.broadcast_to(Z[:, None] ** 2, (len(Z), len(R)))))
    with pytest.raises(ValueError, match="regularization must be a finite non-negative number"):
        reconstruct_equilibrium_inverse(
            R, Z, basis,
            magnetic_probe_response_T_per_coefficient=[[1.0, 0.0], [0.0, 1.0]],
            magnetic_probe_values_T=[1.0, 2.0],
            magnetic_probe_sigma_T=[0.1, 0.1],
            regularization=regularization,
        )


def test_inverse_reconstruction_rejects_rank_deficient_basis_and_invalid_uncertainty():
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_inverse

    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    basis = np.ones((2, len(Z), len(R)))
    with pytest.raises(ValueError, match="not identifiable"):
        reconstruct_equilibrium_inverse(
            R, Z, basis,
            magnetic_probe_response_T_per_coefficient=[[1.0, 1.0]],
            magnetic_probe_values_T=[1.0],
            magnetic_probe_sigma_T=[0.1],
        )
    basis = np.stack((np.ones((len(Z), len(R))), np.arange(len(Z) * len(R)).reshape(len(Z), len(R))))
    with pytest.raises(ValueError, match="uncertainties positive"):
        reconstruct_equilibrium_inverse(
            R, Z, basis,
            magnetic_probe_response_T_per_coefficient=[[1.0, 0.0], [0.0, 1.0]],
            magnetic_probe_values_T=[1.0, 2.0],
            magnetic_probe_sigma_T=[0.1, 0.0],
        )



def test_reference_flux_profile_inference_recovers_constant_source():
    R, Z = grid()
    rr, zz = np.meshgrid(R, Z)
    reference = rr**2 + zz**2
    profiles = infer_grad_shafranov_profiles_from_flux(R, Z, reference, n_profile_points=8)
    np.testing.assert_allclose(profiles["pressure_gradient_Pa_per_Wb"], 0.0)
    np.testing.assert_allclose(profiles["ffprime_T2"], -2.0, atol=1e-10)
    assert profiles["reference_operator_Wb_rad_m-2"].shape == reference.shape
    assert np.all(profiles["profile_bin_counts"] >= 0)


def test_reference_flux_workflow_reconstructs_grad_shafranov_solution():
    R, Z = grid()
    rr, zz = np.meshgrid(R, Z)
    reference = rr**2 + zz**2
    result = reconstruct_equilibrium_from_reference_flux(
        R, Z, reference, n_profile_points=8, tolerance=1e-9
    )
    assert result["converged"] is True
    np.testing.assert_allclose(result["psi_Wb_rad"], reference, atol=2e-6)
    assert result["reference_comparison"]["interior_flux_nrmse_by_reference_range"] < 1e-6
    assert result["reference_psi_Wb_rad"].shape == reference.shape


def test_reference_flux_workflow_rejects_bad_reference_map():
    R, Z = grid()
    with pytest.raises(ValueError, match="finite"):
        infer_grad_shafranov_profiles_from_flux(R, Z, np.full((len(Z), len(R)), np.nan))
    with pytest.raises(ValueError, match="nonzero finite range"):
        reconstruct_equilibrium_from_reference_flux(R, Z, np.ones((len(Z), len(R))))



def test_public_efit_loader_selects_real_flux_slice_and_records_provenance(monkeypatch):
    import sys
    from types import SimpleNamespace
    from tis.data_sources import fetch_public_efit_flux_map

    R = np.linspace(1.0, 2.0, 4)
    Z = np.linspace(-0.5, 0.5, 3)
    first = np.zeros((3, 4))
    second = np.arange(12, dtype=float).reshape(3, 4)
    row = {
        "efit_times": [100.0, 200.0],
        "efit_grid_R": R.tolist(),
        "efit_grid_Z": Z.tolist(),
        "efit_psirz": np.stack((first, second)).tolist(),
    }
    fake_datasets = SimpleNamespace(
        load_dataset=lambda *args, **kwargs: [row]
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    r_out, z_out, psi, manifest = fetch_public_efit_flux_map(0, 0.19)
    np.testing.assert_array_equal(r_out, R)
    np.testing.assert_array_equal(z_out, Z)
    np.testing.assert_array_equal(psi, second)
    assert manifest["dataset"] == "Sophelio/Fusion Equilibrium Challenge"
    assert manifest["selected_time_ms"] == 200.0
    assert manifest["license"] == "CC BY 4.0"



def test_inverse_regularization_is_invariant_to_response_column_scaling():
    """Tikhonov penalty acts on physical coefficients, not normalized columns."""
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_inverse

    R = np.linspace(1.0, 2.0, 5)
    Z = np.linspace(-0.5, 0.5, 5)
    rr, zz = np.meshgrid(R, Z)
    basis = np.stack((rr**2 + zz**2, rr + 2.0 * zz))
    response = np.array([[100.0, 0.0], [0.0, 1.0]])
    expected = np.array([10000.0 / 10001.0, 0.5])
    result = reconstruct_equilibrium_inverse(
        R, Z, basis,
        magnetic_probe_response_T_per_coefficient=response,
        magnetic_probe_values_T=[100.0, 1.0],
        magnetic_probe_sigma_T=[1.0, 1.0],
        regularization=1.0,
    )
    np.testing.assert_allclose(result["basis_coefficients"], expected, rtol=1e-10, atol=1e-10)


def test_joint_equilibrium_profile_reconstruction_extracts_pressure_and_current_profiles():
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_profiles_from_flux

    R = np.linspace(1.0, 2.0, 65)
    Z = np.linspace(-0.5, 0.5, 65)
    rr, zz = np.meshgrid(R, Z)
    psi = -((rr - 1.5) ** 2 + zz**2)
    result = reconstruct_equilibrium_profiles_from_flux(
        R, Z, psi, n_profile_points=24, toroidal_field_boundary_Tm=3.0
    )
    assert result["joint_pressure_ffprime_identifiable"] is True
    assert result["pressure_gradient_Pa_per_Wb"].shape == (24,)
    assert result["ffprime_T2"].shape == (24,)
    assert result["pressure_Pa_relative_to_edge"].shape == (24,)
    assert result["F_Tm"].shape == (24,)
    assert result["q_profile"].shape == (24,)
    assert result["q95"] is None or np.isfinite(result["q95"])
    assert result["axis_R_m"] == pytest.approx(1.5, abs=R[1] - R[0])
    assert result["axis_Z_m"] == pytest.approx(0.0, abs=Z[1] - Z[0])
    assert result["pressure_Pa_relative_to_edge"][-1] == pytest.approx(0.0)
    assert np.all(np.isfinite(result["pressure_gradient_Pa_per_Wb"]))
    assert np.all(np.isfinite(result["ffprime_T2"]))


def test_joint_equilibrium_profile_reconstruction_does_not_guess_toroidal_field():
    from tokamak_analysis.equilibrium_reconstruction import reconstruct_equilibrium_profiles_from_flux

    R = np.linspace(1.0, 2.0, 65)
    Z = np.linspace(-0.5, 0.5, 65)
    rr, zz = np.meshgrid(R, Z)
    psi = -((rr - 1.5) ** 2 + zz**2)
    result = reconstruct_equilibrium_profiles_from_flux(R, Z, psi, n_profile_points=16)
    assert "F_Tm" not in result
    assert "q_profile" not in result
    assert result["joint_pressure_ffprime_identifiable"] is True


def test_joint_profile_fit_recovers_analytic_pressure_and_ffprime_source_terms():
    from tokamak_analysis.equilibrium_reconstruction import MU0, reconstruct_equilibrium_profiles_from_flux

    R = np.linspace(1.0, 2.0, 81)
    Z = np.linspace(-0.5, 0.5, 81)
    rr, zz = np.meshgrid(R, Z)
    pprime_true = -1200.0
    ffprime_true = 1.5
    a = -MU0 * pprime_true / 8.0
    r_axis = 1.5
    psi = a * rr**4 - 2.0 * a * r_axis**2 * rr**2 - 0.5 * ffprime_true * zz**2
    result = reconstruct_equilibrium_profiles_from_flux(R, Z, psi, n_profile_points=16)
    assert result["joint_pressure_ffprime_identifiable"] is True
    valid = (result["profile_bin_counts"] >= 3) & (result["profile_bin_design_rank"] == 2)
    assert np.count_nonzero(valid) >= 2
    np.testing.assert_allclose(
        np.median(result["pressure_gradient_Pa_per_Wb"][valid]),
        pprime_true, rtol=0.03, atol=2.0,
    )
    np.testing.assert_allclose(
        np.median(result["ffprime_T2"][valid]),
        ffprime_true, rtol=0.03, atol=0.03,
    )

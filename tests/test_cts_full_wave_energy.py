import numpy as np

from tokamak_analysis.cts_full_wave import solve_3d_periodic_full_wave
from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def _uniform_lossy_case():
    x = np.linspace(0.0, 0.02, 3)
    y = np.linspace(0.0, 0.02, 3)
    z = np.linspace(0.0, 0.02, 3)
    omega = 2.0 * np.pi * 5.0e9
    epsilon = np.eye(3, dtype=complex) * (1.0 + 0.2j)
    current = np.zeros((z.size, y.size, x.size, 3), dtype=complex)
    current[..., 0] = 1.0
    return omega, x, y, z, epsilon, current


def test_3d_periodic_maxwell_reports_h_and_b_with_correct_units():
    omega, x, y, z, epsilon, current = _uniform_lossy_case()
    result = solve_3d_periodic_full_wave(
        omega, x, y, z, epsilon,
        impressed_current_density_A_m2=current,
    )
    np.testing.assert_allclose(
        result["magnetic_flux_density_T"],
        4.0e-7 * np.pi * result["magnetic_field_intensity_A_m"],
        rtol=1e-12, atol=1e-18,
    )
    np.testing.assert_allclose(
        result["magnetic_field_A_m"],
        result["magnetic_field_intensity_A_m"],
        rtol=0.0, atol=0.0,
    )


def test_uniform_lossy_periodic_cell_closes_poynting_energy_balance():
    omega, x, y, z, epsilon, current = _uniform_lossy_case()
    result = solve_3d_periodic_full_wave(
        omega, x, y, z, epsilon,
        impressed_current_density_A_m2=current,
    )
    assert result["integrated_absorbed_power_W"] > 0.0
    assert result["integrated_impressed_source_power_W"] > 0.0
    np.testing.assert_allclose(
        result["integrated_absorbed_power_W"],
        result["integrated_impressed_source_power_W"],
        rtol=2e-8, atol=1e-18,
    )
    assert result["global_energy_balance_relative_residual"] < 2e-8
    np.testing.assert_allclose(
        result["time_averaged_poynting_vector_W_m2"], 0.0,
        atol=1e-14, rtol=0.0,
    )


def test_3d_periodic_maxwell_is_executable_through_public_analysis_dispatcher():
    omega, x, y, z, epsilon, current = _uniform_lossy_case()
    result = run_advanced_thomson_physics(
        "periodic_3d_full_wave", omega, x, y, z, epsilon,
        impressed_current_density_A_m2=current,
    )
    assert result["public_physics_entrypoint"] == "periodic_3d_full_wave"
    assert result["magnetic_flux_density_T"].shape == (z.size, y.size, x.size, 3)
    assert "global_energy_balance_relative_residual" in result

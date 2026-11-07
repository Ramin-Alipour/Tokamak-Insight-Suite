import numpy as np
import pytest

import tokamak_analysis.cts_full_physics as physics


def _fake_full_em_result(level_value=1.0):
    eye = np.eye(3, dtype=complex)[None, :, :] * level_value
    scalar = np.ones((1, 4), dtype=float) * level_value
    return {
        "constitutive_response": {
            "conductivity_tensor_S_m": eye,
            "current_fluctuation_spectral_matrix": eye,
        },
        "electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s": eye,
        "magnetic_field_fluctuation_spectral_matrix_T2_per_rad_s": eye,
        "transverse_stokes_spectral_density_IQUV": scalar,
        "quantitative_release_gate": {
            "velocity_grid_convergence_required": True,
            "velocity_domain_convergence_required": True,
            "frequency_resolution_convergence_required": True,
            "magnetized_response_required_for_magnetized_plasma": True,
            "experimental_validation_performed": False,
        },
        "cts_three_wave_vertex_connected": False,
    }


def test_grid_convergence_solver_compares_all_physical_spectral_tensors(monkeypatch):
    calls = []

    def fake_solver(omega, x, y, z, species, **kwargs):
        calls.append((len(x), len(y), len(z)))
        return _fake_full_em_result()

    monkeypatch.setattr(physics, "analyze_landau_fdt_full_em_spectrum", fake_solver)
    result = physics.analyze_landau_fdt_full_em_spectrum_converged(
        [1.0e5], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0],
        [{"name": "e", "density_m3": 1.0e18, "mass_kg": 9.109e-31, "charge_C": -1.602e-19}],
        common_temperature_J=1.0e-17, wave_number_m_inverse=1.0e-3,
        convergence_tolerance=1.0e-12, max_refinements=1,
    )
    assert calls == [(3, 3, 3), (5, 5, 5)]
    assert result["velocity_grid_convergence"]["converged"] is True
    assert result["reportable_as_grid_converged"] is True
    assert set(result["velocity_grid_convergence"]["history"][-1]["relative_changes"]) == {
        "conductivity_tensor_S_m",
        "current_fluctuation_spectral_matrix",
        "electric_field_fluctuation_spectral_matrix_V2_m2_per_rad_s",
        "magnetic_field_fluctuation_spectral_matrix_T2_per_rad_s",
        "transverse_stokes_spectral_density_IQUV",
    }
    assert result["quantitative_release_gate"]["velocity_domain_convergence_required"] is True
    assert result["quantitative_release_gate"]["frequency_resolution_convergence_required"] is True


def test_grid_convergence_solver_withholds_convergence_when_node_budget_stops_refinement(monkeypatch):
    monkeypatch.setattr(physics, "analyze_landau_fdt_full_em_spectrum", lambda *a, **k: _fake_full_em_result())
    result = physics.analyze_landau_fdt_full_em_spectrum_converged(
        [1.0e5], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0],
        [{"name": "e", "density_m3": 1.0e18, "mass_kg": 9.109e-31, "charge_C": -1.602e-19}],
        common_temperature_J=1.0e-17, wave_number_m_inverse=1.0e-3,
        max_refinements=2, max_velocity_nodes=27,
    )
    assert result["velocity_grid_convergence"]["status"] == "not_converged"
    assert result["velocity_grid_convergence"]["limiting_reason"] == "max_velocity_nodes_reached"
    assert result["reportable_as_grid_converged"] is False
    assert result["quantitative_release_gate"]["velocity_grid_convergence_required"] is True


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_grid_convergence_solver_rejects_invalid_tolerance(bad, monkeypatch):
    monkeypatch.setattr(physics, "analyze_landau_fdt_full_em_spectrum", lambda *a, **k: _fake_full_em_result())
    with pytest.raises(ValueError, match="convergence_tolerance"):
        physics.analyze_landau_fdt_full_em_spectrum_converged(
            [1.0e5], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0], [-2.0, 0.0, 2.0],
            [{"name": "e", "density_m3": 1.0e18, "mass_kg": 9.109e-31, "charge_C": -1.602e-19}],
            common_temperature_J=1.0e-17, wave_number_m_inverse=1.0e-3,
            convergence_tolerance=bad,
        )


def test_converged_solver_is_connected_to_public_dispatcher_and_dashboard(monkeypatch):
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    model = "landau_fdt_full_em_spectrum_converged"
    assert model in MODEL_LABELS
    assert model in EXAMPLES
    monkeypatch.setattr(physics, "analyze_landau_fdt_full_em_spectrum", lambda *a, **k: _fake_full_em_result())
    config = dict(EXAMPLES[model])
    result = run_advanced_thomson_physics(model, **config)
    assert result["public_physics_entrypoint"] == model
    assert result["velocity_grid_convergence"]["status"] == "converged"
    assert result["reportable_as_grid_converged"] is True
    assert result["cts_three_wave_vertex_connected"] is False

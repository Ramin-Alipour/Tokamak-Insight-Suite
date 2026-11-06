"""Public dispatcher integration tests for physically implemented Thomson/CTS solvers."""

import numpy as np

import tokamak_analysis.cts_full_physics as physics


def test_multiview_3d_evdf_is_reachable_through_public_dispatcher(monkeypatch):
    captured = {}

    def fake_solver(**kwargs):
        captured.update(kwargs)
        return {"status": "identified_from_measurement", "data_only_rank": 3}

    monkeypatch.setattr(physics, "reconstruct_multiview_3d_evdf", fake_solver)
    result = physics.run_advanced_thomson_physics(
        "thomson_multiview_3d_evdf", measured_views=[1, 2, 3],
        velocity_grid_m_s=[-1.0, 0.0, 1.0],
    )
    assert captured == {"measured_views": [1, 2, 3], "velocity_grid_m_s": [-1.0, 0.0, 1.0]}
    assert result["status"] == "identified_from_measurement"
    assert result["public_physics_entrypoint"] == "thomson_multiview_3d_evdf"
    assert result["automatic_model_substitution"] is False


def test_full_em_response_name_reaches_landau_fdt_constitutive_solver(monkeypatch):
    captured = {}

    def fake_solver(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {"response_tensor": np.eye(3), "fdt_noise": np.eye(3)}

    monkeypatch.setattr(physics, "analyze_landau_fdt_full_em_constitutive", fake_solver)
    args = ([0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [{"name": "D+"}])
    result = physics.run_advanced_thomson_physics(
        "landau_fdt_full_em_response", *args, common_temperature_J=1.0,
        wave_number_m_inverse=1.0,
    )
    assert captured["args"] == args
    assert captured["kwargs"]["common_temperature_J"] == 1.0
    assert result["response_tensor"].shape == (3, 3)
    assert result["public_physics_entrypoint"] == "landau_fdt_full_em_response"


def test_path_averaged_thomson_density_is_reachable_through_public_dispatcher():
    path = np.linspace(0.0, 0.2, 21)
    result = physics.run_advanced_thomson_physics(
        "thomson_path_averaged_density",
        1.0e-12, path, np.ones(path.size), np.full(path.size, 6.0e-29),
        np.full(path.size, 0.02), np.full(path.size, 0.5),
    )
    assert result["electron_density_path_averaged_m3"] > 0.0
    assert result["public_physics_entrypoint"] == "thomson_path_averaged_density"
    assert result["automatic_model_substitution"] is False

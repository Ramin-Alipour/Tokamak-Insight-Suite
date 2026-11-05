"""Regression tests for public advanced Thomson/Full-EM solver dispatch."""
from __future__ import annotations

import tokamak_analysis.cts_full_physics as physics


def test_full_em_landau_response_is_reachable_through_public_dispatch(monkeypatch):
    sentinel = {"response": "cartesian_full_em"}
    calls = {}

    def fake_solver(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(physics, "landau_full_em_current_response", fake_solver)
    result = physics.run_advanced_thomson_physics(
        "landau_fdt_full_em_response",
        system="landau-system",
        angular_frequency_rad_s=[1.0, 2.0],
        species=["e", "D"],
        common_temperature_J=2.0,
        wave_number_m_inverse=12.0,
    )

    assert result["response"] == "cartesian_full_em"
    assert result["public_physics_entrypoint"] == "landau_fdt_full_em_response"
    assert calls == {
        "system": "landau-system",
        "angular_frequency_rad_s": [1.0, 2.0],
        "species": ["e", "D"],
        "common_temperature_J": 2.0,
        "wave_number_m_inverse": 12.0,
    }


def test_multiview_3d_evdf_is_reachable_through_public_dispatch(monkeypatch):
    sentinel = {"status": "identified_from_multi_view_measurements"}
    calls = {}

    def fake_solver(**kwargs):
        calls.update(kwargs)
        return sentinel

    monkeypatch.setattr(physics, "reconstruct_multiview_3d_evdf", fake_solver)
    result = physics.run_advanced_thomson_physics(
        "thomson_multiview_3d_evdf",
        spectral_kernels="kernels",
        measured_spectra="spectra",
        velocity_grid_m_s="velocity-grid",
        voxel_volume_m3_s3="voxel-volumes",
        regularization_matrix="regularizer",
        regularization_strength=0.01,
    )

    assert result["status"] == "identified_from_multi_view_measurements"
    assert result["public_physics_entrypoint"] == "thomson_multiview_3d_evdf"
    assert calls == {
        "spectral_kernels": "kernels",
        "measured_spectra": "spectra",
        "velocity_grid_m_s": "velocity-grid",
        "voxel_volume_m3_s3": "voxel-volumes",
        "regularization_matrix": "regularizer",
        "regularization_strength": 0.01,
    }

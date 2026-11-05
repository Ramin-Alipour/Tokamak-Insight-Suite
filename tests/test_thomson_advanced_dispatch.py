"""Regression tests for public advanced Thomson/Full-EM solver dispatch."""
from __future__ import annotations

import tokamak_analysis.cts_full_physics as physics


def test_full_em_landau_response_is_reachable_through_public_dispatch(monkeypatch):
    sentinel = {"response": "cartesian_full_em"}
    calls = {}

    def fake_solver(system, omega, **kwargs):
        calls.update(system=system, omega=omega, kwargs=kwargs)
        return sentinel

    monkeypatch.setattr(physics, "landau_full_em_current_response", fake_solver)
    result = physics.run_advanced_thomson_physics(
        "landau_fdt_full_em_response",
        "landau-system",
        [1.0, 2.0],
        species=["e", "D"],
        wave_number_m_inverse=12.0,
    )

    assert result is sentinel
    assert calls == {
        "system": "landau-system",
        "omega": [1.0, 2.0],
        "kwargs": {"species": ["e", "D"], "wave_number_m_inverse": 12.0},
    }


def test_multiview_3d_evdf_is_reachable_through_public_dispatch(monkeypatch):
    sentinel = {"status": "identified_from_multi_view_measurements"}
    calls = {}

    def fake_solver(kernels, spectra, velocity_grid, **kwargs):
        calls.update(
            kernels=kernels,
            spectra=spectra,
            velocity_grid=velocity_grid,
            kwargs=kwargs,
        )
        return sentinel

    monkeypatch.setattr(physics, "reconstruct_multiview_3d_evdf", fake_solver)
    result = physics.run_advanced_thomson_physics(
        "thomson_multiview_3d_evdf",
        "kernels",
        "spectra",
        "velocity-grid",
        voxel_volume_m3_s3="voxel-volumes",
        regularization_matrix="regularizer",
        regularization_strength=0.01,
    )

    assert result is sentinel
    assert calls == {
        "kernels": "kernels",
        "spectra": "spectra",
        "velocity_grid": "velocity-grid",
        "kwargs": {
            "voxel_volume_m3_s3": "voxel-volumes",
            "regularization_matrix": "regularizer",
            "regularization_strength": 0.01,
        },
    }

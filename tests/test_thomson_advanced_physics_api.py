"""Public dispatcher contract for advanced Thomson/CTS physics solvers."""
import numpy as np
import pytest

import tokamak_analysis.cts_full_physics as physics


@pytest.mark.parametrize(
    "model, target",
    [
        ("landau_fdt_cts", "analyze_landau_fdt_cts"),
        ("polarized_kinetic_ece", "solve_polarized_kinetic_ece_with_mode_conversion"),
        ("stratified_nonparaxial_vector", "propagate_stratified_nonparaxial_vector_field"),
        ("stratified_mode_conversion", "solve_stratified_mode_conversion"),
    ],
)
def test_advanced_physics_dispatch_is_explicit_and_preserves_solver_result(
    monkeypatch, model, target
):
    sentinel = {"solver_value": 17, "limitations": ["explicit limitation"]}
    calls = []

    def fake_solver(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(physics, target, fake_solver)
    result = physics.run_advanced_thomson_physics(model, supplied_input=3)

    assert calls == [{"supplied_input": 3}]
    assert result["solver_value"] == 17
    assert result["limitations"] == ["explicit limitation"]
    assert result["public_physics_entrypoint"] == model
    assert result["automatic_model_substitution"] is False


def test_advanced_physics_dispatch_rejects_unknown_models_without_fallback():
    with pytest.raises(ValueError, match="no automatic model substitution"):
        physics.run_advanced_thomson_physics("approximate_fallback")


def test_advanced_physics_dispatch_rejects_non_mapping_solver_result(monkeypatch):
    monkeypatch.setattr(
        physics, "analyze_landau_fdt_cts", lambda **kwargs: np.asarray([1.0])
    )
    with pytest.raises(TypeError, match="must return a mapping"):
        physics.run_advanced_thomson_physics("landau_fdt_cts")


def test_projected_evdf_dispatch_forwards_compton_controls(monkeypatch):
    calls = []
    sentinel = {"recoil_included": True, "model": "Compton"}

    def fake_inverse(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        physics, "reconstruct_relativistic_projected_thomson_evdf", fake_inverse
    )
    result = physics.run_advanced_thomson_physics(
        "thomson_relativistic_projected_evdf",
        include_compton_recoil=True,
        incident_polarization=[0.0, 0.0, 1.0],
        maximum_recoil_correction_fraction=0.01,
    )
    assert calls == [{
        "include_compton_recoil": True,
        "incident_polarization": [0.0, 0.0, 1.0],
        "maximum_recoil_correction_fraction": 0.01,
    }]
    assert result["recoil_included"] is True
    assert result["public_physics_entrypoint"] == "thomson_relativistic_projected_evdf"
    assert result["automatic_model_substitution"] is False

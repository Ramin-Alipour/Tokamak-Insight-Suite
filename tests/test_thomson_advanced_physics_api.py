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


def test_tabulated_evdf_landau_cts_dispatch_is_public_and_preserves_solver(monkeypatch):
    calls = []
    sentinel = {"dielectric_function": np.asarray([1.0 + 0.2j])}

    def fake_solver(**kwargs):
        calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(
        physics, "collective_tabulated_electron_spectral_density", fake_solver
    )
    supplied = {
        "wavelength_nm": [499.9, 500.0, 500.1],
        "electron_density_m3": 1.0e19,
        "electron_velocity_grid_m_s": [-1.0, 0.0, 1.0],
    }
    result = physics.run_advanced_thomson_physics(
        "cts_tabulated_evdf_landau", **supplied
    )
    assert calls == [supplied]
    assert result["dielectric_function"][0] == 1.0 + 0.2j
    assert result["public_physics_entrypoint"] == "cts_tabulated_evdf_landau"
    assert result["automatic_model_substitution"] is False


def test_collective_kappa_cts_is_exposed_through_public_dispatch(monkeypatch):
    sentinel = {"spectral_density_per_nm": np.asarray([1.0, 2.0]), "model": "collective Kappa CTS"}
    calls = []
    def fake_solver(**kwargs):
        calls.append(kwargs)
        return sentinel
    monkeypatch.setattr(physics, "collective_kappa_spectral_density", fake_solver)
    result = physics.run_advanced_thomson_physics("cts_collective_kappa_spectrum", wavelength_nm=[499.9, 500.1], kappa=4.0)
    assert calls == [{"wavelength_nm": [499.9, 500.1], "kappa": 4.0}]
    assert result["spectral_density_per_nm"] is sentinel["spectral_density_per_nm"]
    assert result["public_physics_entrypoint"] == "cts_collective_kappa_spectrum"
    assert result["automatic_model_substitution"] is False


def test_collective_kappa_cts_inference_is_exposed_through_public_dispatch(monkeypatch):
    sentinel = {"electron_density_m3": 2.0e18, "kappa_index": 4.0}
    calls = []
    def fake_solver(**kwargs):
        calls.append(kwargs)
        return sentinel
    monkeypatch.setattr(physics, "fit_collective_kappa", fake_solver)
    result = physics.run_advanced_thomson_physics("cts_collective_kappa_inference", signal=[1.0, 2.0])
    assert calls == [{"signal": [1.0, 2.0]}]
    assert result["electron_density_m3"] == 2.0e18
    assert result["public_physics_entrypoint"] == "cts_collective_kappa_inference"

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


def test_collective_kappa_dispatch_forwards_mixed_kinetic_populations():
    wavelength = np.linspace(1048.0, 1080.0, 41)
    angle = np.pi / 2.0
    result = physics.run_advanced_thomson_physics(
        "cts_collective_kappa_spectrum",
        wavelength_nm=wavelength,
        laser_wavelength_nm=1064.0,
        scattering_angle_rad=angle,
        electron_density_m3=1.0e19,
        electron_temperature_eV=150.0,
        kappa=4.0,
        ion_species=[{"name": "D+", "charge_state": 1, "mass_amu": 2.014}],
        ion_fractions=[1.0],
        ion_temperature_eV=[100.0],
        electron_velocity_m_s=[0.0, 0.0, 0.0],
        ion_velocity_m_s=[[0.0, 0.0, 0.0]],
        electron_populations=[
            {"density_fraction": 0.8, "temperature_ratio": 0.7, "kappa": 5.0},
            {"density_fraction": 0.2, "temperature_ratio": 3.0, "distribution": "maxwellian"},
        ],
        ion_populations=[{"distribution": "kappa", "kappa": 3.0}],
        incident_direction=[-np.sin(angle / 2.0), np.cos(angle / 2.0), 0.0],
        scattered_direction=[np.sin(angle / 2.0), np.cos(angle / 2.0), 0.0],
    )
    assert result["public_physics_entrypoint"] == "cts_collective_kappa_spectrum"
    assert result["electron_populations"][1]["distribution"] == "maxwellian"
    assert result["ion_populations"][0]["distribution"] == "kappa"
    assert result["spectral_density_per_nm"].shape == wavelength.shape
    assert np.all(np.isfinite(result["dielectric_function"]))

def test_gyrotropic_public_api_closes_multi_view_projections_to_3d_flow():
    from tokamak_analysis.thomson_anisotropic_inference import _gaussian_bin_average

    wavelength = np.linspace(1062.0, 1066.0, 161)
    laser = 1064.0
    angle = np.pi / 2.0
    magnetic = np.array([0.0, 0.0, 1.0])
    directions = np.array([
        [0.0, 0.0, 1.0],
        [np.sqrt(3.0) / 2.0, 0.0, 0.5],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    true_flow = np.array([1.2e4, -8.0e3, 1.5e4])
    projections = directions @ true_flow
    pitch2 = (directions @ magnetic) ** 2
    directional_temperature = 350.0 + (900.0 - 350.0) * pitch2
    spectra = []
    for temperature, flow in zip(directional_temperature, projections):
        sigma_v = np.sqrt(1.602176634e-19 * temperature / 9.1093837015e-31)
        center = laser * (1.0 - 2.0*np.sin(angle/2.0)*flow/299792458.0)
        sigma_lambda = laser * 2.0*np.sin(angle/2.0)*sigma_v/299792458.0
        shape, _ = _gaussian_bin_average(wavelength, center, sigma_lambda)
        spectra.append(0.03 + 0.8*shape + 1.0e-4*(wavelength-laser))

    result = physics.run_advanced_thomson_physics(
        "gyrotropic_bi_maxwellian_spectral_fit",
        wavelength, np.asarray(spectra),
        laser_wavelength_nm=laser,
        scattering_angle_rad=angle,
        magnetic_field_unit=magnetic,
        scattering_vector_unit=directions,
        instrument_response_matrix=np.eye(wavelength.size),
        temperature_bounds_eV=(20.0, 3000.0),
        flow_bounds_m_s=(-1.0e6, 1.0e6),
    )
    flow = result["three_dimensional_flow_inference"]
    assert flow["identifiability"] == "identifiable_from_three_dimensional_view_geometry"
    assert flow["reportable_as_inferred_parameter"] is True
    np.testing.assert_allclose(flow["electron_velocity_vector_m_s"], true_flow, atol=2.0e3)
    assert flow["electron_velocity_covariance_m2_s2"].shape == (3, 3)
    assert result["public_physics_entrypoint"] == "gyrotropic_bi_maxwellian_spectral_fit"


def test_gyrotropic_public_api_does_not_invent_3d_flow_when_covariance_is_missing(monkeypatch):
    sentinel = {
        "electron_velocity_projection_m_s": np.array([1.0, 2.0, 3.0]),
        "parameter_order": [
            "T_parallel_eV", "T_perpendicular_eV",
            "flow_projection_view_0_m_s", "log_amplitude_view_0",
            "background_offset_view_0", "background_slope_view_0_per_nm",
            "flow_projection_view_1_m_s", "log_amplitude_view_1",
            "background_offset_view_1", "background_slope_view_1_per_nm",
            "flow_projection_view_2_m_s", "log_amplitude_view_2",
            "background_offset_view_2", "background_slope_view_2_per_nm",
        ],
        "parameter_covariance": None,
    }
    monkeypatch.setattr(physics, "fit_gyrotropic_bi_maxwellian_thomson", lambda *a, **k: sentinel)
    result = physics.run_advanced_thomson_physics(
        "gyrotropic_bi_maxwellian_spectral_fit",
        scattering_vector_unit=np.eye(3),
    )
    flow = result["three_dimensional_flow_inference"]
    assert flow["reportable_as_inferred_parameter"] is False
    assert flow["electron_velocity_vector_m_s"] is None
    assert "covariance is required" in flow["reason"]

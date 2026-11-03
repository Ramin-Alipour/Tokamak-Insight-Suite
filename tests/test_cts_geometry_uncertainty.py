import numpy as np
import pytest

from tokamak_analysis.cts_geometry_uncertainty import (
    propagate_full_em_cts_ray_geometry_uncertainty,
)


PARAMETER_NAMES = [
    "electron_density_m3",
    "electron_temperature_eV",
    "electron_flow_projection_m_s",
    "ion_temperature_eV:D",
    "ion_flow_projection_m_s:D",
]


def _geometry(scale=1.0):
    sample = {
        "incident_direction": [1.0, 0.0, 0.0],
        "scattered_direction": [0.0, 1.0, 0.0],
        "magnetic_field_T": [0.0, 0.0, 1.0],
        "scattering_location_m": [scale, 0.0, 0.0],
        "sample_weight": 1.0,
        "beam_overlap_m_minus3": 0.5 * scale,
        "path_length_weight_m": 0.1,
    }
    return {
        "scale": scale,
        "frequency_resolved_geometry": [
            {"scattering_samples": [sample], "beam_overlap_relative": 1.0}
        ],
    }


def test_geometry_uncertainty_refits_and_returns_full_covariance(monkeypatch):
    import tokamak_analysis.cts_full_em as full_em

    def fake_analyze(**kwargs):
        scale = float(kwargs["scattering_geometry"].get("scale", 1.0))
        return {
            "optimizer_success": True,
            "parameter_names": PARAMETER_NAMES,
            "electron_density_m3": 1.0e19 * scale,
            "electron_temperature_eV": 100.0 + scale,
            "electron_flow_projection_m_s": 1.0e3 * scale,
            "ion_temperature_eV_by_species": {"D": 20.0 + scale},
            "ion_flow_projection_m_s_by_species": {"D": 2.0e2 * scale},
            "conditional_fit_covariance": np.eye(5),
            "instrument_response_parameter_covariance": 0.5 * np.eye(5),
            "fit_covariance": 1.5 * np.eye(5),
        }

    monkeypatch.setattr(full_em, "analyze_full_em_cts_spectrum", fake_analyze)
    counter = {"value": 0}

    def sampler(rng, index):
        counter["value"] += 1
        scale = 0.9 + 0.2 * rng.random()
        return {"scattering_geometry": _geometry(scale)}

    result = propagate_full_em_cts_ray_geometry_uncertainty(
        inference_kwargs={"scattering_geometry": {"scale": 1.0}},
        geometry_sampler=sampler,
        samples=20,
        seed=17,
    )

    assert counter["value"] == 20
    assert result["ray_geometry_uncertainty"]["propagated"] is True
    assert result["ray_geometry_parameter_covariance"].shape == (5, 5)
    assert np.all(np.isfinite(result["fit_covariance"]))
    assert np.all(np.diag(result["fit_covariance"]) >= 1.5)


def test_geometry_uncertainty_rejects_local_mean_only(monkeypatch):
    import tokamak_analysis.cts_full_em as full_em

    def fake_analyze(**kwargs):
        return {
            "optimizer_success": True,
            "parameter_names": PARAMETER_NAMES,
            "electron_density_m3": 1.0e19,
            "electron_temperature_eV": 100.0,
            "electron_flow_projection_m_s": 0.0,
            "ion_temperature_eV_by_species": {"D": 20.0},
            "ion_flow_projection_m_s_by_species": {"D": 0.0},
            "conditional_fit_covariance": np.eye(5),
            "fit_covariance": np.eye(5),
        }

    monkeypatch.setattr(full_em, "analyze_full_em_cts_spectrum", fake_analyze)

    result = propagate_full_em_cts_ray_geometry_uncertainty(
        inference_kwargs={"scattering_geometry": {"scale": 1.0}},
        geometry_sampler=lambda rng, index: {"scattering_geometry": {"incident_direction": [1, 0, 0]}},
        samples=20,
    )
    assert result["ray_geometry_uncertainty"]["propagated"] is False
    assert result["ray_geometry_uncertainty"]["failed_samples"] == 20


def test_geometry_uncertainty_requires_sampler_and_sufficient_draws():
    with pytest.raises(ValueError, match="geometry_sampler"):
        propagate_full_em_cts_ray_geometry_uncertainty(
            inference_kwargs={"scattering_geometry": {}},
            geometry_sampler=None,
            samples=20,
        )
    with pytest.raises(ValueError, match="samples"):
        propagate_full_em_cts_ray_geometry_uncertainty(
            inference_kwargs={"scattering_geometry": {}},
            geometry_sampler=lambda rng, index: {},
            samples=5,
        )


def test_joint_geometry_sampler_applies_correlated_external_input_updates(monkeypatch):
    import tokamak_analysis.cts_full_em as full_em

    def fake_analyze(**kwargs):
        scale = float(kwargs["scattering_geometry"].get("scale", 1.0))
        external = float(kwargs["probe_frequency_Hz"]) / 1.0e9
        value = scale + external
        return {
            "optimizer_success": True,
            "parameter_names": PARAMETER_NAMES,
            "electron_density_m3": 1.0e19 * value,
            "electron_temperature_eV": 100.0 + value,
            "electron_flow_projection_m_s": 1.0e3 * value,
            "ion_temperature_eV_by_species": {"D": 20.0 + value},
            "ion_flow_projection_m_s_by_species": {"D": 2.0e2 * value},
            "conditional_fit_covariance": np.eye(5),
            "fit_covariance": np.eye(5),
        }

    monkeypatch.setattr(full_em, "analyze_full_em_cts_spectrum", fake_analyze)

    def joint_sampler(rng, index):
        scale = 0.9 + 0.2 * rng.random()
        # Deliberately correlate the external probe-frequency draw with
        # geometry. Both must enter the same forward refit.
        return {
            "scattering_geometry": _geometry(scale),
            "inference_updates": {"probe_frequency_Hz": 1.0e9 * (2.0 * scale)},
        }

    result = propagate_full_em_cts_ray_geometry_uncertainty(
        inference_kwargs={
            "scattering_geometry": {"scale": 1.0},
            "probe_frequency_Hz": 1.0e9,
        },
        geometry_sampler=joint_sampler,
        samples=20,
        seed=31,
    )
    assert result["ray_geometry_uncertainty"]["propagated"] is True
    assert "joint_full_forward_monte_carlo" in result["ray_geometry_uncertainty"]["method"]
    assert result["ray_geometry_parameter_covariance"][0, 1] > 0.0


def test_joint_geometry_sampler_cannot_replace_measured_spectrum(monkeypatch):
    import tokamak_analysis.cts_full_em as full_em

    def fake_analyze(**kwargs):
        return {
            "optimizer_success": True,
            "parameter_names": PARAMETER_NAMES,
            "electron_density_m3": 1.0e19,
            "electron_temperature_eV": 100.0,
            "electron_flow_projection_m_s": 0.0,
            "ion_temperature_eV_by_species": {"D": 20.0},
            "ion_flow_projection_m_s_by_species": {"D": 0.0},
            "conditional_fit_covariance": np.eye(5),
            "fit_covariance": np.eye(5),
        }

    monkeypatch.setattr(full_em, "analyze_full_em_cts_spectrum", fake_analyze)
    result = propagate_full_em_cts_ray_geometry_uncertainty(
        inference_kwargs={"scattering_geometry": {"scale": 1.0}, "measured_spectrum": [1.0]},
        geometry_sampler=lambda rng, index: {
            "scattering_geometry": _geometry(),
            "inference_updates": {"measured_spectrum": [2.0]},
        },
        samples=20,
    )
    assert result["ray_geometry_uncertainty"]["propagated"] is False
    assert result["ray_geometry_uncertainty"]["failed_samples"] == 20



def test_cold_plasma_ray_traces_supplied_paraxial_beam_covariance():
    from tokamak_analysis.cts_em_geometry import trace_cold_plasma_ray
    from tokamak_analysis.advanced_inference import propagate_paraxial_beam_covariance

    electron_charge = -1.602176634e-19
    electron_mass = 9.1093837015e-31
    ion_charge = 1.602176634e-19
    ion_mass = 2.0*1.66053906660e-27
    def equilibrium(position):
        return {
            "magnetic_field_T": np.array([0.0, 0.0, 2.0]),
            "species": [
                {"density_m3": 1.0e12, "charge_C": electron_charge, "mass_kg": electron_mass},
                {"density_m3": 1.0e12, "charge_C": ion_charge, "mass_kg": ion_mass},
            ],
        }
    free_space = np.array([[1.0,0.0,0.1,0.0],[0.0,1.0,0.0,0.1],
                           [0.0,0.0,1.0,0.0],[0.0,0.0,0.0,1.0]])
    sigma0 = np.diag([1e-6, 1e-6, 1e-4, 1e-4])
    result = trace_cold_plasma_ray(
        [0.0,0.0,0.0], [1000.0,0.0,0.0], 2*np.pi*100e9,
        equilibrium=equilibrium, mode_index=0, path_parameter_max=1.0,
        sample_count=4, initial_beam_phase_space_covariance=sigma0,
        paraxial_transfer_matrices=np.repeat(free_space[None,:,:], 3, axis=0),
    )
    assert result["paraxial_beam_propagation_status"] == "propagated_from_supplied_symplectic_ABCD_maps"
    assert result["transverse_position_covariance_m2"].shape == (4,2,2)
    assert np.all(np.linalg.eigvalsh(result["transverse_position_covariance_m2"]) > 0)



def test_cold_plasma_ray_propagates_full_6d_covariance_from_dispersion_hessian():
    from tokamak_analysis.cts_em_geometry import trace_cold_plasma_ray

    electron_charge = -1.602176634e-19
    electron_mass = 9.1093837015e-31
    ion_charge = 1.602176634e-19
    ion_mass = 2.0*1.66053906660e-27
    def equilibrium(position):
        return {
            "magnetic_field_T": np.array([0.0, 0.0, 2.0]),
            "species": [
                {"density_m3": 1.0e12, "charge_C": electron_charge, "mass_kg": electron_mass},
                {"density_m3": 1.0e12, "charge_C": ion_charge, "mass_kg": ion_mass},
            ],
        }
    sigma0 = np.diag([1e-6, 1e-6, 1e-6, 1e-3, 1e-3, 1e-3])
    result = trace_cold_plasma_ray(
        [0.0,0.0,0.0], [1000.0,0.0,0.0], 2*np.pi*100e9,
        equilibrium=equilibrium, mode_index=0, path_parameter_max=1.0,
        sample_count=4, initial_ray_phase_space_covariance=sigma0,
    )
    assert result["paraxial_beam_propagation_status"] == "propagated_from_full_cold_plasma_dispersion_Hessian"
    assert result["phase_space_covariance"].shape == (4,6,6)
    assert result["transverse_position_covariance_m2"].shape == (4,2,2)
    assert np.all(np.linalg.eigvalsh(result["transverse_position_covariance_m2"]) > 0)



def test_full_em_geometry_ensemble_is_exposed_by_dashboard():
    from pathlib import Path

    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "cts_full_geometry_ensemble_upload" in app
    assert "cts_full_geometry_mc_samples" in app
    assert "propagate_full_em_cts_ray_geometry_uncertainty" in app
    assert "_sample_precomputed_geometry" in app

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

import numpy as np
import pytest

import tokamak_analysis.cts_spatial_inference as spatial


def _geometry(n_frequency=8):
    records = []
    for _ in range(n_frequency):
        samples = []
        for i, weight in enumerate((0.2, 0.5, 0.3)):
            samples.append({
                "sample_id": f"node-{i}", "sample_weight": weight,
                "incident_direction": [1.0, 0.0, 0.0],
                "scattered_direction": [0.0, 1.0, 0.0],
                "magnetic_field_T": [0.0, 0.0, 1.0],
                "scattering_location_m": [float(i), 0.0, 0.0],
            })
        records.append({"beam_overlap_relative": 1.0, "scattering_samples": samples})
    return {"frequency_resolved_geometry": records}


def _kwargs():
    return {
        "electron_density_m3": 2e19, "electron_temperature_eV": 1000.0,
        "ion_temperature_eV": 500.0,
        "ion_species": [{"name": "D", "charge_state": 1, "mass_amu": 2.0}],
        "ion_fractions": [1.0], "magnetic_field_T": [0.0, 0.0, 1.0],
        "incident_polarization": [0.0, 1.0, 0.0],
        "scattered_polarization": [1.0, 0.0, 0.0],
    }


def test_spatial_full_em_inverse_uses_local_profiles_and_full_covariance(monkeypatch):
    frequency = np.linspace(-3e6, 3e6, 8)
    truth_ne = np.array([0.8, 1.0, 1.2])
    truth_te = np.array([1.1, 1.0, 0.9])

    def fake_forward(frequency_shift_Hz, probe_frequency_Hz, geometry, **kwargs):
        values = np.zeros(len(frequency_shift_Hz))
        for j, record in enumerate(geometry["frequency_resolved_geometry"]):
            samples = record["scattering_samples"]
            values[j] = sum(
                s["sample_weight"] * (
                    2.0 * s["electron_density_relative"]
                    + 0.7 * s["electron_temperature_relative"]
                ) for s in samples
            ) * (1.0 + 0.02 * j)
        return {"spectral_density": values}

    monkeypatch.setattr(spatial, "full_em_cts_spectral_density", fake_forward)
    geometry = _geometry()
    for record in geometry["frequency_resolved_geometry"]:
        for i, sample in enumerate(record["scattering_samples"]):
            sample["electron_density_relative"] = truth_ne[i]
            sample["electron_temperature_relative"] = truth_te[i]
    measured = fake_forward(frequency, 140e9, geometry)["spectral_density"]
    # Start at a uniform profile; a weak smoothness prior keeps the inverse stable.
    result = spatial.analyze_full_em_cts_spatial_profiles(
        frequency, measured, probe_frequency_Hz=140e9,
        scattering_geometry=_geometry(), forward_model_kwargs=_kwargs(),
        spectral_covariance=np.eye(8) * 1e-4,
        fit_electron_density_profile=True,
        fit_electron_temperature_profile=True,
        initial_profiles={
            "electron_density_relative": np.ones(3),
            "electron_temperature_relative": np.ones(3),
        },
        regularization_strength=1e-8, max_nfev=100,
    )
    assert result["forward_model_evaluations_use_local_full_em_kinetics"] is True
    assert result["profiles_relative_to_reference"]["electron_density_relative"].shape == (3,)
    assert result["profiles_relative_to_reference"]["electron_temperature_relative"].shape == (3,)
    assert result["spectral_covariance"].shape == (8, 8)
    assert result["parameter_order"][0] == {
        "parameter": "electron_density_relative", "sample_id": "node-0"
    }


def test_spatial_full_em_inverse_rejects_unstable_node_identity():
    geometry = _geometry()
    geometry["frequency_resolved_geometry"][1]["scattering_samples"][0]["sample_id"] = "renamed"
    with pytest.raises(ValueError, match="same scattering sample_id set"):
        spatial._geometry_nodes(geometry, 8)


def test_spatial_full_em_inverse_requires_positive_regularization():
    with pytest.raises(ValueError, match="at least one positive"):
        spatial.analyze_full_em_cts_spatial_profiles(
            np.linspace(-1, 1, 8), np.ones(8), probe_frequency_Hz=140e9,
            scattering_geometry=_geometry(), forward_model_kwargs=_kwargs(),
            signal_uncertainty=np.ones(8), regularization_strength=0.0,
        )



def test_spatial_inverse_is_registered_in_dispatcher_and_dashboard():
    from tokamak_analysis.cts_analysis import analyze_cts_spectrum_model
    from tokamak_analysis.analysis_registry import get_analysis
    from pathlib import Path
    import inspect
    import tokamak_analysis.cts_spatial_inference as spatial

    assert callable(spatial.analyze_full_em_cts_spatial_profiles)
    assert "forward_model_kwargs" in inspect.signature(
        spatial.analyze_full_em_cts_spatial_profiles
    ).parameters
    contract = get_analysis(30)["execution_contract"]
    modes = {item["name"] for item in contract["workflow_modes"]}
    assert "spatially_varying_magnetized_full_electromagnetic_cts_profile_inverse" in modes
    app = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "full_em_spatial_profiles" in app
    assert "run_cts_spatial_profiles" in app
    assert callable(analyze_cts_spectrum_model)


def test_spatial_full_em_inverse_reports_regularization_choice_sensitivity(monkeypatch):
    frequency = np.linspace(-3e6, 3e6, 8)

    def fake_forward(frequency_shift_Hz, probe_frequency_Hz, geometry, **kwargs):
        values = []
        for j, record in enumerate(geometry["frequency_resolved_geometry"]):
            values.append(sum(
                sample["sample_weight"] * sample["electron_density_relative"]
                for sample in record["scattering_samples"]
            ) * (1.0 + 0.1*j))
        return {"spectral_density": np.asarray(values)}

    monkeypatch.setattr(spatial, "full_em_cts_spectral_density", fake_forward)
    result = spatial.analyze_full_em_cts_spatial_profiles(
        frequency, np.ones(8), probe_frequency_Hz=140e9,
        scattering_geometry=_geometry(), forward_model_kwargs=_kwargs(),
        signal_uncertainty=np.ones(8), fit_electron_density_profile=True,
        fit_electron_temperature_profile=False, regularization_strength=0.2,
        regularization_sensitivity_strengths=[0.1, 0.2, 1.0],
        max_nfev=40,
    )
    sensitivity = result["regularization_sensitivity"]
    assert sensitivity["regularization_strengths"].shape == (3,)
    assert sensitivity["profiles_relative_to_reference"]["electron_density_relative"].shape == (3, 3)
    assert sensitivity["measurement_covariance_unchanged"] is True
    assert "not measurement uncertainty" in sensitivity["interpretation"]

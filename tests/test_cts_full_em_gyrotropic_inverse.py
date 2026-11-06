import numpy as np

import tokamak_analysis.cts_full_em as full_em
from tokamak_analysis.cts_analysis import analyze_cts_spectrum_model


def test_full_em_gyrotropic_electron_shape_inverse_calls_relativistic_forward(monkeypatch):
    rng = np.random.default_rng(31)
    p_perp = np.linspace(1e-22, 4e-22, 4)
    p_parallel = np.linspace(-3e-22, 3e-22, 4)
    n_voxel = p_perp.size*p_parallel.size
    frequency = np.linspace(-2e9, 2e9, 24)
    kernel = rng.uniform(0.1, 1.0, size=(frequency.size, n_voxel))
    weights = 2*np.pi*p_perp[:,None]*np.gradient(p_perp)[:,None]*np.gradient(p_parallel)[None,:]
    truth = np.arange(1, n_voxel+1, dtype=float).reshape(4,4)
    truth /= np.sum(truth*weights)
    measured = 250.0*(kernel@truth.ravel()) + 0.03

    def fake_forward(frequency_shift_Hz, probe_frequency_Hz, scattering_geometry, **kwargs):
        distribution = kwargs["electron_distribution"]["distribution_kg_m_s_inverse3"]
        return {"spectral_density": kernel@np.asarray(distribution).ravel()}

    monkeypatch.setattr(full_em, "full_em_cts_spectral_density", fake_forward)
    result = analyze_cts_spectrum_model(
        model="full_em_gyrotropic_vdf_shape",
        frequency_shift_Hz=frequency, measured_spectrum=measured,
        probe_frequency_Hz=140e9,
        scattering_geometry={"incident_direction": [1,0,0], "scattered_direction": [0,1,0]},
        population_kind="electron",
        p_perpendicular_kg_m_s=p_perp,
        p_parallel_kg_m_s=p_parallel,
        initial_distribution_kg_m_s_inverse3=np.ones((4,4)),
        electron_density_m3=2e19, electron_temperature_eV=1000.0,
        ion_temperature_eV=500.0,
        ion_species=[{"name":"D+","charge_state":1,"mass_amu":2.014}],
        ion_fractions=[1.0], magnetic_field_T=[0,0,2.0],
        incident_polarization=[0,1,0], scattered_polarization=[1,0,0],
        instrument_response=np.eye(frequency.size),
        regularization_matrix=np.diff(np.eye(n_voxel), n=2, axis=0),
        regularization_strength=1e-5,
        electron_flow_projection_m_s=0.0,
        max_nfev=300,
    )
    assert result["population_kind"] == "electron"
    if result["distribution_kg_m_s_inverse3"] is not None:
        assert result["distribution_kg_m_s_inverse3"].shape == (4,4)
    assert result["regularized_candidate_distribution_kg_m_s_inverse3"].shape == (4,4)
    assert np.all(result["regularized_candidate_distribution_kg_m_s_inverse3"] >= 0)
    assert np.isclose(np.sum(result["regularized_candidate_distribution_kg_m_s_inverse3"]*weights), 1.0)
    assert result["bulk_plasma_and_geometry_fixed"] is True

    from tokamak_analysis.cts_full_em import C, ME
    candidate = result["regularized_candidate_distribution_kg_m_s_inverse3"]
    momentum = np.sqrt(p_perp[:, None]**2 + p_parallel[None, :]**2)
    gamma = np.sqrt(1.0 + (momentum / (ME*C))**2)
    velocity_parallel = p_parallel[None, :] / (gamma*ME)
    expected_candidate_mean = float(np.sum(candidate*weights*velocity_parallel))
    assert np.isclose(result["electron_parallel_mean_velocity_candidate_m_s"], expected_candidate_mean, rtol=1e-12, atol=1e-9)
    assert result["electron_parallel_velocity_definition"].startswith("integral(")
    if result["distribution_kg_m_s_inverse3"] is None:
        assert result["electron_parallel_mean_velocity_m_s"] is None
        assert result["electron_parallel_mean_velocity_status"] == "conditional_candidate_only_shape_not_identifiable"
    else:
        fitted = result["distribution_kg_m_s_inverse3"]
        expected_fitted_mean = float(np.sum(fitted*weights*velocity_parallel))
        assert np.isclose(result["electron_parallel_mean_velocity_m_s"], expected_fitted_mean, rtol=1e-12, atol=1e-9)
        assert result["electron_parallel_mean_velocity_status"] == "identified_from_fitted_gyrotropic_EVDF"

    from tokamak_analysis.cts_full_em import E_CHARGE
    velocity_perpendicular = p_perp[:, None] / (gamma*ME)
    kinetic_energy_eV = (gamma - 1.0)*ME*C**2/E_CHARGE
    expected_energy = float(np.sum(candidate*weights*kinetic_energy_eV))
    expected_t_parallel = float(np.sum(candidate*weights*p_parallel[None, :]*velocity_parallel)/E_CHARGE)
    expected_t_perpendicular = float(
        0.5*np.sum(candidate*weights*p_perp[:, None]*velocity_perpendicular)/E_CHARGE
    )
    moments = result["population_relativistic_moments_candidate"]
    assert np.isclose(moments["mean_kinetic_energy_eV"], expected_energy, rtol=1e-12)
    assert np.isclose(moments["parallel_pressure_temperature_eV"], expected_t_parallel, rtol=1e-12)
    assert np.isclose(moments["perpendicular_pressure_temperature_eV"], expected_t_perpendicular, rtol=1e-12)
    assert np.isclose(
        moments["pressure_anisotropy_Tperp_over_Tparallel"],
        expected_t_perpendicular/expected_t_parallel, rtol=1e-12,
    )
    assert np.isclose(moments["mean_parallel_velocity_m_s"], expected_candidate_mean, rtol=1e-12, atol=1e-9)
    if result["distribution_kg_m_s_inverse3"] is None:
        assert result["population_relativistic_moments_identified"] is None
        assert result["population_relativistic_moment_status"] == "conditional_candidate_only_shape_not_identifiable"
    else:
        assert result["population_relativistic_moments_identified"] is not None
        assert result["population_relativistic_moment_status"] == "identified_from_fitted_distribution"


def test_relativistic_gyrotropic_cts_inverse_is_connected_to_dashboard_and_registries():
    from pathlib import Path
    import json

    root = Path(__file__).resolve().parents[1]
    dashboard = (root / "app.py").read_text(encoding="utf-8")
    assert "'full_em_gyrotropic_vdf_shape'" in dashboard
    assert "analyze_cts_spectrum_model(model='full_em_gyrotropic_vdf_shape'" in dashboard
    assert "population_relativistic_moments_identified" in dashboard

    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_formula = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "cts_relativistic_gyrotropic_vdf_pressure_moments"
    assert formula == packaged_formula
    assert sum(item.get("execution_contract", {}).get("formula_key") == key for item in formula) == 1

    diagnostics = json.loads((root / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    packaged_diagnostics = json.loads((root / "tis" / "data" / "DIAGNOSTIC_REGISTRY.json").read_text(encoding="utf-8"))
    assert diagnostics == packaged_diagnostics
    cts = next(item for item in diagnostics if item["diagnostic"] == "Collective Thomson / coherent scattering (CTS)")
    output_names = {item["name"] for item in cts["execution_contract"]["output_specs"]}
    assert {
        "population_relativistic_moments_candidate",
        "population_relativistic_moments_identified",
        "population_relativistic_moment_status",
    } <= output_names

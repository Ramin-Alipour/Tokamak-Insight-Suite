import numpy as np

import tokamak_analysis.cts_full_em as full_em


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
    result = full_em.fit_full_em_gyrotropic_population_shape(
        frequency, measured,
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
    assert result["distribution_kg_m_s_inverse3"].shape == (4,4)
    assert np.all(result["regularized_candidate_distribution_kg_m_s_inverse3"] >= 0)
    assert np.isclose(np.sum(result["regularized_candidate_distribution_kg_m_s_inverse3"]*weights), 1.0)
    assert result["bulk_plasma_and_geometry_fixed"] is True

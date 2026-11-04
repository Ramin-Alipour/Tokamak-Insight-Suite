import numpy as np
import pytest

from tokamak_analysis.cts_stokes_transfer import (
    propagate_polarized_lte_stokes,
    stokes_mueller_absorption_matrix,
)


def test_polarized_lte_source_obeys_exact_kirchhoff_equilibrium_with_faraday_terms():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance
    frequency, temperature = 120e9, 2.5e7
    equilibrium = np.array([float(planck_spectral_radiance(frequency, temperature)), 0.0, 0.0, 0.0])
    result = propagate_polarized_lte_stokes(
        frequency, temperature, equilibrium,
        alpha_I_m_inverse=0.8, alpha_Q_m_inverse=0.2,
        alpha_U_m_inverse=-0.1, alpha_V_m_inverse=0.15,
        rho_Q_m_inverse=0.7, rho_U_m_inverse=-0.4,
        rho_V_m_inverse=1.1, path_length_m=0.35,
    )
    np.testing.assert_allclose(result["stokes_radiance"], equilibrium, rtol=2e-12, atol=0.0)
    assert result["kirchhoff_relation"] == "j_stokes = M_mueller @ [B_nu(T),0,0,0]"
    assert result["source_closure"] == "polarized_LTE_Planck_Kirchhoff"


def test_polarized_lte_emission_generates_dichroic_stokes_components_from_unpolarized_incident():
    result = propagate_polarized_lte_stokes(
        90e9, 1.0e7, [0.0, 0.0, 0.0, 0.0],
        alpha_I_m_inverse=1.0, alpha_Q_m_inverse=0.25,
        path_length_m=0.5,
    )
    stokes = result["stokes_radiance"]
    assert stokes[0] > 0.0 and stokes[1] > 0.0
    assert abs(stokes[2]) < 1e-14 * stokes[0]
    assert abs(stokes[3]) < 1e-14 * stokes[0]
    assert stokes[0] >= np.linalg.norm(stokes[1:])


def test_polarized_lte_scalar_limit_matches_unpolarized_planck_transfer():
    from tokamak_analysis.cts_em_background import planck_spectral_radiance
    frequency, temperature, alpha, length = 75e9, 8.0e6, 0.6, 0.9
    incident = 0.3 * float(planck_spectral_radiance(frequency, temperature))
    result = propagate_polarized_lte_stokes(
        frequency, temperature, [incident, 0.0, 0.0, 0.0],
        alpha_I_m_inverse=alpha, path_length_m=length,
    )
    source = float(planck_spectral_radiance(frequency, temperature))
    expected = source + (incident-source)*np.exp(-alpha*length)
    assert result["I"] == pytest.approx(expected, rel=2e-12)
    np.testing.assert_allclose(result["stokes_radiance"][1:], 0.0, atol=1e-14*source)


def test_mueller_passivity_rejects_dichroism_larger_than_total_absorption():
    with pytest.raises(ValueError, match="passive-medium"):
        stokes_mueller_absorption_matrix(0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_public_dispatcher_exposes_polarized_lte_stokes_model():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    result = run_advanced_thomson_physics(
        "polarized_lte_stokes", frequency_Hz=100e9,
        temperature_K=1.0e7, incident_stokes=[0.0, 0.0, 0.0, 0.0],
        alpha_I_m_inverse=0.4, alpha_Q_m_inverse=0.1,
        rho_V_m_inverse=0.8, path_length_m=0.2,
    )
    assert result["public_physics_entrypoint"] == "polarized_lte_stokes"
    assert result["automatic_model_substitution"] is False
    assert result["source_closure"] == "polarized_LTE_Planck_Kirchhoff"

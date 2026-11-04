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


def test_stratified_polarized_transfer_applies_segment_sources_in_ray_order():
    from tokamak_analysis.cts_stokes_transfer import propagate_stratified_polarized_lte_stokes
    from tokamak_analysis.cts_em_background import planck_spectral_radiance

    frequency = 110e9
    temperatures = [8.0e6, 2.0e7, 1.1e7]
    lengths = [0.12, 0.31, 0.08]
    alpha_i = [0.7, 0.3, 0.9]
    alpha_q = [0.1, -0.05, 0.2]
    rho_v = [0.4, 1.2, -0.3]
    initial = np.array([0.0, 0.0, 0.0, 0.0])
    result = propagate_stratified_polarized_lte_stokes(
        frequency, temperatures, lengths, initial,
        alpha_I_m_inverse=alpha_i,
        alpha_Q_m_inverse=alpha_q,
        rho_V_m_inverse=rho_v,
    )
    expected = initial
    for temp, length, ai, aq, rv in zip(temperatures, lengths, alpha_i, alpha_q, rho_v):
        step = propagate_polarized_lte_stokes(
            frequency, temp, expected,
            alpha_I_m_inverse=ai, alpha_Q_m_inverse=aq,
            rho_V_m_inverse=rv, path_length_m=length,
        )
        expected = step["stokes_radiance"]
    np.testing.assert_allclose(result["stokes_radiance"], expected, rtol=2e-12, atol=1e-30)
    assert result["segment_count"] == 3
    assert result["coefficient_averaging"] is False
    assert result["path_order"] == "segment_0_to_segment_n_minus_1"


def test_stratified_polarized_transfer_does_not_average_noncommuting_layers():
    from tokamak_analysis.cts_stokes_transfer import propagate_stratified_polarized_lte_stokes

    kwargs = dict(
        frequency_Hz=100e9, incident_stokes=[0.0, 0.0, 0.0, 0.0],
        alpha_I_m_inverse=[0.8, 0.8],
        alpha_Q_m_inverse=[0.35, 0.0],
        alpha_V_m_inverse=[0.0, 0.0],
        rho_Q_m_inverse=[0.0, 0.0],
        rho_U_m_inverse=[0.0, 0.0],
        rho_V_m_inverse=[0.0, 1.4],
        segment_length_m=[0.4, 0.7],
    )
    forward = propagate_stratified_polarized_lte_stokes(
        segment_temperature_K=[1.0e7, 2.0e7], **kwargs
    )
    reverse = propagate_stratified_polarized_lte_stokes(
        segment_temperature_K=[2.0e7, 1.0e7],
        **{**kwargs, "alpha_Q_m_inverse": [0.0, 0.35],
           "rho_V_m_inverse": [1.4, 0.0],
           "segment_length_m": [0.7, 0.4],
           "alpha_I_m_inverse": [0.8, 0.8]},
    )
    assert np.linalg.norm(forward["stokes_radiance"] - reverse["stokes_radiance"]) > 1e-8 * max(
        np.linalg.norm(forward["stokes_radiance"]), 1e-300
    )


def test_public_dispatcher_exposes_stratified_polarized_lte_transfer():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
    result = run_advanced_thomson_physics(
        "stratified_polarized_lte_stokes",
        frequency_Hz=95e9,
        segment_temperature_K=[1.0e7, 1.5e7],
        segment_length_m=[0.2, 0.3],
        incident_stokes=[0.0, 0.0, 0.0, 0.0],
        alpha_I_m_inverse=[0.5, 0.4],
        alpha_Q_m_inverse=[0.1, 0.05],
        rho_V_m_inverse=[0.2, 0.6],
    )
    assert result["public_physics_entrypoint"] == "stratified_polarized_lte_stokes"
    assert result["segment_count"] == 2
    assert result["automatic_model_substitution"] is False

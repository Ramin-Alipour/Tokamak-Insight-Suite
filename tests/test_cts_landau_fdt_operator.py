import numpy as np
from scipy.sparse.linalg import norm
from tokamak_analysis.cts_landau_fdt import (
    build_linearized_landau_fdt,
    landau_density_response,
)


def _system():
    axis = np.linspace(-4.0e6, 4.0e6, 3)
    species = [
        {"name": "electrons", "density_m3": 1.0e18,
         "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
        {"name": "ions", "density_m3": 1.0e18,
         "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19},
    ]
    return build_linearized_landau_fdt(
        axis, axis, axis, species,
        common_temperature_J=100.0*1.602176634e-19,
        coulomb_logarithm=12.0,
    )


def test_landau_operator_has_explicit_conservation_contract():
    system = _system()
    residual = norm(system.entropy_metric @ system.operator_s_inverse @ system.velocity_moments)
    scale = max(norm(system.entropy_metric @ system.operator_s_inverse), 1.0)
    assert residual/scale < 1e-10
    assert system.metadata["conservation"] == {
        "species_particle_number": True,
        "total_momentum": True,
        "total_energy": True,
    }


def test_landau_entropy_and_matching_fdt_covariance_are_positive():
    system = _system()
    K = (-system.entropy_metric @ system.operator_s_inverse).toarray()
    Q = system.fdt_source_covariance_s_inverse.toarray()
    assert np.allclose(K, K.T, rtol=1e-10, atol=1e-10*np.max(np.abs(K)))
    assert np.linalg.eigvalsh(K).min() > -1e-10*np.max(np.abs(K))
    assert np.linalg.eigvalsh(0.5*(Q+Q.T)).min() > -1e-10*np.max(np.abs(Q))
    Winv = np.diag(1.0/system.equilibrium_weights)
    L = system.operator_s_inverse.toarray()
    np.testing.assert_allclose(Q, -(L@Winv+Winv@L.T), rtol=1e-10, atol=1e-10*np.max(np.abs(Q)))


def test_landau_resolvent_returns_density_response_and_noise_spectrum():
    system = _system()
    observation = np.zeros((2, system.operator_s_inverse.shape[0]))
    for sl in system.species_slices:
        observation[system.species_slices.index(sl), sl] = system.equilibrium_weights[sl]
    result = landau_density_response(system, [1.0e5, 2.0e5], observation)
    assert result["response"].shape[0] == 2
    assert result["fluctuation_spectral_matrix"].shape == (2, 2, 2)
    assert np.all(np.isfinite(result["fluctuation_spectral_matrix"]))

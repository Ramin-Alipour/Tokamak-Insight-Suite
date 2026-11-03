import numpy as np
import pytest

from tokamak_analysis.cts_collisional_bgk import conserving_bgk_longitudinal_susceptibility


def _run(nu):
    return conserving_bgk_longitudinal_susceptibility(
        2.0e5, np.array([0.0, 1.0e5, 4.0e5]),
        species_densities_m3=[1.0e19, 1.0e19],
        species_masses_kg=[9.1093837015e-31, 1.67262192369e-27],
        common_temperature_J=100.0 * 1.602176634e-19,
        collision_frequencies_s_inverse=nu,
        quadrature_order=64,
    )


def test_conserving_bgk_returns_finite_multispecies_response():
    result = _run([2.0e7, 1.0e5])
    response = result["density_response_matrix_m3_per_J"]
    assert response.shape == (3, 2, 2)
    assert np.all(np.isfinite(response.real))
    assert np.all(np.isfinite(response.imag))


def test_operator_conservation_contract_is_explicit():
    result = _run([2.0e7, 1.0e5])
    assert result["conservation"] == {
        "species_particle_number": True,
        "total_mixture_momentum": True,
        "total_energy": False,
    }
    assert result["cts_integration_status"] == "response_kernel_only_no_FDT_fluctuation_source_or_Skomega"


def test_unsupported_collision_inputs_are_rejected():
    with pytest.raises(ValueError, match="must match"):
        conserving_bgk_longitudinal_susceptibility(
            1e5, [0.0], species_densities_m3=[1e19, 1e18],
            species_masses_kg=[1.0, 2.0],
            common_temperature_J=1.0, collision_frequencies_s_inverse=[1.0, 1.0],
        )
    with pytest.raises(ValueError, match="quadrature_order"):
        conserving_bgk_longitudinal_susceptibility(
            1e5, [0.0], species_densities_m3=[1e19],
            species_masses_kg=[1.0],
            common_temperature_J=1.0, collision_frequencies_s_inverse=[1.0],
            quadrature_order=8,
        )



def test_static_zero_flow_limit_recovers_boltzmann_response():
    result = conserving_bgk_longitudinal_susceptibility(
        2.0e5, np.array([0.0]),
        species_densities_m3=[1.0e19, 2.0e18],
        species_masses_kg=[9.1093837015e-31, 1.67262192369e-27],
        common_temperature_J=100.0 * 1.602176634e-19,
        collision_frequencies_s_inverse=[2.0e7, 1.0e5],
        quadrature_order=96,
    )
    response = result["density_response_matrix_m3_per_J"][0]
    expected = -np.diag([1.0e19, 2.0e18]) / (100.0 * 1.602176634e-19)
    assert np.allclose(response, expected, rtol=2e-5, atol=1e-8)

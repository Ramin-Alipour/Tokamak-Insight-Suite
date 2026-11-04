import numpy as np
import pytest

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics
from tokamak_analysis.cts_landau_fdt import build_linearized_landau_fdt
from tokamak_analysis.cts_landau_em import landau_full_em_current_response


def _inputs(flow=(0.0, 0.0, 0.0)):
    axis = np.linspace(-4.0, 4.0, 3)
    species = [
        {"name": "electrons", "density_m3": 1.0e18,
         "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
        {"name": "ions", "density_m3": 1.0e18,
         "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19},
    ]
    temperature = 100.0 * 1.602176634e-19
    system = build_linearized_landau_fdt(
        axis, axis, axis, species,
        common_temperature_J=temperature,
        common_bulk_flow_m_s=flow,
        coulomb_logarithm=12.0,
    )
    return system, species, temperature


def test_landau_fdt_projects_to_full_em_conductivity_and_noise_tensors():
    system, species, temperature = _inputs()
    result = landau_full_em_current_response(
        system, [1.0e5, 2.0e5], species=species,
        common_temperature_J=temperature,
        wave_number_m_inverse=1.0e-3,
    )
    assert result["conductivity_tensor_S_m"].shape == (2, 3, 3)
    assert result["relative_dielectric_tensor"].shape == (2, 3, 3)
    assert result["current_fluctuation_spectral_matrix"].shape == (2, 3, 3)
    assert np.all(np.isfinite(result["relative_dielectric_tensor"]))
    for matrix in result["current_fluctuation_spectral_matrix"]:
        eigenvalues = np.linalg.eigvalsh(matrix)
        scale = max(np.max(np.abs(eigenvalues)), np.finfo(float).tiny)
        assert eigenvalues.min() >= -1e-9 * scale
    assert result["cts_three_wave_vertex_connected"] is False
    assert result["quantitative_release_gate"]["velocity_grid_convergence_required"] is True


def test_landau_fdt_full_em_response_is_available_from_public_dispatcher():
    system, species, temperature = _inputs()
    result = run_advanced_thomson_physics(
        "landau_fdt_full_em_response",
        system=system, angular_frequency_rad_s=[1.0e5],
        species=species, common_temperature_J=temperature,
        wave_number_m_inverse=1.0e-3,
    )
    assert result["public_physics_entrypoint"] == "landau_fdt_full_em_response"
    assert result["automatic_model_substitution"] is False
    assert result["conductivity_tensor_S_m"].shape == (1, 3, 3)


def test_landau_fdt_full_em_rejects_drifted_equilibrium():
    system, species, temperature = _inputs(flow=(1.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="zero-flow"):
        landau_full_em_current_response(
            system, [1.0e5], species=species,
            common_temperature_J=temperature,
            wave_number_m_inverse=1.0e-3,
        )


def test_landau_fdt_full_em_rejects_species_or_temperature_mismatch():
    system, species, temperature = _inputs()
    with pytest.raises(ValueError, match="match the assembled Landau equilibrium"):
        landau_full_em_current_response(
            system, [1.0e5], species=species,
            common_temperature_J=1.01*temperature,
            wave_number_m_inverse=1.0e-3,
        )
    changed = [dict(item) for item in species]
    changed[0]["density_m3"] *= 2
    with pytest.raises(ValueError, match="match the species"):
        landau_full_em_current_response(
            system, [1.0e5], species=changed,
            common_temperature_J=temperature,
            wave_number_m_inverse=1.0e-3,
        )

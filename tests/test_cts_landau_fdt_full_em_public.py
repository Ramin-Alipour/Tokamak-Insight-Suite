"""Public Full-EM Landau--FDT assembly contract regressions.

These tests exercise the public Analysis-1 dispatcher, not only the lower-level
constitutive projection. They establish API reachability and declared scope;
they are not experimental validation or a magnetized CTS three-wave vertex.
"""
import numpy as np
import pytest

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def _inputs():
    axis = np.linspace(-4.0, 4.0, 3)
    species = [
        {
            "name": "electrons",
            "density_m3": 1.0e18,
            "mass_kg": 9.1093837015e-31,
            "charge_C": -1.602176634e-19,
        },
        {
            "name": "ions",
            "density_m3": 1.0e18,
            "mass_kg": 1.67262192369e-27,
            "charge_C": 1.602176634e-19,
        },
    ]
    return {
        "angular_frequency_rad_s": np.array([1.0e5, 2.0e5]),
        "normalized_velocity_x": axis,
        "normalized_velocity_y": axis,
        "normalized_velocity_z": axis,
        "species": species,
        "common_temperature_J": 100.0 * 1.602176634e-19,
        "wave_number_m_inverse": 1.0e-3,
        "coulomb_logarithm": 12.0,
    }


def test_public_full_em_landau_fdt_dispatch_assembles_and_projects_response():
    result = run_advanced_thomson_physics(
        "landau_fdt_full_em_constitutive", **_inputs()
    )

    assert result["public_physics_entrypoint"] == "landau_fdt_full_em_constitutive"
    assert result["automatic_model_substitution"] is False
    assert result["conductivity_tensor_S_m"].shape == (2, 3, 3)
    assert result["relative_dielectric_tensor"].shape == (2, 3, 3)
    assert result["current_fluctuation_spectral_matrix"].shape == (2, 3, 3)
    assert np.all(np.isfinite(result["conductivity_tensor_S_m"]))
    assert np.all(np.isfinite(result["relative_dielectric_tensor"]))
    assert np.all(np.isfinite(result["current_fluctuation_spectral_matrix"]))
    assert result["cts_three_wave_vertex_connected"] is False
    assert result["quantitative_release_gate"]["magnetized_response_required_for_magnetized_plasma"] is True

    for matrix in result["current_fluctuation_spectral_matrix"]:
        np.testing.assert_allclose(matrix, matrix.conj().T, rtol=1e-10, atol=1e-18)
        assert np.linalg.eigvalsh(matrix).min() >= -1e-9 * max(
            np.linalg.norm(matrix, ord=2), np.finfo(float).tiny
        )


def test_public_full_em_landau_fdt_rejects_nonzero_flow_outside_its_domain():
    inputs = _inputs()
    inputs["common_bulk_flow_m_s"] = (1.0e3, 0.0, 0.0)
    with pytest.raises(ValueError, match="zero-flow"):
        run_advanced_thomson_physics(
            "landau_fdt_full_em_constitutive", **inputs
        )

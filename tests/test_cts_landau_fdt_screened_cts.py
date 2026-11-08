import numpy as np
import pytest
from tokamak_analysis.cts_landau_fdt import (
    build_linearized_landau_fdt,
    landau_cts_dynamic_structure_factor,
)


def _system():
    axis = np.linspace(-4.0, 4.0, 3)
    species = [
        {"name": "electrons", "density_m3": 1.0e18,
         "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
        {"name": "ions", "density_m3": 1.0e18,
         "mass_kg": 1.67262192369e-27, "charge_C": 1.602176634e-19},
    ]
    temperature = 100.0*1.602176634e-19
    system = build_linearized_landau_fdt(
        axis, axis, axis, species, common_temperature_J=temperature,
        coulomb_logarithm=12.0,
    )
    return system, species, temperature


def test_landau_cts_closure_builds_screened_response_and_fdt_spectrum():
    system, species, temperature = _system()
    result = landau_cts_dynamic_structure_factor(
        system, [1.0e5, 2.0e5], species,
        common_temperature_J=temperature,
        wave_number_m_inverse=1.0e-3,
    )
    assert result["screened_density_response_m3_per_J"].shape == (2, 2, 2)
    assert result["dynamic_structure_factor_m3_s"].shape == (2, 2, 2)
    assert np.all(np.isfinite(result["dielectric_matrix"]))
    for matrix in result["dynamic_structure_factor_m3_s"]:
        assert np.linalg.eigvalsh(0.5*(matrix+matrix.conj().T)).min() >= -1e-10*np.linalg.norm(matrix)
    assert result["positive_semidefinite_fdt"]


def test_landau_cts_closure_requires_nonzero_wave_number_and_frequency():
    system, species, temperature = _system()
    with pytest.raises(ValueError, match="wave_number"):
        landau_cts_dynamic_structure_factor(system, [1e5], species,
            common_temperature_J=temperature, wave_number_m_inverse=0.0)
    with pytest.raises(ValueError, match="zero frequency"):
        landau_cts_dynamic_structure_factor(system, [0.0], species,
            common_temperature_J=temperature, wave_number_m_inverse=1e-3)

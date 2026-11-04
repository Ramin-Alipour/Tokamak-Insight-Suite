import numpy as np
import pytest
from scipy.sparse import csr_matrix

from tokamak_analysis.cts_landau_fdt import (
    LandauFDTSystem,
    landau_cts_dynamic_structure_factor,
)


def _minimal_system():
    return LandauFDTSystem(
        operator_s_inverse=csr_matrix((2, 2), dtype=float),
        entropy_metric=csr_matrix((2, 2), dtype=float),
        fdt_source_covariance_s_inverse=csr_matrix((2, 2), dtype=float),
        equilibrium_weights=np.ones(2),
        velocity_moments=np.zeros((2, 6)),
        species_slices=(slice(0, 1), slice(1, 2)),
        velocity_grid_m_s=np.zeros((2, 1, 3)),
        metadata={
            "common_temperature_J": 1.0,
            "species": (
                {"name": "e-", "mass_kg": 1.0, "charge_C": -1.0, "density_m3": 2.0},
                {"name": "D+", "mass_kg": 4.0, "charge_C": 1.0, "density_m3": 0.5},
            ),
        },
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("mass_kg", 1.01),
        ("charge_C", -0.99),
        ("density_m3", 2.01),
    ],
)
def test_electrostatic_cts_rejects_species_mixture_different_from_landau_operator(field, value):
    species = [
        {"name": "e-", "mass_kg": 1.0, "charge_C": -1.0, "density_m3": 2.0},
        {"name": "D+", "mass_kg": 4.0, "charge_C": 1.0, "density_m3": 0.5},
    ]
    species[0][field] = value
    with pytest.raises(ValueError, match="match the mixture used to assemble"):
        landau_cts_dynamic_structure_factor(
            _minimal_system(),
            np.asarray([1.0]),
            species,
            common_temperature_J=1.0,
            wave_number_m_inverse=1.0,
        )


def test_electrostatic_cts_rejects_missing_species_metadata_contract():
    system = _minimal_system()
    system.metadata["species"] = ()
    species = [
        {"name": "e-", "mass_kg": 1.0, "charge_C": -1.0, "density_m3": 2.0},
        {"name": "D+", "mass_kg": 4.0, "charge_C": 1.0, "density_m3": 0.5},
    ]
    with pytest.raises(ValueError, match="complete species contract"):
        landau_cts_dynamic_structure_factor(
            system,
            np.asarray([1.0]),
            species,
            common_temperature_J=1.0,
            wave_number_m_inverse=1.0,
        )


def test_electrostatic_cts_rejects_temperature_different_from_landau_equilibrium():
    species = [
        {"name": "e-", "mass_kg": 1.0, "charge_C": -1.0, "density_m3": 2.0},
        {"name": "D+", "mass_kg": 4.0, "charge_C": 1.0, "density_m3": 0.5},
    ]
    with pytest.raises(ValueError, match="must match the temperature used to assemble"):
        landau_cts_dynamic_structure_factor(
            _minimal_system(),
            np.asarray([1.0]),
            species,
            common_temperature_J=1.01,
            wave_number_m_inverse=1.0,
        )

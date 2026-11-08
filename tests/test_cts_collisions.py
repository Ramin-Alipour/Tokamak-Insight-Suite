import numpy as np
import pytest

from tokamak_analysis.cts_collisions import (
    coulomb_logarithm_electron,
    full_em_cts_coulomb_collision_rates,
)


def test_coulomb_logarithm_is_finite_and_positive_for_weakly_coupled_plasma():
    result = coulomb_logarithm_electron(1.0e19, 1000.0)
    assert np.isfinite(result["coulomb_logarithm"])
    assert result["coulomb_logarithm"] > 0.0
    assert result["debye_length_m"] > result["minimum_impact_parameter_m"]


def test_electron_ion_rate_scales_linearly_with_density_at_fixed_temperature():
    args = (1000.0,)
    a = full_em_cts_coulomb_collision_rates(
        1.0e19, args[0], [1.0e19], [1.0], [2.0]
    )
    b = full_em_cts_coulomb_collision_rates(
        2.0e19, args[0], [2.0e19], [1.0], [2.0]
    )
    assert b["electron_ion_collision_frequency_s_inverse"] > a["electron_ion_collision_frequency_s_inverse"]
    assert b["effective_charge_Zeff"] == pytest.approx(a["effective_charge_Zeff"])


def test_collision_rate_input_contract_rejects_mismatched_species_vectors():
    with pytest.raises(ValueError, match="matching"):
        full_em_cts_coulomb_collision_rates(
            1.0e19, 1000.0, [1.0e19, 2.0e19], [1.0], [2.0, 4.0]
        )


def test_rate_output_does_not_claim_a_full_landau_operator():
    result = full_em_cts_coulomb_collision_rates(
        1.0e19, 1000.0, [1.0e19], [1.0], [2.0]
    )
    assert result["collision_operator_status"] == "rates_only_full_linearized_Landau_operator_not_implemented"
    assert "Do not add" in result["integration_warning"]



def test_multispecies_collision_matrix_has_species_resolved_pair_rates():
    from tokamak_analysis.cts_collisions import multispecies_thermal_coulomb_collision_matrix

    species = [
        {"name": "e", "density_m3": 2.0e19, "mass_kg": 9.1093837015e-31,
         "charge_C": -1.602176634e-19, "temperature_eV": 1500.0},
        {"name": "D", "density_m3": 1.0e19, "mass_kg": 2.0 * 1.66053906660e-27,
         "charge_C": 1.602176634e-19, "temperature_eV": 1200.0},
        {"name": "C6", "density_m3": 1.0e18, "mass_kg": 12.0 * 1.66053906660e-27,
         "charge_C": 6.0 * 1.602176634e-19, "temperature_eV": 900.0},
    ]
    result = multispecies_thermal_coulomb_collision_matrix(species)
    rates = result["collision_frequency_matrix_s_inverse"]
    assert rates.shape == (3, 3)
    assert np.all(np.isfinite(rates)) and np.all(rates >= 0.0)
    assert rates[0, 2] > rates[0, 1]
    assert result["species_order"] == ["e", "D", "C6"]
    assert result["collision_operator_status"] == "thermal_rates_only_no_field_particle_conservation_or_FDT"


def test_pairwise_collision_matrix_reduces_to_analytic_leading_log_rate():
    from tokamak_analysis.cts_collisions import multispecies_thermal_coulomb_collision_matrix

    e = 1.602176634e-19
    eps0 = 8.8541878128e-12
    me = 9.1093837015e-31
    density = 1.0e19
    temperature_eV = 1000.0
    log_lambda = 12.0
    species = [
        {"name": "e", "density_m3": density, "mass_kg": me,
         "charge_C": -e, "temperature_eV": temperature_eV},
        {"name": "D", "density_m3": density, "mass_kg": 2.0 * 1.66053906660e-27,
         "charge_C": e, "temperature_eV": temperature_eV},
    ]
    result = multispecies_thermal_coulomb_collision_matrix(
        species, coulomb_logarithm_matrix=np.full((2, 2), log_lambda)
    )
    kt = temperature_eV * e
    vrel = np.sqrt(kt / me + kt / species[1]["mass_kg"])
    expected = (
        4.0 * np.sqrt(2.0 * np.pi) / 3.0
        * density * e**4 * log_lambda
        / ((4.0 * np.pi * eps0)**2 * me**2 * vrel**3)
    )
    assert result["collision_frequency_matrix_s_inverse"][0, 1] == pytest.approx(expected, rel=1e-13)


def test_pairwise_collision_matrix_rejects_asymmetric_supplied_coulomb_logs():
    from tokamak_analysis.cts_collisions import multispecies_thermal_coulomb_collision_matrix

    species = [
        {"name": "e", "density_m3": 1.0e19, "mass_kg": 9.1093837015e-31,
         "charge_C": -1.602176634e-19, "temperature_eV": 1000.0},
        {"name": "D", "density_m3": 1.0e19, "mass_kg": 3.3452439e-27,
         "charge_C": 1.602176634e-19, "temperature_eV": 1000.0},
    ]
    with pytest.raises(ValueError, match="symmetric"):
        multispecies_thermal_coulomb_collision_matrix(
            species, coulomb_logarithm_matrix=[[10.0, 11.0], [10.0, 10.0]]
        )

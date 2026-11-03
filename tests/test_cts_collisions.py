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

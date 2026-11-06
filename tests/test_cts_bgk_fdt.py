import numpy as np
import pytest

from tokamak_analysis.cts_bgk_fdt import analyze_conserving_bgk_cts
from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def _inputs():
    return dict(
        angular_frequency_rad_s=np.array([1.0e10, 2.0e10]),
        wave_number_m_inverse=2.0e5,
        species=[
            {"name": "electron", "density_m3": 1.0e19,
             "mass_kg": 9.1093837015e-31, "charge_C": -1.602176634e-19},
            {"name": "deuterium", "density_m3": 1.0e19,
             "mass_kg": 3.3435837724e-27, "charge_C": 1.602176634e-19},
        ],
        common_temperature_J=100.0 * 1.602176634e-19,
        collision_frequencies_s_inverse=[2.0e7, 1.0e5],
        quadrature_order=64,
    )


def test_bgk_cts_connects_bare_response_poisson_screening_and_fdt():
    result = analyze_conserving_bgk_cts(**_inputs())
    assert result["integration_status"] == (
        "BGK_response_Poisson_screening_and_matching_classical_FDT_connected"
    )
    assert result["bare_density_response_m3_per_J"].shape == (2, 2, 2)
    assert result["screened_density_response_m3_per_J"].shape == (2, 2, 2)
    assert result["dynamic_structure_factor_m3_s"].shape == (2, 2, 2)
    assert np.all(np.isfinite(result["dielectric_condition_number"]))
    assert np.all(result["dielectric_condition_number"] < 1.0e14)
    assert result["fdt_positive_semidefinite"] is True


def test_bgk_cts_screening_is_the_declared_poisson_matrix_closure():
    result = analyze_conserving_bgk_cts(**_inputs())
    identity = np.eye(2, dtype=complex)
    for index, bare in enumerate(result["bare_density_response_m3_per_J"]):
        expected_dielectric = (
            identity - bare @ result["coulomb_interaction_matrix_J_m3"]
        )
        assert np.allclose(result["dielectric_matrix"][index], expected_dielectric)
        expected_screened = np.linalg.solve(expected_dielectric, bare)
        assert np.allclose(
            result["screened_density_response_m3_per_J"][index], expected_screened
        )


def test_bgk_cts_rejects_zero_frequency_instead_of_fabricating_static_spectrum():
    args = _inputs()
    args["angular_frequency_rad_s"] = np.array([0.0, 1.0e10])
    with pytest.raises(ValueError, match="zero frequency"):
        analyze_conserving_bgk_cts(**args)


def test_bgk_cts_is_reachable_through_public_analysis_one_physics_api():
    result = run_advanced_thomson_physics("cts_conserving_bgk_fdt", **_inputs())
    assert result["public_physics_entrypoint"] == "cts_conserving_bgk_fdt"
    assert result["automatic_model_substitution"] is False
    assert result["cts_model"].startswith("isothermal_momentum_conserving_multispecies_BGK")

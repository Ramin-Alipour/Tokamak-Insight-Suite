import numpy as np
import pytest

from tis.ui.advanced_thomson_physics_page import decode_complex_json, json_safe


def test_complex_json_arrays_decode_without_shape_or_phase_loss():
    value = {"real": [[1.0, 0.0]], "imag": [[0.0, -2.0]]}
    decoded = decode_complex_json(value)
    np.testing.assert_allclose(decoded, [[1.0 + 0.0j, -2.0j]])
    assert decoded.dtype.kind == "c"


def test_complex_json_rejects_mismatched_real_and_imag_shapes():
    with pytest.raises(ValueError, match="identical shapes"):
        decode_complex_json({"real": [1.0, 2.0], "imag": [1.0]})


def test_json_safe_preserves_complex_arrays_and_omits_internal_landau_operator():
    value = {
        "field": np.array([1.0 + 2.0j]),
        "landau_system": object(),
        "count": np.int64(3),
    }
    safe = json_safe(value)
    assert safe == {
        "field": {"real": [1.0], "imag": [2.0]},
        "count": 3,
    }


def test_full_tail_kappa_cts_solver_is_exposed_in_dashboard_and_formula_registry():
    import json
    from pathlib import Path
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    model = "kappa_electrostatic_susceptibility"
    assert model in MODEL_LABELS
    configuration = json.loads(json.dumps(EXAMPLES[model]))
    result = run_advanced_thomson_physics(model, **configuration)
    assert result["landau_integral_tail_complete"] is True
    assert result["public_physics_entrypoint"] == model

    root = Path(__file__).resolve().parents[1]
    formula = json.loads((root / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    packaged = json.loads((root / "tis" / "data" / "FORMULA_REGISTRY.json").read_text(encoding="utf-8"))
    key = "cts_isotropic_kappa_landau_susceptibility"
    matches = [item for item in formula if item.get("execution_contract", {}).get("formula_key") == key]
    assert len(matches) == 1
    assert formula == packaged


def test_full_em_landau_fdt_constitutive_kernel_is_exposed_without_overclaiming():
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS

    model = "landau_fdt_full_em_constitutive"
    assert model in MODEL_LABELS
    assert "not a three-wave spectrum" in MODEL_LABELS[model]
    config = EXAMPLES[model]
    assert config["common_bulk_flow_m_s"] == [0.0, 0.0, 0.0]
    assert len(config["species"]) == 2
    assert config["common_temperature_J"] > 0.0


def test_dashboard_exposes_all_json_callable_advanced_physics_routes():
    from tis.ui.advanced_thomson_physics_page import MODEL_LABELS

    expected = {
        "kappa_electrostatic_susceptibility",
        "landau_fdt_cts",
        "landau_fdt_full_em_constitutive",
        "quantum_fdt",
        "quantum_fdt_unsymmetrized",
        "relativistic_kinetic_ece_profile",
        "relativistic_kinetic_ece",
        "microwave_background",
        "multispecies_thermal_collision_matrix",
        "polarized_kinetic_ece",
        "stratified_nonparaxial_vector",
        "stratified_mode_conversion",
        "polarized_lte_stokes",
        "stratified_polarized_lte_stokes",
        "cold_plasma_dielectric",
        "cold_plasma_magnetoionic_modes",
        "cold_plasma_absorbed_power_density",
        "cold_plasma_mode_radiative_coefficients",
        "cold_plasma_mode_group_velocity",
        "thomson_projected_evdf",
        "thomson_relativistic_projected_evdf",
        "thomson_psf_profile_inverse",
        "gyrotropic_bi_maxwellian_anisotropy",
        "thomson_three_dimensional_flow",
        "plasma_charge_state_moments",
    }
    assert expected <= set(MODEL_LABELS)


def test_dashboard_routes_calibrated_anisotropic_spectra_to_joint_physical_fit():
    from tis.ui.advanced_thomson_physics_page import EXAMPLES, MODEL_LABELS
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    model = "gyrotropic_bi_maxwellian_spectral_fit"
    assert model in MODEL_LABELS
    result = run_advanced_thomson_physics(model, **EXAMPLES[model])
    assert result["public_physics_entrypoint"] == model
    assert result["model"] == "gyrotropic_bi_maxwellian_nonrelativistic_optical_thomson"
    assert result["model_validity"]["relativistic_corrections"] is False
    assert result["directional_temperature_eV"].shape == (3,)

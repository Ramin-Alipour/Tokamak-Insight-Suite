"""Public CTS background-dispatch regression tests."""
import numpy as np
import pytest

from tokamak_analysis.cts_full_physics import (
    analyze_cts_microwave_background,
    run_advanced_thomson_physics,
)


def _zero_background_inputs():
    return dict(
        frequency_Hz=np.array([90e9, 100e9]),
        path_length_m=np.array([0.1, 0.2]),
        electron_temperature_eV=np.array([1000.0, 1200.0]),
        electron_density_m3=np.array([0.0, 0.0]),
        ion_densities_m3=np.zeros((2, 1)),
        ion_charge_states=np.array([1.0]),
    )


def test_public_background_route_preserves_exact_zero_contract():
    result = analyze_cts_microwave_background(**_zero_background_inputs())
    assert result["zero_background"] is True
    assert result["background_status"] == "zero_background_exact_zero_radiance"
    np.testing.assert_array_equal(
        result["spectral_radiance_W_m2_sr_Hz"], np.zeros(2)
    )


def test_dispatcher_exposes_background_route_without_implicit_ece_model():
    result = run_advanced_thomson_physics(
        "microwave_background", **_zero_background_inputs()
    )
    assert result["public_physics_entrypoint"] == "microwave_background"
    assert result["background_inputs_are_explicit"] is True
    assert result["ece_model_status"].startswith("ECE omitted")


def test_background_route_rejects_mode_weights_without_mode_resolved_opacity():
    inputs = _zero_background_inputs()
    inputs["ece_mode_weights"] = np.array([0.5, 0.5])
    with pytest.raises(ValueError, match="requires mode-resolved ECE absorption"):
        analyze_cts_microwave_background(**inputs)


def _ece_background_inputs():
    inputs = _zero_background_inputs()
    inputs["electron_density_m3"] = np.array([1.0e19, 1.0e19])
    inputs["ion_densities_m3"] = np.array([[1.0e19], [1.0e19]])
    inputs["ece_absorption_coefficient_m_inverse"] = np.full((2, 2), 1.0e-3)
    return inputs


def test_public_background_route_forwards_explicit_non_lte_ece_emissivity():
    inputs = _ece_background_inputs()
    inputs["ece_emissivity_W_m3_sr_Hz"] = np.full((2, 2), 2.0e-12)
    result = analyze_cts_microwave_background(**inputs)
    assert result["ece_source_model"] == "explicit_mode_resolved_ECE_emissivity"
    assert np.all(np.isfinite(result["spectral_radiance_W_m2_sr_Hz"]))
    assert np.all(result["spectral_radiance_W_m2_sr_Hz"] > 0.0)


def test_public_background_route_forwards_explicit_lte_selection():
    inputs = _ece_background_inputs()
    inputs["ece_lte"] = True
    result = run_advanced_thomson_physics("microwave_background", **inputs)
    assert result["ece_source_model"] == "explicit_LTE_Planck_source_function"
    assert result["public_physics_entrypoint"] == "microwave_background"


def test_public_background_route_does_not_infer_lte_from_absorption():
    inputs = _ece_background_inputs()
    with pytest.raises(ValueError, match="absorption alone does not define emissivity"):
        analyze_cts_microwave_background(**inputs)

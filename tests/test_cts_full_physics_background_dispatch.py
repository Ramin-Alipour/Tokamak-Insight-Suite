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

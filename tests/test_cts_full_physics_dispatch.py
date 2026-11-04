import numpy as np
import pytest

from tokamak_analysis.cts_full_physics import (
    analyze_cts_microwave_background,
    run_advanced_thomson_physics,
)


def _vacuum_background_inputs():
    return {
        "frequency_Hz": np.array([90e9, 100e9, 110e9]),
        "path_length_m": np.array([0.1, 0.2]),
        "electron_temperature_eV": np.array([100.0, 150.0]),
        "electron_density_m3": np.zeros(2),
        "ion_densities_m3": np.zeros((2, 1)),
        "ion_charge_states": np.array([1.0]),
    }


def test_microwave_background_is_reachable_through_public_dispatcher():
    result = run_advanced_thomson_physics(
        "microwave_background", **_vacuum_background_inputs()
    )
    np.testing.assert_array_equal(
        result["spectral_radiance_W_m2_sr_Hz"], np.zeros(3)
    )
    assert result["public_physics_entrypoint"] == "microwave_background"
    assert result["background_status"] == "zero_background_exact_zero_radiance"
    assert result["automatic_model_substitution"] is False


def test_microwave_background_direct_entrypoint_matches_dispatcher():
    direct = analyze_cts_microwave_background(**_vacuum_background_inputs())
    dispatched = run_advanced_thomson_physics(
        "microwave_background", **_vacuum_background_inputs()
    )
    np.testing.assert_array_equal(
        direct["spectral_radiance_W_m2_sr_Hz"],
        dispatched["spectral_radiance_W_m2_sr_Hz"],
    )


def test_dispatcher_rejects_unknown_physics_without_fallback():
    with pytest.raises(ValueError, match="no automatic model substitution"):
        run_advanced_thomson_physics("unimplemented_relativistic_ece")

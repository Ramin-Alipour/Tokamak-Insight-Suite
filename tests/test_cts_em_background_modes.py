import numpy as np
import pytest

from tokamak_analysis.cts_em_background import integrate_thermal_microwave_background


def _base_inputs():
    return {
        "frequency_Hz": np.array([20e9, 21e9, 22e9]),
        "path_length_m": np.array([0.2, 0.3]),
        "electron_temperature_eV": np.array([1.0e3, 1.2e3]),
        "electron_density_m3": np.array([1.0e19, 1.1e19]),
        "ion_densities_m3": np.array([[1.0e19], [1.1e19]]),
        "ion_charge_states": np.array([1.0]),
        "gaunt_factor": 1.0,
    }


def test_mode_resolved_ece_is_propagated_independently_and_receiver_weighted():
    inputs = _base_inputs()
    alpha = np.zeros((2, 3, 2), dtype=float)
    alpha[:, :, 0] = 0.1
    alpha[:, :, 1] = 0.8
    result = integrate_thermal_microwave_background(
        **inputs,
        ece_absorption_coefficient_m_inverse=alpha,
        ece_mode_weights=np.array([0.25, 0.75]),
        ece_lte=True,
        incident_radiance_W_m2_sr_Hz=0.0,
    )
    by_mode = result["spectral_radiance_by_mode_W_m2_sr_Hz"]
    assert by_mode.shape == (2, 3)
    assert np.all(by_mode[1] > by_mode[0])
    np.testing.assert_allclose(
        result["spectral_radiance_W_m2_sr_Hz"],
        0.25 * by_mode[0] + 0.75 * by_mode[1],
    )
    assert result["ece_optical_depth_by_mode"].shape == (2, 3)
    assert result["mode_coupling_model"] == "neglected; modes propagated independently"


def test_mode_resolved_ece_requires_explicit_normalized_receiver_weights():
    inputs = _base_inputs()
    alpha = np.ones((2, 3, 2), dtype=float)
    with pytest.raises(ValueError, match="ece_mode_weights is required"):
        integrate_thermal_microwave_background(
            **inputs, ece_absorption_coefficient_m_inverse=alpha
        )
    with pytest.raises(ValueError, match="summing to one"):
        integrate_thermal_microwave_background(
            **inputs,
            ece_absorption_coefficient_m_inverse=alpha,
            ece_mode_weights=np.array([0.2, 0.2]),
        )


def test_legacy_single_effective_mode_ece_contract_is_preserved():
    inputs = _base_inputs()
    alpha = np.full((2, 3), 0.2)
    result = integrate_thermal_microwave_background(
        **inputs, ece_absorption_coefficient_m_inverse=alpha, ece_lte=True
    )
    assert result["spectral_radiance_by_mode_W_m2_sr_Hz"].shape == (1, 3)
    assert "single-effective-mode" in result["ece_model_status"]

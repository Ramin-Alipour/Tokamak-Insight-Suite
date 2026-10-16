import numpy as np
import pytest

from tokamak_analysis.mirnov_analysis import analyze_mirnov_signal


def test_mirnov_coil_bdot_uses_integer_turn_count():
    time = np.arange(8, dtype=float) * 1e-4
    voltage = np.sin(2 * np.pi * 1000 * time)
    result = analyze_mirnov_signal(time, voltage, coil_area_m2=1e-4, turns=10)
    np.testing.assert_allclose(result["approx_Bdot_T_s"], -voltage / (10 * 1e-4))


@pytest.mark.parametrize("turns", [0, -1, 1.5, float("nan"), float("inf")])
def test_mirnov_rejects_invalid_turn_count(turns):
    time = np.arange(8, dtype=float) * 1e-4
    voltage = np.sin(2 * np.pi * 1000 * time)
    with pytest.raises(ValueError, match="turns"):
        analyze_mirnov_signal(time, voltage, coil_area_m2=1e-4, turns=turns)

from tokamak_analysis.mirnov_analysis import magnetic_perturbation_from_mirnov_voltage


@pytest.mark.parametrize("turns", [0, -1, 1.5, float("nan"), float("inf")])
def test_mirnov_field_integration_rejects_invalid_turn_count(turns):
    time = np.arange(4, dtype=float) * 1e-4
    voltage = np.ones(4)
    with pytest.raises(ValueError, match="turns"):
        magnetic_perturbation_from_mirnov_voltage(time, voltage, turns, 1e-4)

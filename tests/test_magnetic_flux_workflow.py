import numpy as np
import pytest
from tis.magnetic import magnetic_flux_from_voltage


def test_flux_integral_matches_constant_voltage_faraday_law():
    t = np.linspace(0, 2, 101)
    v = np.full_like(t, 3.0)
    result = magnetic_flux_from_voltage(t, v, polarity_sign=-1)
    assert result["magnetic_flux_Wb"][-1] == pytest.approx(-6.0)
    assert result["flux_change_Wb"][-1] == pytest.approx(-6.0)


def test_flux_integral_initial_value_and_sign():
    t = np.linspace(0, 1, 21)
    result = magnetic_flux_from_voltage(t, np.ones_like(t), polarity_sign=1, initial_flux_Wb=0.4)
    assert result["magnetic_flux_Wb"][-1] == pytest.approx(1.4)
    assert result["flux_change_Wb"][-1] == pytest.approx(1.0)


def test_flux_rejects_bad_inputs():
    with pytest.raises(ValueError):
        magnetic_flux_from_voltage([0, 1], [1, 2], polarity_sign=0)
    with pytest.raises(ValueError):
        magnetic_flux_from_voltage([0, 1], [1, np.nan])
    with pytest.raises(ValueError):
        magnetic_flux_from_voltage([0, 0], [1, 2])

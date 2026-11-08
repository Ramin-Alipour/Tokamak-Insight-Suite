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


@pytest.mark.parametrize("time, voltage", [
    ([[0.0, 1.0]], [[1.0, 1.0]]),
    ([0.0, 1.0], [[1.0], [1.0]]),
])
def test_flux_rejects_multidimensional_inputs_instead_of_flattening(time, voltage):
    with pytest.raises(ValueError, match="one-dimensional"):
        magnetic_flux_from_voltage(time, voltage)


@pytest.mark.parametrize("initial_flux", [np.nan, np.inf, True, [0.0]])
def test_flux_rejects_invalid_initial_flux(initial_flux):
    with pytest.raises(ValueError, match="initial_flux_Wb"):
        magnetic_flux_from_voltage([0.0, 1.0], [0.0, 0.0], initial_flux_Wb=initial_flux)


@pytest.mark.parametrize("polarity", [True, 1.0, 0, 2, "1"])
def test_flux_rejects_non_integer_or_invalid_polarity(polarity):
    with pytest.raises(ValueError, match="polarity_sign"):
        magnetic_flux_from_voltage([0.0, 1.0], [0.0, 0.0], polarity_sign=polarity)

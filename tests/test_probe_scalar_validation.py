"""Regression tests for scalar validation in electrostatic probe analysis."""
import numpy as np
import pytest

from tis.probe_analysis import langmuir_iv_analysis, triple_probe_temperature_eV

_BAD = [True, np.bool_(True), "1.0", np.nan, np.inf, 10**1000]


@pytest.mark.parametrize("field,bad", [
    ("area_m2", True),
    ("ion_mass_amu", "2.0"),
    ("ti_eV", np.inf),
    ("sheath_factor", np.bool_(True)),
    ("gamma_i", 10**1000),
])
def test_langmuir_rejects_invalid_physics_scalars(field, bad):
    kwargs = dict(area_m2=1e-5, ion_mass_amu=2.0)
    kwargs[field] = bad
    voltage = np.linspace(-20.0, 20.0, 41)
    current = np.tanh(voltage / 5.0)
    with pytest.raises(ValueError):
        langmuir_iv_analysis(voltage, current, **kwargs)


@pytest.mark.parametrize("bad", _BAD)
def test_triple_probe_rejects_invalid_scalar_inputs(bad):
    with pytest.raises(ValueError):
        triple_probe_temperature_eV(bad, 1.0)
    with pytest.raises(ValueError):
        triple_probe_temperature_eV(2.0, bad)


@pytest.mark.parametrize("bad_range", [("0", "1"), (True, 2.0), (0.0, np.inf)])
def test_langmuir_fit_ranges_require_numeric_finite_limits(bad_range):
    voltage = np.linspace(-20.0, 20.0, 41)
    current = np.tanh(voltage / 5.0)
    with pytest.raises(ValueError):
        langmuir_iv_analysis(voltage, current, electron_fit_range=bad_range)


def test_langmuir_rejects_unhashable_current_convention():
    voltage = np.linspace(-20.0, 20.0, 41)
    current = np.tanh(voltage / 5.0)
    with pytest.raises(ValueError, match="current_convention"):
        langmuir_iv_analysis(voltage, current, current_convention=[])

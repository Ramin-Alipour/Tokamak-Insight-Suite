import numpy as np
import pytest

from tis.core import langmuir_iv_analysis


def test_langmuir_iv_converts_current_convention_and_estimates_te():
    voltage = np.linspace(-12, 8, 101)
    current_electron_positive = np.where(
        voltage < -7, -0.01, 0.02 * np.exp(np.clip(voltage / 5, -20, 5)) - 0.01
    )
    result = langmuir_iv_analysis(
        voltage, -current_electron_positive,
        electron_fit_range=(-5, 0), ion_fit_range=(-10, -7),
        current_convention="ion_positive",
    )
    assert result["electron_temperature_eV"] == pytest.approx(5, rel=0.15)
    assert result["current_convention"] == "ion_positive"
    assert result["floating_potential_V"] is not None
    assert result["ion_saturation_current_A"] < 0


def test_langmuir_density_requires_temperature_and_geometry():
    voltage = np.linspace(-10, 5, 61)
    current = np.where(voltage < -6, -0.01, 0.02 * np.exp(np.clip(voltage / 3, -20, 5)) - 0.01)
    with pytest.raises(ValueError, match="area_m2 and ion_mass_amu"):
        langmuir_iv_analysis(voltage, current, area_m2=1e-6)
    with pytest.raises(ValueError, match="current_convention"):
        langmuir_iv_analysis(voltage, current, current_convention="unknown")

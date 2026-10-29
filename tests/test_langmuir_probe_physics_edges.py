"""Physics-oriented regression tests for the Langmuir I-V workflow."""
import numpy as np
import pytest

from tokamak_analysis.langmuir_probe import analyze_langmuir_probe


def iv_sweep(te_eV=4.0):
    voltage = np.linspace(-20.0, 20.0, 161)
    current = -2.0e-3 + 5.0e-6 * np.exp((voltage + 20.0) / te_eV)
    return voltage, current


def test_temperature_fit_is_invariant_to_current_unit_scale():
    voltage, current = iv_sweep()
    kwargs = {"electron_fit_range": (-2.0, 12.0), "ion_fit_range": (-20.0, -12.0)}
    baseline = analyze_langmuir_probe(voltage, current, **kwargs)
    scaled = analyze_langmuir_probe(voltage, current * 1e3, **kwargs)
    assert scaled["electron_temperature_eV"] == pytest.approx(
        baseline["electron_temperature_eV"], rel=2e-5
    )


def test_density_inversion_is_excluded_even_when_fit_windows_are_supplied():
    voltage, current = iv_sweep()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-2.0, 12.0),
        ion_fit_range=(-20.0, -12.0),
    )
    assert result["electron_temperature_eV"] == pytest.approx(4.0, rel=2e-5)
    assert "ion_density_estimate_m-3" not in result
    assert "ion_density_status" not in result

def test_density_does_not_depend_on_collection_area_because_not_inferred():
    voltage, current = iv_sweep()
    result = analyze_langmuir_probe(
        voltage, current, electron_fit_range=(-2.0, 12.0),
        ion_fit_range=(-20.0, -12.0),
    )
    assert "ion_density_estimate_m-3" not in result
    assert "ion_sound_speed_m_s" not in result

@pytest.mark.parametrize(
    "kwargs",
    [
        {"current_convention": "automatic"},
        {"electron_fit_range": (0.0, 0.0)},
        {"ion_fit_range": (1.0, 1.0)},
        {"electron_fit_range": (float("nan"), 2.0)},
    ],
)
def test_nonphysical_or_nonfinite_model_inputs_are_rejected(kwargs):
    voltage, current = iv_sweep()
    with pytest.raises(ValueError):
        analyze_langmuir_probe(voltage, current, **kwargs)

def test_fit_ranges_must_contain_enough_measured_points():
    voltage, current = iv_sweep()
    with pytest.raises(ValueError, match="at least 3 valid samples"):
        analyze_langmuir_probe(voltage, current, ion_fit_range=(0.01, 0.02))
    with pytest.raises(ValueError, match="at least 4 points"):
        analyze_langmuir_probe(voltage, current, electron_fit_range=(0.01, 0.02))

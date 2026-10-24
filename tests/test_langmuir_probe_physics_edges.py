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


def test_density_requires_temperature_fit_even_when_geometry_is_supplied():
    voltage, current = iv_sweep()
    with pytest.raises(ValueError, match="valid positive electron temperature"):
        analyze_langmuir_probe(
            voltage, current, ion_fit_range=(-20.0, -12.0),
            area_m2=1e-5, ion_mass_amu=2.0,
        )


def test_density_estimate_scales_inverse_with_collection_area():
    voltage, current = iv_sweep()
    kwargs = {
        "electron_fit_range": (-2.0, 12.0),
        "ion_fit_range": (-20.0, -12.0),
        "ion_mass_amu": 2.0,
    }
    small_area = analyze_langmuir_probe(voltage, current, area_m2=1e-5, **kwargs)
    large_area = analyze_langmuir_probe(voltage, current, area_m2=2e-5, **kwargs)
    assert small_area["ion_density_estimate_m-3"] == pytest.approx(
        2.0 * large_area["ion_density_estimate_m-3"], rel=1e-10
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ti_eV": -1.0},
        {"ti_eV": float("nan")},
        {"sheath_factor": 0.0},
        {"gamma_i": -0.1},
        {"area_m2": float("inf"), "ion_mass_amu": 2.0},
        {"area_m2": 1e-5, "ion_mass_amu": float("nan")},
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

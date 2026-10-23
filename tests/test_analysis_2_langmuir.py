import numpy as np
import pytest
from tokamak_analysis.analysis_registry import get_analysis, resolve_analysis


def test_analysis_2_is_langmuir_probe_with_complete_contract():
    entry = get_analysis(2)
    assert entry["name"] == "Langmuir Probe"
    contract = entry["execution_contract"]
    assert contract["completeness"] == "complete"
    assert contract["raw_data_to_physics"] is True
    assert {x["name"] for x in contract["input_specs"]} >= {"voltage_V", "current_A", "electron_fit_range", "ion_fit_range"}
    assert callable(resolve_analysis(2))


def test_analysis_2_langmuir_probe_synthetic_sweep():
    analyze = resolve_analysis(2)
    voltage = np.linspace(-10.0, 10.0, 101)
    current = 2e-3 * np.tanh(voltage / 3.0) + 1e-4 * voltage
    result = analyze(voltage, current)
    assert result["voltage_V"].shape == voltage.shape
    assert result["current_A"].shape == current.shape
    assert np.isfinite(result["ion_saturation_current_A"])
    assert np.isfinite(result["electron_saturation_current_A"])


def test_analysis_2_langmuir_rejects_invalid_input():
    analyze = resolve_analysis(2)
    with pytest.raises(ValueError):
        analyze([0, 1, 2], [0, 1, 2])

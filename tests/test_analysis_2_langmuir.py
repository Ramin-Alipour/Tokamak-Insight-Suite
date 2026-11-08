import numpy as np
import pytest
from tokamak_analysis.analysis_registry import get_analysis, resolve_analysis


def test_analysis_2_is_langmuir_probe_with_complete_contract():
    entry = get_analysis(2)
    assert entry["name"] == "Langmuir Probe"
    contract = entry["execution_contract"]
    assert contract["completeness"] == "complete_with_declared_scope"
    assert contract["raw_data_to_physics"] is False
    assert {x["name"] for x in contract["input_specs"]} >= {"voltage_V", "current_A", "electron_fit_range", "ion_fit_range"}
    assert callable(resolve_analysis(2))


def test_analysis_2_langmuir_probe_synthetic_sweep():
    analyze = resolve_analysis(2)
    voltage = np.linspace(-10.0, 10.0, 101)
    current = 2e-3 * np.tanh(voltage / 3.0) + 1e-4 * voltage
    result = analyze(voltage, current)
    assert np.isfinite(result["ion_saturation_current_A"])
    assert "floating_potential_V" in result
    assert "plasma_potential_proxy_V" in result
    assert "interpretation_warning" in result


def test_analysis_2_langmuir_rejects_invalid_input():
    analyze = resolve_analysis(2)
    with pytest.raises(ValueError):
        analyze([0, 1, 2], [0, 1, 2])



def test_analysis_2_current_convention_metadata_matches_normalized_outputs():
    analyze = resolve_analysis(2)
    voltage = np.linspace(-5.0, 5.0, 21)
    electron_positive_current = 1e-3 * voltage
    electron_result = analyze(
        voltage, electron_positive_current, current_convention="electron_positive"
    )
    ion_positive_result = analyze(
        voltage, -electron_positive_current, current_convention="ion_positive"
    )
    assert electron_result["current_convention_input"] == "electron_positive"
    assert ion_positive_result["current_convention_input"] == "ion_positive"
    assert electron_result["current_convention"] == "electron_positive"
    assert ion_positive_result["current_convention"] == "electron_positive"
    assert np.allclose(
        electron_result["processed_current_A"],
        ion_positive_result["processed_current_A"],
    )


def test_analysis_2_fit_window_suggester_rejects_non_string_convention():
    from tis.probe_analysis import suggest_langmuir_fit_windows

    voltage = np.linspace(-5.0, 5.0, 21)
    current = 1e-3 * voltage
    with pytest.raises(ValueError, match="current_convention"):
        suggest_langmuir_fit_windows(voltage, current, current_convention=[])

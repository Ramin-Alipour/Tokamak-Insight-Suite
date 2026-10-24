import numpy as np
import pytest

from tokamak_analysis.langmuir_probe import analyze_langmuir_probe


def synthetic_iv():
    voltage = np.linspace(-5.0, 20.0, 101)
    current = -1.0e-3 + 2.0e-5 * np.exp((voltage - 10.0) / 3.0)
    return voltage, current


def test_langmuir_probe_recovers_floating_potential_and_electron_temperature():
    voltage, current = synthetic_iv()
    # This variant crosses zero within the measured voltage sweep; the default
    # synthetic amplitude intentionally does not, and should report no crossing.
    current = -1.0e-3 + 4.0e-5 * np.exp((voltage - 10.0) / 3.0)
    result = analyze_langmuir_probe(
        voltage,
        current,
        electron_fit_range=(10.0, 20.0),
        ion_fit_range=(-5.0, 0.0),
    )
    assert result["floating_potential_V"] is not None
    assert result["electron_temperature_eV"] == pytest.approx(3.0, rel=0.04)
    assert result["electron_retarding_fit_R2"] > 0.999
    assert result["ion_saturation_current_A"] < 0


def test_langmuir_probe_optional_density_estimate_requires_fit_and_geometry():
    voltage, current = synthetic_iv()
    result = analyze_langmuir_probe(
        voltage,
        current,
        electron_fit_range=(2.0, 10.0),
        ion_fit_range=(-5.0, -1.0),
        area_m2=1.0e-5,
        ion_mass_amu=2.0,
        ti_eV=1.0,
        sheath_factor=0.61,
        gamma_i=1.0,
    )
    assert result["ion_density_estimate_m-3"] > 0
    assert result["ion_sound_speed_m_s"] > 0


def test_langmuir_probe_accepts_ion_positive_circuit_convention():
    voltage, current = synthetic_iv()
    result = analyze_langmuir_probe(
        voltage,
        -current,
        electron_fit_range=(2.0, 10.0),
        ion_fit_range=(-5.0, -1.0),
        current_convention="ion_positive",
    )
    assert result["current_convention"] == "ion_positive"
    assert result["electron_temperature_eV"] == pytest.approx(3.0, rel=0.04)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"area_m2": 1.0e-5},
        {"ion_mass_amu": 2.0},
        {"area_m2": 0.0, "ion_mass_amu": 2.0},
        {"area_m2": 1.0e-5, "ion_mass_amu": -2.0},
        {"current_convention": "unknown"},
    ],
)
def test_langmuir_probe_rejects_invalid_model_inputs(kwargs):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError):
        analyze_langmuir_probe(voltage, current, **kwargs)


def test_langmuir_probe_rejects_mismatched_and_undersampled_sweeps():
    with pytest.raises(ValueError, match="equal lengths"):
        analyze_langmuir_probe([0, 1, 2], [0, 1])
    with pytest.raises(ValueError, match="8 valid"):
        analyze_langmuir_probe([0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6])


def test_langmuir_probe_without_fit_windows_returns_labelled_tail_estimates():
    voltage = np.linspace(-10.0, 10.0, 101)
    current = 2e-3 * np.tanh(voltage / 3.0) + 1e-4 * voltage
    result = analyze_langmuir_probe(voltage, current)
    assert np.isfinite(result["ion_saturation_current_A"])
    assert np.isfinite(result["electron_saturation_current_A"])
    assert result["ion_saturation_current_method"] == "median_lowest_voltage_tail_fallback"
    assert result["electron_saturation_current_method"] == "median_highest_voltage_tail_descriptive"
    assert result["electron_temperature_eV"] is None
    assert "interpretation_warning" in result


@pytest.mark.parametrize("voltage,current", [
    (np.zeros((3, 3)), np.zeros((3, 3))),
    (np.arange(8).reshape(2, 4), np.arange(8)),
    (np.arange(8), np.arange(8).reshape(2, 4)),
])
def test_langmuir_probe_rejects_non_one_dimensional_sweeps(voltage, current):
    with pytest.raises(ValueError, match="one-dimensional"):
        analyze_langmuir_probe(voltage, current)


def test_langmuir_probe_selected_ion_range_overrides_tail_fallback():
    voltage = np.linspace(-5.0, 20.0, 101)
    current = -1.0e-3 + 2.0e-5 * np.exp((voltage - 10.0) / 3.0)
    result = analyze_langmuir_probe(
        voltage, current,
        electron_fit_range=(10.0, 20.0),
        ion_fit_range=(-5.0, 0.0),
    )
    assert result["ion_saturation_current_method"] == "median_user_selected_ion_fit_range"
    selected = (voltage >= -5.0) & (voltage <= 0.0)
    assert result["ion_saturation_current_A"] == pytest.approx(float(np.median(current[selected])), abs=1e-12)


def test_langmuir_probe_does_not_infer_density_from_tail_fallback():
    voltage = np.linspace(-10.0, 10.0, 101)
    current = 2e-3 * np.tanh(voltage / 3.0) + 1e-4 * voltage
    result = analyze_langmuir_probe(
        voltage, current, area_m2=1e-5, ion_mass_amu=2.0
    )
    assert "ion_density_estimate_m-3" not in result
    assert result["ion_density_status"].startswith("not_estimated:")



def test_langmuir_probe_rejects_ion_mass_without_collection_area():
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="supplied together"):
        analyze_langmuir_probe(voltage, current, ion_mass_amu=2.0)


def test_langmuir_probe_rejects_collection_area_without_ion_mass():
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="supplied together"):
        analyze_langmuir_probe(voltage, current, area_m2=1.0e-5)


@pytest.mark.parametrize("fit_range", [(100.0, 120.0), (-120.0, -100.0)])
def test_langmuir_probe_rejects_fit_windows_outside_measured_sweep(fit_range):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="overlap"):
        analyze_langmuir_probe(voltage, current, ion_fit_range=fit_range)
    with pytest.raises(ValueError, match="overlap"):
        analyze_langmuir_probe(voltage, current, electron_fit_range=fit_range)


@pytest.mark.parametrize("fit_range", [(1.0,), (1.0, 2.0, 3.0), (float("nan"), 2.0)])
def test_langmuir_probe_rejects_malformed_fit_windows(fit_range):
    voltage, current = synthetic_iv()
    with pytest.raises(ValueError, match="two finite voltage limits"):
        analyze_langmuir_probe(voltage, current, electron_fit_range=fit_range)

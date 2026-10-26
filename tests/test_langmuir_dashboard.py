"""Interactive Streamlit regression test for the Langmuir probe workflow."""
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_probe_analysis_dashboard_can_run_and_render_langmuir_results():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2.0e-3 + 2.5e-4 * np.exp((voltage + 10.0) / 4.0)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    app.session_state["tis_df"] = pd.DataFrame({"voltage_V": voltage, "current_A": current})
    app.session_state["tis_filename"] = "synthetic_langmuir_iv.csv"
    app.session_state["tis_sha"] = "synthetic-test"
    app.radio(key="workspace_page").set_value("Probe Analysis")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    app.selectbox(key="probe_v").set_value("voltage_V")
    app.selectbox(key="probe_i").set_value("current_A")
    app.checkbox(key="probe_enable_electron_fit").check()
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    analyze_button = next(button for button in app.button if button.label == "Analyze probe I–V")
    analyze_button.click()
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    assert len(app.json) >= 1
    assert any("electron_temperature_eV" in str(item.value) for item in app.json)

def test_langmuir_dashboard_rejects_columns_without_finite_paired_samples():
    voltage = np.linspace(-4.0, 4.0, 17)
    current = np.full_like(voltage, np.nan)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    app.session_state["tis_df"] = pd.DataFrame({"voltage_V": voltage, "current_A": current})
    app.session_state["tis_filename"] = "invalid_langmuir_iv.csv"
    app.session_state["tis_sha"] = "synthetic-invalid-test"
    app.radio(key="workspace_page").set_value("Probe Analysis")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    app.selectbox(key="probe_v").set_value("voltage_V")
    app.selectbox(key="probe_i").set_value("current_A")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    assert any(
        "at least eight finite paired samples" in str(item.value)
        for item in app.error
    )

def test_langmuir_dashboard_normalizes_ion_positive_current_polarity():
    voltage = np.linspace(-10.0, 10.0, 81)
    electron_positive_current = -2.0e-3 + 2.5e-4 * np.exp((voltage + 10.0) / 4.0)
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    app.session_state["tis_df"] = pd.DataFrame({
        "voltage_V": voltage,
        "current_A": -electron_positive_current,
    })
    app.session_state["tis_filename"] = "synthetic_ion_positive_langmuir_iv.csv"
    app.session_state["tis_sha"] = "synthetic-ion-positive-test"
    app.radio(key="workspace_page").set_value("Probe Analysis")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    app.selectbox(key="probe_v").set_value("voltage_V")
    app.selectbox(key="probe_i").set_value("current_A")
    app.radio(key="probe_current_convention").set_value("ion_positive")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    next(button for button in app.button if button.label == "Analyze probe I–V").click()
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    payloads = [str(item.value) for item in app.json]
    assert any("current_convention" in payload and "ion_positive" in payload for payload in payloads)
    assert any("single_zero_crossing" in payload for payload in payloads)


def test_langmuir_dashboard_requires_eight_distinct_voltage_values_among_finite_pairs():
    voltage = np.array([-4., -3., -2., -1., 0., 1., 2., 3., 3., 3.])
    current = np.array([np.nan, -3., -2., -1., 0., 1., 2., 3., 4., 5.])
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)

    app.session_state["tis_df"] = pd.DataFrame({"voltage_V": voltage, "current_A": current})
    app.session_state["tis_filename"] = "duplicate_paired_voltage.csv"
    app.session_state["tis_sha"] = "synthetic-duplicate-paired-voltage"
    app.radio(key="workspace_page").set_value("Probe Analysis")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    app.selectbox(key="probe_v").set_value("voltage_V")
    app.selectbox(key="probe_i").set_value("current_A")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    assert any("eight distinct voltage values among those pairs" in str(item.value) for item in app.error)


def test_langmuir_dashboard_reports_scientific_fit_quality_and_density_preconditions():
    from tis.probe_analysis import langmuir_iv_analysis
    import pytest

    voltage = np.linspace(-10.0, 10.0, 101)
    current = -2.0e-3 + 2.5e-4 * np.exp((voltage + 10.0) / 4.0)
    fitted = langmuir_iv_analysis(voltage, current, electron_fit_range=(-2.0, 8.0))
    assert fitted["electron_temperature_eV"] == pytest.approx(4.0, rel=0.03)
    assert fitted["electron_retarding_fit_R2"] > 0.999
    assert fitted["electron_temperature_standard_error_eV"] is not None
    assert fitted["interpretation_warning"]

    with pytest.raises(ValueError, match="requires a valid positive electron temperature"):
        langmuir_iv_analysis(
            voltage, current, ion_fit_range=(-10.0, -7.0),
            area_m2=1e-5, ion_mass_amu=2.0,
        )
    density = langmuir_iv_analysis(
        voltage, current, electron_fit_range=(-2.0, 8.0),
        ion_fit_range=(-10.0, -7.0), area_m2=1e-5, ion_mass_amu=2.0,
    )
    assert density["ion_density_estimate_m-3"] > 0
    assert density["ion_sound_speed_m_s"] > 0


def test_langmuir_dashboard_tracks_duplicate_and_nonfinite_input_pairs():
    from tis.probe_analysis import langmuir_iv_analysis

    voltage = np.linspace(-10.0, 10.0, 101)
    current = -2.0e-3 + 2.5e-4 * np.exp((voltage + 10.0) / 4.0)
    voltage = np.r_[voltage, voltage[10], np.nan]
    current = np.r_[current, current[10] * 1.1, 3.0]
    result = langmuir_iv_analysis(voltage, current)
    quality = result["input_quality"]
    assert quality["input_pair_count"] == len(voltage)
    assert quality["discarded_nonfinite_pair_count"] == 1
    assert quality["duplicate_voltage_pair_count_aggregated"] == 1
    assert quality["processed_unique_voltage_count"] == 101


def test_langmuir_dashboard_rejects_invalid_fit_windows_and_partial_geometry():
    import pytest
    from tis.probe_analysis import langmuir_iv_analysis

    voltage = np.linspace(-10.0, 10.0, 101)
    current = -2.0e-3 + 2.5e-4 * np.exp((voltage + 10.0) / 4.0)
    for fit_range in ((1.0, 1.0), (np.nan, 2.0), (2.0, 1.0), ("a", 2.0)):
        with pytest.raises(ValueError, match="electron_fit_range"):
            langmuir_iv_analysis(voltage, current, electron_fit_range=fit_range)
    with pytest.raises(ValueError, match="supplied together"):
        langmuir_iv_analysis(voltage, current, area_m2=1e-5)
    with pytest.raises(ValueError, match="current_convention"):
        langmuir_iv_analysis(voltage, current, current_convention="automatic")

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

"""Interactive Streamlit regression test for the Langmuir probe workflow."""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_probe_analysis_dashboard_can_run_and_render_langmuir_results():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -2.0e-3 + 2.5e-4 * np.exp((voltage + 10.0) / 4.0)
    app = AppTest.from_file("app.py", default_timeout=120).run()
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

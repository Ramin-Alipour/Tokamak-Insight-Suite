"""Interactive Streamlit regressions for the ECE analysis workflow."""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _open_ece():
    app = AppTest.from_file("app.py", default_timeout=120).run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert app.selectbox(key="ece_workflow_mode")
    return app


def test_ece_brightness_temperature_dashboard_and_export():
    app = _open_ece()
    app.text_input(key="ece_radiance").set_value("1e-10, 2e-10, 3e-10")
    app.button(key="ece_run_brightness").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("ECE brightness-temperature analysis completed" in x.value for x in app.success)
    assert any(x.label == "Export ECE brightness-temperature result" for x in app.get("download_button"))


def test_ece_resonance_mapping_dashboard():
    app = _open_ece()
    app.selectbox(key="ece_workflow_mode").set_value("Resonance mapping").run()
    app.button(key="ece_run_mapping").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("ECE resonance mapping completed" in x.value for x in app.success)
    assert any(x.label == "Export ECE resonance mapping" for x in app.get("download_button"))


def test_ece_fluctuation_dashboard_with_loaded_data():
    app = _open_ece()
    t = np.arange(128, dtype=float) * 1e-6
    frame = pd.DataFrame({
        "time_s": t,
        "ece_80": np.sin(2 * np.pi * 5e3 * t),
        "ece_90": 0.5 * np.sin(2 * np.pi * 8e3 * t),
        "ece_100": 0.25 * np.sin(2 * np.pi * 12e3 * t),
    })
    app.session_state["tis_df"] = frame
    app.run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(key="ece_workflow_mode").set_value("Channel fluctuations / PSD").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(key="ece_time_col").set_value("time_s").run()
    app.multiselect(key="ece_signal_cols").set_value(["ece_80", "ece_90", "ece_100"]).run()
    app.button(key="ece_run_fluctuations").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("ECE fluctuation analysis completed" in x.value for x in app.success)
    assert any(x.label == "Export ECE fluctuation / PSD result" for x in app.get("download_button"))

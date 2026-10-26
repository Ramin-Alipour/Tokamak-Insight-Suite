"""Interactive Streamlit regressions for the Mirnov / pickup-coil workflow."""
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _open_mirnov():
    app = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not app.exception, [str(x.value) for x in app.exception]
    t = np.arange(128, dtype=float) / 20000.0
    app.session_state["tis_df"] = pd.DataFrame({"time_s": t, "mirnov_V": np.sin(2 * np.pi * 1250 * t)})
    app.run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.radio(key="workspace_page").set_value("Spectral Analysis").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.radio(label="Workflow").set_value("Mirnov / pickup coils").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    return app


def test_mirnov_dashboard_analysis_and_export():
    app = _open_mirnov()
    t = np.arange(2048, dtype=float) / 20000.0
    app.session_state["tis_df"] = pd.DataFrame({
        "time_s": t,
        "mirnov_V": 0.4 * np.sin(2 * np.pi * 1250 * t),
    })
    app.run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(key="mirnov_time").set_value("time_s").run()
    app.selectbox(key="mirnov_voltage").set_value("mirnov_V").run()
    app.button(key="mirnov_analyze").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("Mirnov signal analysis completed" in x.value for x in app.success)
    assert any(x.label == "Export Mirnov analysis JSON" for x in app.get("download_button"))


def test_mirnov_dashboard_geometry_and_relative_perturbation():
    app = _open_mirnov()
    t = np.arange(256, dtype=float) * 1e-5
    app.session_state["tis_df"] = pd.DataFrame({
        "time_s": t,
        "mirnov_V": np.full_like(t, -10.0 * 1e-4 * 0.2),
    })
    app.run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(key="mirnov_time").set_value("time_s").run()
    app.selectbox(key="mirnov_voltage").set_value("mirnov_V").run()
    app.checkbox(key="mirnov_geometry").check().run()
    app.checkbox(key="mirnov_integrate").check().run()
    app.button(key="mirnov_analyze").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("Mirnov signal analysis completed" in x.value for x in app.success)
    assert any(x.label == "Export Mirnov analysis JSON" for x in app.get("download_button"))


def test_mirnov_dashboard_multichannel_mode_candidate():
    app = _open_mirnov()
    fs = 20000.0
    t = np.arange(2048, dtype=float) / fs
    angles = np.array([0.0, 1.2, 2.4])
    app.session_state["tis_df"] = pd.DataFrame({
        "time_s": t,
        "mirnov_a_V": np.sin(2 * np.pi * 750 * t + 2 * angles[0]),
        "mirnov_b_V": np.sin(2 * np.pi * 750 * t + 2 * angles[1]),
        "mirnov_c_V": np.sin(2 * np.pi * 750 * t + 2 * angles[2]),
    })
    app.run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(key="mirnov_time").set_value("time_s").run()
    app.selectbox(key="mirnov_voltage").set_value("mirnov_a_V").run()
    app.checkbox(key="mirnov_array_mode").check().run()
    app.multiselect(key="mirnov_array_channels").set_value(
        ["mirnov_a_V", "mirnov_b_V", "mirnov_c_V"]
    ).run()
    app.text_input(key="mirnov_array_angles").set_value("0, 1.2, 2.4").run()
    app.button(key="mirnov_analyze").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("Mirnov signal analysis completed" in x.value for x in app.success)
    assert any("Toroidal mode candidate" in x.value for x in app.metric)
    assert any(x.label == "Export Mirnov analysis JSON" for x in app.get("download_button"))


def test_mirnov_dashboard_rejects_incomplete_array_configuration():
    app = _open_mirnov()
    app.checkbox(key="mirnov_array_mode").check().run()
    app.multiselect(key="mirnov_array_channels").set_value(["mirnov_V"]).run()
    app.text_input(key="mirnov_array_angles").set_value("0").run()
    app.button(key="mirnov_analyze").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("at least three" in x.value for x in app.error)

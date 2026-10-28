"""Interactive Streamlit regressions for the ECE analysis workflow."""
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _open_ece():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.radio(key="workspace_page").set_value("Plasma Parameters").run()
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
    assert app.selectbox(key="ece_cross_pair")
    assert any("Export ECE fluctuation / PSD result" == x.label for x in app.get("download_button"))


def test_ece_fluctuation_dashboard_rejects_missing_samples_without_compressing_timebase():
    app = _open_ece()
    t = np.arange(32, dtype=float) * 1e-6
    frame = pd.DataFrame({
        "time_s": t,
        "ece_80": np.sin(2 * np.pi * 5e3 * t),
        "ece_90": np.sin(2 * np.pi * 8e3 * t),
        "ece_100": np.sin(2 * np.pi * 12e3 * t),
    })
    frame.loc[10, "ece_90"] = np.nan
    app.session_state["tis_df"] = frame
    app.run()
    app.selectbox(key="ece_workflow_mode").set_value("Channel fluctuations / PSD").run()
    app.selectbox(key="ece_time_col").set_value("time_s").run()
    app.multiselect(key="ece_signal_cols").set_value(["ece_80", "ece_90", "ece_100"]).run()
    app.button(key="ece_run_fluctuations").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("Missing/non-numeric samples are not dropped" in x.value for x in app.error)
    assert not any("ECE fluctuation analysis completed" in x.value for x in app.success)

def test_ece_temperature_profile_dashboard_from_uploaded_multichannel_data():
    app = _open_ece()
    t = np.arange(32, dtype=float) * 1e-6
    frame = pd.DataFrame({
        "time_s": t,
        "ece_80": np.full(t.size, 10.0),
        "ece_90": np.full(t.size, 20.0),
        "ece_100": np.full(t.size, 30.0),
    })
    app.session_state["tis_df"] = frame
    app.run()
    app.selectbox(key="ece_workflow_mode").set_value("Temperature profile reconstruction").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(key="ece_profile_time_col").set_value("time_s").run()
    app.multiselect(key="ece_profile_signal_cols").set_value(["ece_80", "ece_90", "ece_100"]).run()
    assert not any("gain to Tb" in x.label or "offset to Tb" in x.label for x in app.text_input)
    app.checkbox(key="ece_profile_thick").check().run()
    app.selectbox(key="ece_profile_method").set_value("Shape-preserving PCHIP").run()
    app.number_input(key="ece_model_discrepancy").set_value(0.05).run()
    app.number_input(key="ece_profile_window_start").set_value(0.0).run()
    app.number_input(key="ece_profile_window_end").set_value(10e-6).run()
    app.button(key="ece_run_profile").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("11 samples" in x.value for x in app.success)
    assert any(x.label == "Export ECE electron-temperature profile" for x in app.get("download_button"))




def test_ece_resonance_dashboard_relativistic_sensitivity_option():
    app = _open_ece()
    app.selectbox(key="ece_workflow_mode").set_value("Resonance mapping").run()
    app.checkbox(key="ece_map_rel").check().run()
    app.number_input(key="ece_map_energy").set_value(10000.0).run()
    app.number_input(key="ece_map_beta").set_value(0.1).run()
    app.button(key="ece_run_mapping").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("ECE resonance mapping completed" in x.value for x in app.success)



def test_ece_receiver_response_convolution_dashboard():
    app = _open_ece()
    app.selectbox(key="ece_workflow_mode").set_value("Receiver response convolution").run()
    app.text_area(key="ece_response_spectrum_frequency").set_value("80000000000,85000000000,90000000000,95000000000,100000000000").run()
    app.text_area(key="ece_response_spectrum_radiance").set_value("1e-10,1e-10,1e-10,1e-10,1e-10").run()
    app.text_area(key="ece_response_frequency").set_value("85000000000,90000000000,95000000000").run()
    app.text_area(key="ece_response_weights").set_value("0,1,0;0,0.5,0.5").run()
    app.button(key="ece_response_run").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("receiver-response convolution completed" in x.value for x in app.success)
    assert any(x.label == "Export ECE receiver-response convolution" for x in app.get("download_button"))

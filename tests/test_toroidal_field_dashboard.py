"""Interactive Streamlit regression for registered toroidal-field analysis."""
import json
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_toroidal_field_registered_analysis_executes_and_exports():
    app = AppTest.from_file("../app.py", default_timeout=120).run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(label="Select workflow").set_value("Registered analysis modules").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.selectbox(label="Analysis").set_value(8).run()
    assert any("Toroidal-field coils / magnetic probes" in x.value for x in app.get("caption"))
    args = {
        "time_s": [0.0, 0.1, 0.2],
        "signal": [100.0, 110.0, 120.0],
        "calibration_T_per_unit": 0.01,
        "signal_kind": "current",
        "baseline": 100.0,
        "signal_uncertainty": [1.0, 1.0, 1.0],
        "calibration_uncertainty_T_per_unit": 0.001,
    }
    app.text_area(label="Function arguments (JSON)").set_value(json.dumps(args)).run()
    app.button(label="Run registered analysis").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any(x.value == "Analysis executed." for x in app.success)
    assert any(x.label == "Export community contribution bundle (JSON)" for x in app.get("download_button"))
    assert any("B_toroidal_T" in str(x.value) for x in app.get("json")) or any("B_toroidal_T" in str(x.value) for x in app.get("markdown"))

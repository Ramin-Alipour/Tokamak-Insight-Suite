"""Interactive Streamlit regression for registered toroidal-field analysis."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def test_toroidal_field_registered_analysis_executes_and_exports():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    next(item for item in app.selectbox if item.label == "Select workflow").set_value("Registered analysis modules").run()
    assert not app.exception, [str(x.value) for x in app.exception]
    next(item for item in app.selectbox if item.label == "Analysis").set_value(8).run()
    assert any("Toroidal-field coils / magnetic probes" in x.value for x in app.get("caption"))
    args = {
        "time_s": [0.0, 0.1, 0.2],
        "signal": [100.0, 110.0, 120.0],
        "calibration_T_per_unit": 0.01,
        "signal_kind": "current",
        "baseline": 100.0,
        "signal_uncertainty": [1.0, 1.0, 1.0],
        "calibration_uncertainty_T_per_unit": 0.001,
        "pickup_voltage_V": [4.0, 4.0, 4.0],
        "pickup_sensitivity_V_per_T_per_s": 2.0,
        "initial_field_T": 3.0,
        "pickup_voltage_uncertainty_V": [0.1, 0.1, 0.1],
        "pickup_sensitivity_uncertainty_V_per_T_per_s": 0.1,
        "initial_field_uncertainty_T": 0.05,
    }
    next(item for item in app.text_area if item.label == "Function arguments (JSON)").set_value(json.dumps(args)).run()
    app.button(label="Run registered analysis").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any(x.value == "Analysis executed." for x in app.success)
    assert any(x.label == "Export community contribution bundle (JSON)" for x in app.get("download_button"))
    rendered = "\n".join(str(x.value) for x in app.get("json") + app.get("markdown"))
    assert "B_toroidal_T" in rendered
    assert "B_toroidal_integrated_T" in rendered

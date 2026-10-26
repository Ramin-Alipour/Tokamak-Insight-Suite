"""Interactive Streamlit acceptance tests for Rogowski-current analysis."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest


def _loaded_app():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    t = np.linspace(0.0, 1.0, 101)
    app.session_state["tis_df"] = pd.DataFrame({
        "time_s": t,
        "derivative_V": np.full_like(t, 2.0),
        "integrator_V": 5.0 + 2.0 * t,
    })
    app.session_state["tis_filename"] = "synthetic_rogowski.csv"
    app.session_state["tis_sha"] = "synthetic-rogowski-test"
    app.radio(key="workspace_page").set_value("Time Series & MHD")
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    app.selectbox(key="ts_t").set_value("time_s")
    app.selectbox(key="ts_s").set_value("derivative_V")
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    return app


def _rogowski_metadata(app):
    """Read st.json output regardless of Streamlit's AppTest value encoding."""
    for item in app.get("json"):
        value = item.value
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (TypeError, ValueError):
                continue
        if isinstance(value, dict) and value.get("diagnostic") == "Rogowski coil":
            return value
    pytest.fail("Rogowski JSON metadata was not rendered by the dashboard")


def test_rogowski_dashboard_runs_derivative_mode_and_exports_current():
    app = _loaded_app()
    mode = next(item for item in app.radio if item.key == "rogowski_acquisition_mode")
    mode.set_value("Derivative output (signal ∝ dIp/dt; integrate once)")
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    next(item for item in app.button if item.label == "Analyze Rogowski / calibrated current trace").click()
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    rendered = "\\n".join(str(item.value) for item in app.get("json") + app.get("markdown"))
    assert "derivative_output" in rendered
    assert any("Rogowski-coil analysis completed within the declared calibrated-input scope." in item.value for item in app.success)
    assert "rogowski coil" in rendered.lower()
    metadata = _rogowski_metadata(app)
    assert metadata["n_samples"] == 101
    assert metadata["current_final_A"] == pytest.approx(2.0)
    assert metadata["current_min_A"] == pytest.approx(0.0)
    assert metadata["current_max_A"] == pytest.approx(2.0)
    assert any(item.label == "Download Rogowski current series" for item in app.get("download_button"))
    assert any(item.label == "Export Rogowski analysis manifest" for item in app.get("download_button"))


def test_rogowski_dashboard_integrator_output_does_not_integrate_twice():
    app = _loaded_app()
    app.selectbox(key="ts_s").set_value("integrator_V")
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    next(item for item in app.radio if item.key == "rogowski_acquisition_mode").set_value(
        "External integrator output (signal ∝ Ip; do not integrate)"
    )
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    next(item for item in app.button if item.label == "Analyze Rogowski / calibrated current trace").click()
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    rendered = "\\n".join(str(item.value) for item in app.get("json") + app.get("markdown"))
    assert "integrator_output" in rendered
    assert any("Rogowski-coil analysis completed within the declared calibrated-input scope." in item.value for item in app.success)
    assert "integration_applied" in rendered
    assert "false" in rendered.lower()
    metadata = _rogowski_metadata(app)
    assert metadata["current_min_A"] == pytest.approx(5.0)
    assert metadata["current_final_A"] == pytest.approx(7.0)
    assert metadata["integration_applied"] is False


def test_rogowski_dashboard_rejects_zero_calibration_scale_without_crashing():
    app = _loaded_app()
    app.number_input(key="int_scale").set_value(0.0)
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    next(item for item in app.button if item.label == "Analyze Rogowski / calibrated current trace").click()
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    assert any("Calibration scale must be finite and nonzero" in item.value for item in app.error)
    assert not any("Rogowski-coil analysis completed" in item.value for item in app.success)

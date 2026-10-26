"""Interactive Streamlit regressions for flux-loop reconstruction."""
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _open_flux_loop():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=120,
    ).run()
    assert not app.exception, [str(item.value) for item in app.exception]

    app.session_state["tis_df"] = pd.DataFrame(
        {"time_s": [0.0, 0.5, 1.0], "loop_voltage_V": [2.0, 2.0, 2.0]}
    )
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]

    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    assert not app.exception, [str(item.value) for item in app.exception]
    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Flux-loop magnetic-flux reconstruction"
    ).run()
    assert not app.exception, [str(item.value) for item in app.exception]
    return app


def test_flux_loop_workflow_is_available_and_renders_without_data():
    app = AppTest.from_file(
        str(Path(__file__).resolve().parents[1] / "app.py"),
        default_timeout=120,
    ).run()
    assert not app.exception

    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    assert not app.exception

    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Flux-loop magnetic-flux reconstruction"
    ).run()
    assert not app.exception
    assert any("calibrated flux-loop voltage" in item.value.lower() for item in app.info)
    assert any("initial flux" in item.value.lower() for item in app.warning)


def test_flux_loop_dashboard_runs_reconstruction_and_offers_exports():
    app = _open_flux_loop()
    app.selectbox(key="tis_flux_loop_time").set_value("time_s").run()
    app.selectbox(key="tis_flux_loop_voltage").set_value("loop_voltage_V").run()
    app.number_input(key="tis_flux_loop_turns").set_value(2).run()
    app.button(key="run_flux_loop_analysis").click().run()

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any(
        "Flux-loop analysis completed within the declared calibrated-input scope."
        in item.value
        for item in app.success
    )
    assert any("Flux change (final)" in item.label for item in app.metric)
    assert any(
        item.label == "Download flux-loop results (CSV)"
        for item in app.get("download_button")
    )
    assert any(
        item.label == "Export flux-loop result and provenance (JSON)"
        for item in app.get("download_button")
    )


def test_flux_loop_dashboard_reports_invalid_time_series_without_crashing():
    app = _open_flux_loop()
    app.session_state["tis_df"] = pd.DataFrame(
        {"time_s": [0.0, 0.5, 0.5], "loop_voltage_V": [2.0, 2.0, 2.0]}
    )
    app.run()
    assert not app.exception, [str(item.value) for item in app.exception]
    app.selectbox(key="tis_flux_loop_time").set_value("time_s").run()
    app.selectbox(key="tis_flux_loop_voltage").set_value("loop_voltage_V").run()
    app.button(key="run_flux_loop_analysis").click().run()

    assert not app.exception, [str(item.value) for item in app.exception]
    assert any("strictly increasing" in item.value for item in app.error)

"""Interactive dashboard acceptance tests for analysis 15 poloidal flux."""
from pathlib import Path
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


def _open_workflow():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    assert not app.exception, [str(x.value) for x in app.exception]
    app.session_state["tis_df"] = pd.DataFrame({
        "R_m": [1., 2., 1., 2., 1., 2.],
        "Z_m": [0., 0., 1., 1., 2., 2.],
        "B_R_T": [0.]*6,
        "B_Z_T": [2.]*6,
    })
    app.run()
    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    next(x for x in app.selectbox if x.label == "Select workflow").set_value(
        "Poloidal magnetic-flux map reconstruction"
    ).run()
    return app


def test_poloidal_flux_workflow_renders_without_input_data():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120).run()
    assert not app.exception
    app.radio(key="workspace_page").set_value("Advanced Analysis").run()
    next(x for x in app.selectbox if x.label == "Select workflow").set_value(
        "Poloidal magnetic-flux map reconstruction"
    ).run()
    assert not app.exception
    assert any("calibrated B_R and B_Z grids" in x.value for x in app.caption)


def test_dashboard_reconstructs_flux_and_exposes_exports():
    app = _open_workflow()
    app.button(key="run_poloidal_flux").click().run()
    assert not app.exception, [str(x.value) for x in app.exception]
    assert any("completed within the declared calibrated-input scope" in x.value for x in app.success)
    assert any(x.label == "Flux minimum (Wb)" for x in app.metric)
    assert any(x.label == "Export poloidal flux map (CSV)" for x in app.get("download_button"))
    assert any(x.label == "Export poloidal flux result and diagnostics (JSON)" for x in app.get("download_button"))


def test_dashboard_rejects_incomplete_coordinate_grid():
    app = _open_workflow()
    app.session_state["tis_df"] = app.session_state["tis_df"].iloc[:-1].copy()
    app.run()
    app.button(key="run_poloidal_flux").click().run()
    assert not app.exception
    assert any("Grid is incomplete" in x.value for x in app.error)

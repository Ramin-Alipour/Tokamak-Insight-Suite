import numpy as np
import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_home_renders_without_uncaught_exceptions():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception


def test_probe_page_default_geometry_settings_do_not_crash():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    voltage = np.linspace(-5.0, 5.0, 101)
    current = 0.01 * np.exp(voltage / 2.0) - 0.02
    app.session_state["jump"] = "Probe Analysis"
    app.session_state["tis_df"] = pd.DataFrame({"voltage_V": voltage, "current_A": current})
    app.session_state["tis_raw"] = b"synthetic regression data"
    app.session_state["tis_filename"] = "synthetic_probe.csv"
    app.session_state["tis_sha"] = "test-sha256"
    app.session_state["tis_imported_at_utc"] = "2026-10-14T00:00:00+00:00"
    app.run()
    assert not app.exception
    assert any("Analyze probe I–V" in button.label for button in app.button)

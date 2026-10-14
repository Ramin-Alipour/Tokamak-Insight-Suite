from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_dashboard_renders_home_without_streamlit_exceptions():
    app = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Tokamak Insight Suite" in str(item.value) for item in app.markdown)


def test_probe_page_renders_with_default_geometry_without_error():
    import numpy as np
    import pandas as pd

    voltage = np.linspace(-20.0, 20.0, 101)
    current = 0.01 * np.exp(voltage / 10.0) - 0.02
    app = AppTest.from_file("app.py", default_timeout=30)
    app.session_state["tis_df"] = pd.DataFrame({"voltage_V": voltage, "current_A": current})
    app.session_state["tis_filename"] = "synthetic_probe.csv"
    app.session_state["tis_raw"] = b"synthetic"
    app.session_state["tis_sha"] = "synthetic"
    app.run()
    app.radio[0].set_value("Probe Analysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]

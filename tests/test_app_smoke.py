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
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.session_state["tis_df"] = pd.DataFrame({"voltage_V": voltage, "current_A": current})
    app.session_state["tis_filename"] = "synthetic_probe.csv"
    app.session_state["tis_raw"] = b"synthetic"
    app.session_state["tis_sha"] = "synthetic"
    app.run()
    app.radio[0].set_value("Probe Analysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]


import pytest

ALL_PAGES = [
    "Home", "Public Data Access", "Data Import & QC", "Time Series & MHD", "Events & Transients",
    "Spectral Analysis", "Probe Analysis", "Plasma Parameters", "Confinement & Scalings",
    "Equilibrium & Reconstruction", "Tomography & Bolometry", "Advanced Analysis",
    "Diagnostic Catalog", "Project Manager", "Reports & Export",
]


def _loaded_app():
    import numpy as np
    import pandas as pd

    fs = 1e5
    t = np.arange(4096) / fs
    frame = pd.DataFrame({
        "time_s": t, "mirnov0": np.sin(2 * np.pi * 8e3 * t), "mirnov1": np.cos(2 * np.pi * 8e3 * t),
        "ip": np.where(t < 0.02, 2e5, np.maximum(2e5 * (1 - (t - 0.02) / 0.01), 0)),
        "voltage_V": np.linspace(-20, 20, 4096), "current_A": 0.01 * np.exp(np.linspace(-20, 20, 4096) / 10) - 0.02,
    })
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic.csv"
    app.session_state["tis_raw"] = b"synthetic"
    app.session_state["tis_sha"] = "synthetic"
    return app


@pytest.mark.parametrize("page", ALL_PAGES)
def test_every_workspace_page_renders_with_loaded_data(page):
    app = _loaded_app()
    app.run()
    app.radio[0].set_value(page).run()
    assert not app.exception, (page, [str(error.value) for error in app.exception])


def test_catalog_displays_analysis_contracts_separately_from_reference_entries():
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Diagnostic Catalog").run()
    catalog_radio = next(item for item in app.radio if item.label == "Catalog")
    catalog_radio.set_value("Analysis entry points").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("callable analysis entry points" in str(item.value).lower() for item in app.markdown)

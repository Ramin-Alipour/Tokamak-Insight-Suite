import numpy as np
import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_dashboard_home_renders_without_uncaught_exceptions():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception
    assert any("Tokamak Insight Suite" in str(item.value) for item in app.markdown)


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
    next(button for button in app.button if "Analyze probe I–V" in button.label).click().run()
    assert not app.exception


def test_spectral_page_renders_end_to_end_doppler_workflow():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    pixels = np.linspace(0.0, 100.0, 101)
    intensity = 2.0 + 10.0*np.exp(-0.5*((pixels-50.0)/8.0)**2)
    app.session_state["jump"] = "Spectral Analysis"
    app.session_state["tis_df"] = pd.DataFrame({"pixel": pixels, "intensity": intensity})
    app.session_state["tis_raw"] = b"synthetic spectral regression data"
    app.session_state["tis_filename"] = "synthetic_spectrum.csv"
    app.session_state["tis_sha"] = "test-spectrum-sha256"
    app.session_state["tis_imported_at_utc"] = "2026-10-14T00:00:00+00:00"
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Run end-to-end Doppler analysis" in button.label for button in app.button)


def test_cli_exposes_dashboard_command():
    from tis.cli import main
    import sys
    from unittest.mock import patch
    with patch.object(sys, "argv", ["tis", "--help"]):
        try:
            main()
        except SystemExit as exc:
            assert exc.code == 0

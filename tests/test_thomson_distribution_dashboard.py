"""Interactive Streamlit regression coverage for Thomson spectral distribution."""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from pathlib import Path

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_thomson_spectral_distribution_dashboard_executes_and_exports():
    wavelength = np.linspace(499.0, 501.0, 501)
    signal = (
        1000.0 * np.exp(-0.5 * ((wavelength - 500.0) / 0.18) ** 2)
        + 12.0 + 0.8 * (wavelength - 500.0)
    )
    frame = pd.DataFrame({"wavelength_nm": wavelength, "signal": signal})
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_thomson_spectrum.csv"
    app.session_state["tis_raw"] = b"synthetic Thomson spectrum"
    app.session_state["tis_sha"] = "synthetic-thomson-sha256"
    app.run()
    app.radio[0].set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "Thomson spectral distribution"
    ).run()

    wavelength_column = next(
        item for item in app.selectbox
        if item.label == "Calibrated Thomson wavelength (nm)"
    )
    signal_column = next(
        item for item in app.selectbox
        if item.label == "Thomson spectral signal"
    )
    wavelength_column.set_value("wavelength_nm")
    signal_column.set_value("signal")
    app.run()

    next(
        item for item in app.button
        if item.label == "Analyze Thomson spectral distribution"
    ).click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("spectral_centroid_nm" in str(item.value) for item in app.json)
    assert any(
        item.label == "Export Thomson spectral-distribution results"
        for item in app.download_button
    )
    result = app.session_state["tis_thomson_distribution"]["result"]
    assert result["spectral_centroid_nm"] == pytest.approx(500.0, abs=0.01)
    assert result["instrument_width_resolved"] is True

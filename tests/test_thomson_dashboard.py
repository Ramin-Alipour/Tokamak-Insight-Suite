"""End-to-end Streamlit regression tests for Thomson-scattering analysis workflows."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app_with_table(frame: pd.DataFrame) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_thomson_input.csv"
    app.session_state["tis_raw"] = frame.to_csv(index=False).encode("utf-8")
    app.session_state["tis_sha"] = "synthetic-thomson-regression"
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.radio[0].set_value("Advanced Analysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    next(item for item in app.selectbox if item.label == "Select workflow")
    return app


def test_thomson_calibrated_profile_dashboard_executes_and_exports():
    radius = np.linspace(0.0, 0.4, 9)
    frame = pd.DataFrame({
        "radius_m": radius,
        "density_m3": 2.0e19 * (1.0 - 0.5 * radius),
        "temperature_eV": 120.0 * (1.0 - 0.25 * radius),
        "density_sigma_m3": np.full(radius.size, 1.0e17),
        "temperature_sigma_eV": np.full(radius.size, 1.0),
    })
    app = _app_with_table(frame)
    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Thomson calibrated profile analysis"
    ).run()

    selectors = {item.label: item for item in app.selectbox}
    selectors["Radius / minor-radius coordinate (m)"].set_value("radius_m")
    selectors["Electron density (m⁻³)"].set_value("density_m3")
    selectors["Electron temperature (eV)"].set_value("temperature_eV")
    selectors["Optional density 1-sigma uncertainty column (m⁻³)"].set_value(
        "density_sigma_m3"
    )
    selectors["Optional temperature 1-sigma uncertainty column (eV)"].set_value(
        "temperature_sigma_eV"
    )
    app.run()
    next(item for item in app.button if item.label == "Analyze calibrated Thomson profiles").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Calibrated Thomson profile analysis completed." in str(item.value) for item in app.success)
    assert any(item.label == "Download Thomson profile results" for item in app.get("download_button"))
    assert any(item.label == "Export Thomson profile summary" for item in app.get("download_button"))


def test_thomson_raw_spectrum_dashboard_executes_fit_residuals_and_exports():
    wavelength = np.linspace(530.8, 533.2, 241)
    physical_sigma_nm = 532.0 * np.sqrt(4.0 * 0.01 / 510_998.95069) * np.sin(np.pi / 4.0)
    fitted_sigma_nm = np.sqrt(physical_sigma_nm**2 + 0.05**2)
    signal = (
        1000.0 * np.exp(-0.5 * ((wavelength - 532.0) / fitted_sigma_nm) ** 2)
        + 12.0 + 0.8 * (wavelength - 532.0)
    )
    frame = pd.DataFrame({"wavelength_nm": wavelength, "signal": signal})
    app = _app_with_table(frame)
    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Thomson raw-spectrum temperature fit"
    ).run()

    selectors = {item.label: item for item in app.selectbox}
    selectors["Wavelength (nm)"].set_value("wavelength_nm")
    selectors["Measured spectral signal"].set_value("signal")
    app.run()
    next(
        item for item in app.button
        if item.label == "Fit Thomson spectrum and estimate electron temperature"
    ).click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Thomson spectral fit completed." in str(item.value) for item in app.success)
    assert any(item.label == "Download Thomson spectrum fit and residuals" for item in app.get("download_button"))
    assert any(item.label == "Export Thomson fit summary" for item in app.get("download_button"))
    assert any("Electron temperature" in str(item.label) for item in app.metric)
    result = app.session_state.get("tis_thomson_spectrum_fit")
    if result is not None:
        assert result["electron_temperature_eV"] == pytest.approx(0.01, rel=0.05)

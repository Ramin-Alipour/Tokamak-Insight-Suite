"""Interactive acceptance tests for the Doppler ion-temperature workflow."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app():
    wavelength = np.linspace(529.8, 530.2, 401)
    ti = 40.0
    mass = 12.0
    e = 1.602176634e-19
    u = 1.66053906660e-27
    c = 299792458.0
    thermal_fwhm = 530.0e-9 / c * np.sqrt(8 * np.log(2) * ti * e / (mass * u)) * 1e9
    instrument_fwhm = 0.008
    observed_fwhm = np.hypot(thermal_fwhm, instrument_fwhm)
    sigma = observed_fwhm / 2.354820045
    center = 530.0 * (1 + 1.2e4 / c)
    intensity = 120 * np.exp(-0.5 * ((wavelength - center) / sigma) ** 2) + 4 + 0.2 * (wavelength - center)
    frame = pd.DataFrame({"wavelength_nm": wavelength, "intensity": intensity, "sigma": np.full(wavelength.size, 0.5)})
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_doppler.csv"
    app.session_state["tis_raw"] = frame.to_csv(index=False).encode()
    app.session_state["tis_sha"] = "synthetic-doppler-regression"
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.radio(key="workspace_page").set_value("Spectral Analysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    next(item for item in app.radio if item.label == "Workflow").set_value("Ion temperature from Doppler spectroscopy").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    selectors = {item.label: item for item in app.selectbox}
    selectors["Calibrated wavelength (nm)"].set_value("wavelength_nm")
    selectors["Emission intensity"].set_value("intensity")
    selectors["Per-sample intensity uncertainty σ (optional)"].set_value("sigma")
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.number_input(key="ti_doppler_inst").set_value(instrument_fwhm)
    return app, ti


def test_doppler_dashboard_fits_temperature_and_exports():
    app, expected_ti = _app()
    app.button(key="ti_doppler_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    result = app.session_state["tis_doppler_ti_result"]["result"]
    assert result["ion_temperature_eV"] == pytest.approx(expected_ti, rel=0.04)
    assert result["ion_temperature_uncertainty_eV"] >= 0
    assert result["fit_r_squared"] > 0.999
    assert len(result["fitted_intensity"]) == 401
    assert len(result["fit_residuals"]) == 401
    assert any("Doppler ion-temperature analysis completed" in str(item.value) for item in app.success)
    assert any(item.label == "Export Doppler ion-temperature result" for item in app.get("download_button"))


def test_doppler_dashboard_rejects_unresolved_width_and_clears_old_result():
    app, _ = _app()
    app.number_input(key="ti_doppler_inst").set_value(0.2)
    app.button(key="ti_doppler_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("unresolved" in str(item.value).lower() for item in app.error)
    assert "tis_doppler_ti_result" not in app.session_state


def test_doppler_dashboard_rejects_invalid_sample_uncertainty():
    app, _ = _app()
    frame = app.session_state["tis_df"].copy()
    frame.loc[20, "sigma"] = 0.0
    app.session_state["tis_df"] = frame
    app.run()
    app.button(key="ti_doppler_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("uncertainty" in str(item.value).lower() for item in app.error)
    assert "tis_doppler_ti_result" not in app.session_state


def test_doppler_dashboard_hides_stale_result_after_parameter_change():
    app, _ = _app()
    app.button(key="ti_doppler_run").click().run()
    assert "tis_doppler_ti_result" in app.session_state
    app.number_input(key="ti_doppler_mass").set_value(16.0).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert not any("Doppler ion-temperature analysis completed" in str(item.value) for item in app.success)

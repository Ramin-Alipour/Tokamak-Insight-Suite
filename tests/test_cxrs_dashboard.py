"""Interactive Streamlit acceptance tests for CXRS line-profile analysis."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _cxrs_app():
    wavelength = np.linspace(529.8, 530.2, 401)
    center = 530.0 * (1.0 + 2.5e4 / 299792458.0)
    intrinsic_sigma = 0.025
    instrument_sigma = 0.008
    observed_sigma = np.hypot(intrinsic_sigma, instrument_sigma)
    signal = (
        120.0 * np.exp(-0.5 * ((wavelength - center) / observed_sigma) ** 2)
        + 4.0 + 0.2 * (wavelength - center)
    )
    frame = pd.DataFrame({"wavelength_nm": wavelength, "cxrs_signal": signal,
                          "signal_sigma": np.full(wavelength.size, 0.5)})
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_cxrs.csv"
    app.session_state["tis_raw"] = frame.to_csv(index=False).encode("utf-8")
    app.session_state["tis_sha"] = "synthetic-cxrs-regression"
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.radio(key="workspace_page").set_value("Spectral Analysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    next(item for item in app.radio if item.label == "Workflow").set_value("CXRS line profile").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    selectors = {item.label: item for item in app.selectbox}
    selectors["Calibrated wavelength (nm)"].set_value("wavelength_nm")
    selectors["Spectral intensity"].set_value("cxrs_signal")
    selectors["Per-sample intensity uncertainty σ (optional)"].set_value("signal_sigma")
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    return app, center


def test_cxrs_dashboard_fits_line_reports_physics_and_exports():
    app, expected_center = _cxrs_app()
    app.button(key="cxrs_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    result = app.session_state["tis_cxrs_result"]["result"]
    assert any("CXRS line-profile analysis completed" in str(item.value) for item in app.success)
    assert result["wavelength_center_nm"] == pytest.approx(expected_center, abs=2e-5)
    assert result["line_of_sight_velocity_m_s"] == pytest.approx(2.5e4, abs=15)
    assert result["ion_temperature_eV"] > 0
    assert result["ion_temperature_uncertainty_eV"] >= 0
    assert result["reduced_chi_square"] is not None
    assert len(result["fitted_signal"]) == 401
    assert len(result["residual"]) == 401
    assert any(item.label == "Export CXRS line-profile result"
               for item in app.get("download_button"))


def test_cxrs_dashboard_withholds_unresolved_temperature_and_surfaces_error():
    app, _ = _cxrs_app()
    app.number_input(key="cxrs_inst_sigma").set_value(0.2)
    app.button(key="cxrs_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("unresolved" in str(item.value).lower() for item in app.error)
    assert "tis_cxrs_result" not in app.session_state


def test_cxrs_dashboard_rejects_nonpositive_sample_uncertainty():
    app, _ = _cxrs_app()
    frame = app.session_state["tis_df"].copy()
    frame.loc[20, "signal_sigma"] = 0.0
    app.session_state["tis_df"] = frame
    app.run()
    app.button(key="cxrs_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("finite positive values" in str(item.value) for item in app.error)
    assert "tis_cxrs_result" not in app.session_state


def test_cxrs_dashboard_failed_rerun_clears_previous_successful_result():
    app, _ = _cxrs_app()
    app.button(key="cxrs_run").click().run()
    assert "tis_cxrs_result" in app.session_state

    # Keep the same selected columns and settings but make the next fit invalid.
    frame = app.session_state["tis_df"].copy()
    frame["cxrs_signal"] = 0.0
    app.session_state["tis_df"] = frame
    app.button(key="cxrs_run").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("positive emission feature" in str(item.value) for item in app.error)
    assert "tis_cxrs_result" not in app.session_state


def test_cxrs_dashboard_hides_stale_result_when_analysis_parameters_change():
    app, _ = _cxrs_app()
    app.button(key="cxrs_run").click().run()
    assert "tis_cxrs_result" in app.session_state
    assert any("CXRS line-profile analysis completed" in str(item.value) for item in app.success)

    # Changing a physics-defining input must invalidate the displayed result
    # before the user runs the fit again.
    app.number_input(key="cxrs_inst_sigma").set_value(0.009).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert not any("CXRS line-profile analysis completed" in str(item.value) for item in app.success)


def test_cxrs_dashboard_propagates_instrument_width_uncertainty():
    app, _ = _cxrs_app()
    app.number_input(key="cxrs_inst_sigma_unc").set_value(0.001).run()
    app.button(key="cxrs_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    result = app.session_state["tis_cxrs_result"]["result"]
    assert result["instrument_sigma_uncertainty_nm"] == pytest.approx(0.001)
    assert result["ion_temperature_uncertainty_eV"] > 0


def test_cxrs_dashboard_withholds_under_resolved_spectrum():
    app, _ = _cxrs_app()
    wavelength = np.linspace(529.8, 530.2, 15)
    center = 530.0 * (1.0 + 2.5e4 / 299792458.0)
    sigma_observed = np.hypot(0.025, 0.008)
    frame = pd.DataFrame({
        "wavelength_nm": wavelength,
        "cxrs_signal": 120.0 * np.exp(-0.5 * ((wavelength-center) / sigma_observed) ** 2) + 4.0,
        "signal_sigma": np.full(wavelength.size, 0.5),
    })
    app.session_state["tis_df"] = frame
    app.session_state["tis_raw"] = frame.to_csv(index=False).encode("utf-8")
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.button(key="cxrs_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("under-resolved by wavelength sampling" in str(item.value).lower()
               for item in app.error)
    assert "tis_cxrs_result" not in app.session_state


def test_cxrs_dashboard_propagates_rest_wavelength_uncertainty():
    app, _ = _cxrs_app()
    app.number_input(key="cxrs_rest_unc").set_value(1e-5).run()
    app.button(key="cxrs_run").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    result = app.session_state["tis_cxrs_result"]["result"]
    assert result["rest_wavelength_uncertainty_nm"] == pytest.approx(1e-5)
    assert result["ion_temperature_uncertainty_eV"] > 0
    assert result["velocity_uncertainty_m_s"] > 0

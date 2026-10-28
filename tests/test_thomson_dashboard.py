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


def test_thomson_calibrated_power_density_dashboard_executes_and_exports():
    frame = pd.DataFrame({"unused_signal": [0.0, 1.0, 2.0]})
    app = _app_with_table(frame)
    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Thomson calibrated scattered-power density"
    ).run()

    next(item for item in app.checkbox if item.label == "Propagate independent input uncertainties").check().run()
    next(item for item in app.button if item.label == "Estimate Thomson electron density").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any(
        "Calibrated scattered-power density estimate completed." in str(item.value)
        for item in app.success
    )
    assert any(item.label == "Electron density" for item in app.metric)
    assert any(item.label == "Propagated 1-sigma uncertainty" for item in app.metric)
    assert any(
        item.label == "Export Thomson power-density estimate"
        for item in app.get("download_button")
    )


def test_thomson_calibrated_spectrum_dashboard_estimates_temperature_and_density():
    wavelength = np.linspace(530.8, 533.2, 241)
    physical_sigma_nm = 532.0 * np.sqrt(4.0 * 0.01 / 510_998.95069) * np.sin(np.pi / 4.0)
    fitted_sigma_nm = np.sqrt(physical_sigma_nm**2 + 0.05**2)
    signal = (
        1.0e-12 * np.exp(-0.5 * ((wavelength - 532.0) / fitted_sigma_nm) ** 2)
        + 1.2e-14 + 8.0e-16 * (wavelength - 532.0)
    )
    app = _app_with_table(pd.DataFrame({"wavelength_nm": wavelength, "power_density_W_nm": signal}))
    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Thomson raw-spectrum temperature fit"
    ).run()

    selectors = {item.label: item for item in app.selectbox}
    selectors["Wavelength (nm)"].set_value("wavelength_nm")
    selectors["Measured spectral signal"].set_value("power_density_W_nm")
    next(
        item for item in app.checkbox
        if item.label == "Signal is calibrated spectral power density (W/nm); also estimate electron density"
    ).check().run()
    next(
        item for item in app.button
        if item.label == "Fit Thomson spectrum and estimate electron temperature"
    ).click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Thomson spectral fit completed." in str(item.value) for item in app.success)
    assert any(item.label == "Electron temperature" for item in app.metric)
    assert any(item.label == "Integrated collected scattered power" for item in app.metric)
    assert any(item.label == "Electron density" for item in app.metric)
    assert any(item.label == "Export Thomson fit summary" for item in app.get("download_button"))
    temperature_metric = next(item for item in app.metric if item.label == "Electron temperature")
    density_metric = next(item for item in app.metric if item.label == "Electron density")
    power_metric = next(item for item in app.metric if item.label == "Integrated collected scattered power")
    assert float(temperature_metric.value.split()[0]) == pytest.approx(0.01, rel=0.05)
    assert float(density_metric.value.split()[0]) > 0.0
    assert float(power_metric.value.split()[0]) > 0.0



def test_thomson_spatial_profile_reconstruction_dashboard_executes_and_exports():
    radius = np.linspace(0.0, 0.4, 9)
    frame = pd.DataFrame({
        "radius_m": radius,
        "density_m3": 2.0e19 * (1.0 - 0.5 * radius),
        "temperature_eV": 120.0 * (1.0 - 0.25 * radius),
    })
    app = _app_with_table(frame)
    next(item for item in app.selectbox if item.label == "Select workflow").set_value(
        "Thomson spatial profile reconstruction"
    ).run()

    selectors = {item.label: item for item in app.selectbox}
    selectors["Measurement radius (m)"].set_value("radius_m")
    selectors["Calibrated electron density (m⁻³)"].set_value("density_m3")
    selectors["Calibrated electron temperature (eV)"].set_value("temperature_eV")
    selectors["Reconstruction method"].set_value("Shape-preserving PCHIP")
    app.run()
    next(item for item in app.button if item.label == "Reconstruct Thomson profiles").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any(
        "Spatial Thomson profile reconstruction completed" in str(item.value)
        for item in app.success
    )
    assert any(item.label == "Download reconstructed Thomson profiles (CSV)" for item in app.get("download_button"))
    assert any(item.label == "Export reconstruction, method and limitations (JSON)" for item in app.get("download_button"))



def test_thomson_spatial_reconstruction_never_returns_negative_profiles():
    from tokamak_analysis.thomson_scattering import reconstruct_thomson_profile

    radius = np.linspace(0.0, 0.4, 9)
    # Positive but strongly varying samples exercise spline overshoot protection.
    density = np.array([1.0, 0.02, 0.9, 0.03, 0.8, 0.02, 0.7, 0.03, 0.6]) * 1e19
    temperature = np.array([100.0, 1.0, 90.0, 2.0, 80.0, 1.0, 70.0, 2.0, 60.0])
    result = reconstruct_thomson_profile(
        radius, density, temperature, method="smoothing_spline",
        smoothing_factor=100.0, points=401,
    )

    assert np.all(np.isfinite(result["density_m3"]))
    assert np.all(np.isfinite(result["temperature_eV"]))
    assert np.all(result["density_m3"] >= 0.0)
    assert np.all(result["temperature_eV"] >= 0.0)
    assert np.all(result["density_gradient_m-4"][result["density_m3"] == 0.0] >= 0.0)
    assert np.all(result["temperature_gradient_eV_m-1"][result["temperature_eV"] == 0.0] >= 0.0)


def test_thomson_smoothing_spline_reports_zero_uncertainty_contract():
    from tokamak_analysis.thomson_scattering import reconstruct_thomson_profile

    radius = np.linspace(0.0, 0.4, 5)
    density = np.linspace(1e19, 2e19, 5)
    temperature = np.linspace(50.0, 100.0, 5)
    zeros = np.zeros(5)

    with pytest.raises(ValueError, match="uncertainties must be strictly positive"):
        reconstruct_thomson_profile(
            radius, density, temperature,
            density_uncertainty_m3=zeros,
            method="smoothing_spline",
        )

    # Zero uncertainties remain acceptable for the shape-preserving method,
    # which does not use uncertainty values as inverse-variance fit weights.
    result = reconstruct_thomson_profile(
        radius, density, temperature,
        density_uncertainty_m3=zeros,
        method="pchip",
    )
    assert np.all(result["density_m3"] >= 0.0)

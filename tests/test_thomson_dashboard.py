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


def test_thomson_geometry_dependent_workflows_are_not_in_analysis_1_ui():
    frame = pd.DataFrame({
        "radius_m": [0.0, 0.1, 0.2],
        "density_m3": [2.0e19, 1.8e19, 1.6e19],
        "temperature_eV": [100.0, 90.0, 80.0],
    })
    app = _app_with_table(frame)
    workflow = next(item for item in app.selectbox if item.label == "Select workflow")
    assert "Thomson downstream profile summary (pre-derived ne/Te)" not in workflow.options
    assert "Thomson downstream profile reconstruction (pre-derived ne/Te)" not in workflow.options
    assert "Thomson spatial profile reconstruction from calibrated spectra" in workflow.options



def test_primary_multi_channel_spectral_inference_is_exposed_in_spectral_workspace():
    frame = pd.DataFrame({
        "channel_id": ["ch0", "ch0", "ch1", "ch1"],
        "radius_m": [0.1, 0.1, 0.2, 0.2],
        "wavelength_nm": [530.0, 531.0, 530.0, 531.0],
        "intensity": [1.0, 2.0, 1.5, 2.5],
    })
    app = _app_with_table(frame)
    app.radio[0].set_value("Spectral Analysis").run()
    workflow = next(item for item in app.radio if item.label == "Workflow")
    assert "Thomson multi-channel spectral inference" in workflow.options


def test_thomson_profile_subanalysis_routes_to_canonical_spectral_inference():
    wavelength = np.linspace(531.0, 533.0, 9)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 0.2) ** 2)
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * wavelength.size + ["ch1"] * wavelength.size,
        "radius_m": [0.1] * wavelength.size + [0.2] * wavelength.size,
        "wavelength_nm": np.tile(wavelength, 2),
        "intensity": np.tile(signal, 2),
    })
    app = _app_with_table(frame)
    workflow = next(item for item in app.selectbox if item.label == "Select workflow")
    workflow.set_value("Thomson spatial profile reconstruction from calibrated spectra").run()
    next(item for item in app.button if item.label == "Open Thomson spectral-to-spatial-profile workflow").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert app.radio[0].value == "Spectral Analysis"
    spectral_workflow = next(item for item in app.radio if item.label == "Workflow")
    assert spectral_workflow.value == "Thomson multi-channel spectral inference"


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

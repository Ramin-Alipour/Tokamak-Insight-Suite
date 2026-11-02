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
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Advanced Analysis").run()
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
    assert "Analysis 1 / Spatial-profile reconstruction" in workflow.options



def test_primary_multi_channel_spectral_inference_is_exposed_in_spectral_workspace():
    frame = pd.DataFrame({
        "channel_id": ["ch0", "ch0", "ch1", "ch1"],
        "radius_m": [0.1, 0.1, 0.2, 0.2],
        "wavelength_nm": [530.0, 531.0, 530.0, 531.0],
        "intensity": [1.0, 2.0, 1.5, 2.5],
    })
    app = _app_with_table(frame)
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Spectral Analysis").run()
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
    workflow.set_value("Analysis 1 / Spatial-profile reconstruction").run()
    next(item for item in app.button if item.label == "Open Thomson spectral-to-spatial-profile workflow").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert next(item for item in app.radio if item.label == "WORKSPACE").value == "Spectral Analysis"
    spectral_workflow = next(item for item in app.radio if item.label == "Workflow")
    assert spectral_workflow.value == "Thomson multi-channel spectral inference"



def test_analysis_1_dashboard_exposes_only_three_selected_optical_models():
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * 9,
        "radius_m": [0.1] * 9,
        "wavelength_nm": np.linspace(531.0, 533.0, 9),
        "intensity": np.linspace(1.0, 2.0, 9),
    })
    app = _app_with_table(frame)
    app.radio[0].set_value("Spectral Analysis").run()
    workflow = next(item for item in app.radio if item.label == "Workflow")
    workflow.set_value("Thomson multi-channel spectral inference").run()
    model = next(item for item in app.selectbox if item.label == "Physical spectral model")
    assert model.options == [
        "non_collective_maxwellian",
        "non_collective_kappa",
        "non_collective_relativistic_maxwell_juttner",
    ]


def test_dashboard_exposes_joint_spectral_covariance_for_relativistic_mj():
    wavelength = np.linspace(526.0, 538.0, 9)
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * wavelength.size + ["ch1"] * wavelength.size,
        "radius_m": [0.1] * wavelength.size + [0.2] * wavelength.size,
        "wavelength_nm": np.tile(wavelength, 2),
        "intensity": np.tile(np.linspace(1.0, 2.0, wavelength.size), 2),
    })
    app = _app_with_table(frame)
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "Thomson multi-channel spectral inference"
    ).run()
    next(item for item in app.selectbox if item.label == "Physical spectral model").set_value(
        "non_collective_relativistic_maxwell_juttner"
    ).run()
    assert any(
        item.label.startswith("Optional full cross-channel spectral covariance CSV")
        for item in app.file_uploader
    )
    assert any(
        "joint fit with shared electron temperature" in item.value
        for item in app.caption
    )
    assert not app.exception, [str(error.value) for error in app.exception]


def test_independent_thomson_spectral_distribution_workflow_is_exposed_and_renders():
    wavelength = np.linspace(531.0, 533.0, 41)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 0.18) ** 2) + 0.02
    background = np.full_like(wavelength, 0.02)
    frame = pd.DataFrame({
        "wavelength_nm": wavelength,
        "measured_spectrum": signal,
        "measured_background": background,
    })
    app = _app_with_table(frame)
    workflow = next(item for item in app.selectbox if item.label == "Select workflow")
    assert "Analysis 1 / Spectral-distribution subanalysis" in workflow.options
    workflow.set_value("Analysis 1 / Spectral-distribution subanalysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any(
        item.label == "Open canonical Thomson spectral-to-physics workflow"
        for item in app.button
    )


def test_dashboard_exposes_fixed_operator_joint_spatial_thomson_mode():
    wavelength = np.linspace(531.0, 533.0, 9)
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * wavelength.size + ["ch1"] * wavelength.size,
        "radius_m": [0.1] * wavelength.size + [0.2] * wavelength.size,
        "wavelength_nm": np.tile(wavelength, 2),
        "intensity": np.tile(np.exp(-0.5 * ((wavelength - 532.0) / 0.2) ** 2), 2),
    })
    app = _app_with_table(frame)
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value("Thomson multi-channel spectral inference").run()
    next(item for item in app.selectbox if item.label == "Physical spectral model").set_value("non_collective_maxwellian").run()
    joint = next(item for item in app.checkbox if item.label.startswith("Use joint wavelength-resolved spatial inversion"))
    assert joint.disabled is False
    joint.set_value(True).run()
    assert any(item.label.startswith("Fixed measured/researcher-supplied spatial mixing matrix") for item in app.file_uploader)
    assert any(item.label.startswith("Spatial curvature regularization strength") for item in app.text_input)
    background_model = next(item for item in app.selectbox if item.label == "Joint detector-background basis")
    assert background_model.options == ["constant", "linear", "polynomial", "cubic_spline", "templates"]
    background_model.set_value("polynomial").run()
    assert any(item.label.startswith("Background polynomial degree") for item in app.number_input)
    assert not app.exception, [str(error.value) for error in app.exception]


def test_dashboard_source_renders_projected_evdf_profiles_and_uncertainty():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Projected electron velocity distribution (EVDF) from calibrated Thomson spectra" in source
    assert "velocity_parallel_m_s" in source
    assert "evdf_parallel_s_per_m" in source
    assert "evdf_standard_uncertainty_s_per_m" in source
    assert "regularization_sensitivity" in source


def test_dashboard_exposes_conditional_path_integrated_density_inputs():
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * 9,
        "radius_m": [0.1] * 9,
        "wavelength_nm": np.linspace(531.0, 533.0, 9),
        "intensity": np.linspace(1.0, 2.0, 9),
    })
    app = _app_with_table(frame)
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "Thomson multi-channel spectral inference"
    ).run()
    assert any(
        item.label == "Path-dependent optical/geometry profiles CSV"
        for item in app.file_uploader
    )
    assert any(
        item.label == "Absolutely calibrated integrated scattered power (W)"
        for item in app.text_input
    )
    assert not app.exception, [str(error.value) for error in app.exception]

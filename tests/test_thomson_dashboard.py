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



def test_registered_spectral_distribution_dashboard_renders_fit_and_both_evdf_contracts():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Measured versus fitted Thomson spectrum" in source
    assert "Measured-space fit residual" in source
    assert "{'velocity_grid_m_s', 'projected_evdf_s_m'}" in source
    assert "selected physical model did not pass the primary-result identifiability gate" in source


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
    psf_mode = next(item for item in app.selectbox if item.label == "Spatial PSF forward operator")
    assert psf_mode.options == ["identity", "gaussian_sigma", "measured_matrix"]
    psf_mode.set_value("gaussian_sigma").run()
    assert any(item.label.startswith("Gaussian spatial PSF sigma") for item in app.text_input)
    psf_mode.set_value("measured_matrix").run()
    assert any(item.label.startswith("Measured spatial PSF matrix") for item in app.file_uploader)
    assert any(item.label.startswith("Spatial curvature regularization strength") for item in app.text_input)
    background_model = next(item for item in app.selectbox if item.label == "Joint detector-background basis")
    assert background_model.options == ["constant", "linear", "polynomial", "cubic_spline", "templates"]
    background_model.set_value("polynomial").run()
    assert any(item.label.startswith("Background polynomial degree") for item in app.number_input)
    assert not app.exception, [str(error.value) for error in app.exception]


def test_dashboard_exposes_joint_projected_pressure_heat_flux_covariance():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Joint covariance of projected pressure and parallel kinetic-energy flux" in source
    assert "parallel_pressure_heat_flux_joint_covariance" in source
    assert "density_evdf_moment_joint_covariance" in source
    assert "not the full 3-D heat-flux covariance" in source
    assert "Cov(n_e, μ₃,q) density–third-moment cross-covariance" in source
    assert "electron_density_mu3_cross_covariance_s_minus3" in source


def test_dashboard_exposes_response_aware_evdf_spectral_closure():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "evdf_predicted_response_aware_centroid_nm" in source
    assert "evdf_predicted_response_aware_variance_nm2" in source
    assert "evdf_finite_window_captured_probability" in source
    assert "measured laser/detector response" in source


def test_dashboard_source_renders_projected_evdf_profiles_and_uncertainty():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Projected electron velocity distribution (EVDF) from calibrated Thomson spectra" in source
    assert "velocity_parallel_m_s" in source
    assert "evdf_parallel_s_per_m" in source
    assert "evdf_standard_uncertainty_s_per_m" in source
    assert "projected_evdf_s_m" in source
    assert "regularized_candidate_evdf_s_m" in source
    assert "projected_evdf_standard_uncertainty_s_m" in source
    assert "conditional_regularized_candidate" in source
    assert "regularization_sensitivity" in source
    assert "Frequentist fixed-active-set Tikhonov sandwich" in source
    assert "parametric_model_selection_uncertainty" in source
    assert "excludes regularization-selection uncertainty" in source


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
        item.label.startswith("Optional full optical-profile covariance CSV")
        for item in app.file_uploader
    )
    assert any(
        item.label == "Absolutely calibrated integrated scattered power (W)"
        for item in app.text_input
    )
    assert not app.exception, [str(error.value) for error in app.exception]



def test_relativistic_dashboard_exposes_compton_recoil_switch():
    wavelength = np.linspace(526.0, 538.0, 9)
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * wavelength.size,
        "radius_m": [0.1] * wavelength.size,
        "wavelength_nm": wavelength,
        "intensity": np.linspace(1.0, 2.0, wavelength.size),
    })
    app = _app_with_table(frame)
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "Thomson multi-channel spectral inference"
    ).run()
    next(item for item in app.selectbox if item.label == "Physical spectral model").set_value(
        "non_collective_relativistic_maxwell_juttner"
    ).run()
    recoil = next(
        item for item in app.checkbox
        if item.label.startswith("Include single-scattering Compton recoil")
    )
    assert recoil.value is False
    recoil.set_value(True).run()
    assert recoil.value is True
    assert not app.exception, [str(error.value) for error in app.exception]


def test_advanced_dashboard_exposes_joint_kappa_mj_background_subanalysis():
    frame = pd.DataFrame({
        "radius_m": [0.1, 0.1],
        "wavelength_nm": [530.0, 531.0],
        "intensity": [1.0, 1.5],
    })
    app = _app_with_table(frame)
    workflow = next(item for item in app.selectbox if item.label == "Select workflow")
    assert "Analysis 1 / Kappa-MJ structured-background subanalysis" in workflow.options
    workflow.set_value("Analysis 1 / Kappa-MJ structured-background subanalysis").run()
    assert any(item.label.startswith("Measured calibrated optical Thomson spectrum CSV") for item in app.file_uploader)
    assert any(item.label == "Physical electron model" for item in app.selectbox)
    assert not app.exception, [str(error.value) for error in app.exception]


def test_two_population_dashboard_exposes_compton_recoil_switch():
    frame = pd.DataFrame({
        "radius_m": [0.1, 0.1],
        "wavelength_nm": [530.0, 531.0],
        "intensity": [1.0, 1.5],
    })
    app = _app_with_table(frame)
    workflow = next(item for item in app.selectbox if item.label == "Select workflow")
    workflow.set_value("Analysis 1 / Two-population optical Thomson subanalysis").run()
    assert any(item.label.startswith("Include single-scattering Compton recoil") for item in app.checkbox)
    assert not app.exception, [str(error.value) for error in app.exception]


def test_n_population_thomson_dashboard_exposes_model_inputs():
    frame = pd.DataFrame({
        "radius_m": [0.1, 0.1],
        "wavelength_nm": [530.0, 531.0],
        "intensity": [1.0, 1.5],
    })
    app = _app_with_table(frame)
    workflow = next(item for item in app.selectbox if item.label == "Select workflow")
    assert "Analysis 1 / N-population optical Thomson subanalysis" in workflow.options
    workflow.set_value("Analysis 1 / N-population optical Thomson subanalysis").run()
    assert any(item.label == "Number of electron populations" for item in app.number_input)
    assert any(item.label.startswith("Initial population temperatures") for item in app.text_input)
    assert any(item.label.startswith("Initial population number fractions") for item in app.text_input)
    assert not app.exception, [str(error.value) for error in app.exception]


def test_thomson_dashboard_displays_model_domain_diagnostics():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Optical Thomson physical-model validity by channel" in source
    assert "domain_status" in source
    assert "theta_e_status" in source
    assert "beta_status" in source



def test_thomson_dashboard_exposes_joint_spatial_mixing_covariance_refits():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "ts_infer_spatial_mixing_covariance" in source
    assert "spatial_mixing_covariance" in source
    assert "ts_infer_spatial_mixing_mc_samples" in source


def test_dashboard_exposes_optical_collective_kappa_population_inference_controls():
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * 9,
        "radius_m": [0.1] * 9,
        "wavelength_nm": np.linspace(1048.0, 1080.0, 9),
        "intensity": np.linspace(1.0, 2.0, 9),
    })
    app = _app_with_table(frame)
    next(item for item in app.radio if item.label == "WORKSPACE").set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "Thomson multi-channel spectral inference"
    ).run()
    toggle = next(
        item for item in app.checkbox
        if item.label.startswith("Use optical collective-Kappa dielectric inference")
    )
    toggle.set_value(True).run()
    assert any(
        item.label == "Collective-Kappa physical inputs (JSON)"
        for item in app.text_area
    )
    assert not app.exception, [str(error.value) for error in app.exception]


def test_registered_spectral_distribution_dashboard_exposes_projected_heat_flux_density():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
    assert "Independent electron density for projected heat-flux calculation" in source
    assert "Independent electron-density uncertainty" in source
    assert "'electron_density_m3'" in source
    assert "'electron_density_uncertainty_m3'" in source
    assert "not the full 3-D heat flux" in source
    assert "Projected parallel heat flux (W/m^2)" in source


def test_product_workspace_routes_spectral_distribution_outputs_by_analysis_id():
    source = (Path(__file__).resolve().parents[1] / "tis" / "ui" / "product_workspace_page.py").read_text(encoding="utf-8")
    assert "if selected_id == 1 and selected_analysis_id == 81:" in source
    assert "if selected_id == 81:" not in source
    assert "Projected EVDF physical inference" in source



def test_registered_spectral_distribution_dashboard_exposes_projected_pressure_inputs_and_outputs():
    source = APP_PATH.read_text(encoding="utf-8")
    assert "Cov(n_e, σ²_q) density–EVDF cross-covariance" in source
    assert "electron_density_variance_cross_covariance_m_minus1_s_minus2" in source
    assert "Projected q-direction electron pressure (Pa)" in source
    assert "Projected q-direction pressure 1σ uncertainty (Pa)" in source

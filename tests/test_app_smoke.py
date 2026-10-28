from pathlib import Path

import pytest
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



@pytest.mark.parametrize(
    "workflow",
    [
        "Thomson calibrated profile analysis",
        "Thomson spatial profile reconstruction",
        "Thomson raw-spectrum temperature fit",
        "Thomson calibrated scattered-power density",
    ],
)
def test_thomson_workflow_panels_render_without_streamlit_exceptions(workflow):
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Advanced Analysis").run()
    selector = next(item for item in app.selectbox if item.label == "Select workflow")
    selector.set_value(workflow).run()
    assert not app.exception, (workflow, [str(error.value) for error in app.exception])



def test_thomson_spectral_power_mode_reveals_density_geometry_controls():
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Advanced Analysis").run()
    selector = next(item for item in app.selectbox if item.label == "Select workflow")
    selector.set_value("Thomson raw-spectrum temperature fit").run()
    calibrated_checkbox = next(
        item for item in app.checkbox
        if "calibrated spectral power density" in item.label.lower()
    )
    calibrated_checkbox.set_value(True).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Incident laser power" in item.label for item in app.number_input)


def test_langmuir_analysis_button_executes_and_renders_physics_result():
    """Exercise the real dashboard action, not only the page's initial render."""
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Probe Analysis").run()

    analyze_button = next(item for item in app.button if item.label == "Analyze probe I–V")
    analyze_button.click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    rendered_json = [str(item.value) for item in app.json]
    assert any("floating_potential_V" in value for value in rendered_json)
    assert any("ion_saturation_current_A" in value for value in rendered_json)
    assert any("interpretation_warning" in value for value in rendered_json)



def test_ece_dashboard_resonance_mapping_executes_and_exports_result():
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Plasma Parameters").run()
    mode = next(item for item in app.selectbox if item.label == "ECE workflow")
    mode.set_value("Resonance mapping").run()
    button = next(item for item in app.button if item.label == "Map ECE resonance radii")
    button.click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("resonance_field_T" in str(item.value) for item in app.json)
    assert any("Export ECE resonance mapping" == str(item.label) for item in app.download_button)


def test_ece_dashboard_brightness_temperature_executes_with_calibrated_radiance():
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Plasma Parameters").run()
    mode = next(item for item in app.selectbox if item.label == "ECE workflow")
    mode.set_value("Brightness temperature").run()
    radiance = next(item for item in app.text_input if item.label.startswith("Calibrated spectral radiance"))
    radiance.set_value("1e-10, 1.2e-10, 1.1e-10").run()
    next(item for item in app.button if item.label == "Analyze ECE brightness temperature").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("brightness_temperature_eV" in str(item.value) for item in app.json)



def test_ece_dashboard_channel_fluctuations_executes_and_exports_psd():
    """Exercise the ECE multi-channel workflow end-to-end in Streamlit AppTest."""
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Plasma Parameters").run()
    mode = next(item for item in app.selectbox if item.label == "ECE workflow")
    mode.set_value("Channel fluctuations / PSD").run()
    # The fixture supplies two signal channels; keep the ECE frequency count aligned.
    next(item for item in app.text_input if item.label.startswith("Channel center frequencies")).set_value("80, 90").run()
    time_column = next(item for item in app.selectbox if item.label == "ECE time column (s)")
    time_column.set_value("time_s").run()
    channels = next(item for item in app.multiselect if item.label.startswith("ECE channel signal columns"))
    channels.set_value(["mirnov0", "mirnov1"]).run()
    next(item for item in app.button if item.label == "Analyze ECE fluctuations / PSD").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any(item.label == "Export ECE fluctuation / PSD result" for item in app.download_button)



def test_molecular_spectroscopy_dashboard_executes_feature_and_gaussian_fit_workflows():
    import numpy as np
    import pandas as pd

    wavelength = np.linspace(400.0, 410.0, 201)
    intensity = 1.2 + 5.0 * np.exp(-0.5 * ((wavelength - 405.0) / 0.3) ** 2)
    frame = pd.DataFrame({"wavelength_nm": wavelength, "intensity": intensity})
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_molecular_spectrum.csv"
    app.session_state["tis_raw"] = b"synthetic molecular spectrum"
    app.session_state["tis_sha"] = "synthetic-molecular-spectrum"
    app.run()

    app.radio[0].set_value("Spectral Analysis").run()
    workflow = next(item for item in app.radio if item.label == "Workflow")
    workflow.set_value("Molecular spectroscopy").run()
    next(item for item in app.selectbox if item.label == "Calibrated wavelength (nm)").set_value("wavelength_nm").run()
    next(item for item in app.selectbox if item.label == "Molecular spectral intensity").set_value("intensity").run()
    next(item for item in app.button if item.label == "Analyze molecular spectrum").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("peak_count" in str(item.value) for item in app.json)

    next(item for item in app.text_input if item.label == "Initial band-center guesses (nm, comma-separated)").set_value("405.0").run()
    next(item for item in app.button if item.label == "Fit empirical Gaussian bands").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("parameters" in str(item.value) for item in app.json)
    assert any(item.label == "Export empirical molecular band fit" for item in app.get("download_button"))



def test_molecular_boltzmann_dashboard_executes_temperature_fit():
    import numpy as np
    import pandas as pd

    energy = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    wavelength = np.array([400.0, 405.0, 410.0, 415.0, 420.0, 425.0])
    einstein_a = np.array([1.0e7, 1.2e7, 1.4e7, 1.6e7, 1.8e7, 2.0e7])
    degeneracy = np.array([1, 3, 5, 3, 5, 7], dtype=float)
    intensity = 1.0e-12 * einstein_a * degeneracy / wavelength * np.exp(-energy / 0.8)
    frame = pd.DataFrame({
        "upper_energy_eV": energy, "integrated_intensity": intensity,
        "einstein_A_s": einstein_a, "degeneracy": degeneracy,
        "wavelength_nm": wavelength,
    })
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_molecular_lines.csv"
    app.session_state["tis_raw"] = b"synthetic molecular lines"
    app.session_state["tis_sha"] = "synthetic-molecular-lines"
    app.run()

    app.radio[0].set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value("Molecular Boltzmann temperature").run()
    selections = {
        "Upper-state energy (eV)": "upper_energy_eV",
        "Integrated line intensity (energy units)": "integrated_intensity",
        "Einstein A coefficient (s⁻¹)": "einstein_A_s",
        "Upper-state statistical weight": "degeneracy",
        "Transition wavelength (nm)": "wavelength_nm",
    }
    for label, value in selections.items():
        next(item for item in app.selectbox if item.label == label).set_value(value).run()
    next(item for item in app.button if item.label == "Fit molecular excitation temperature").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("excitation_temperature_eV" in str(item.value) for item in app.json)
    assert any("Export molecular Boltzmann temperature" == item.label for item in app.get("download_button"))


def test_thomson_spectral_distribution_dashboard_executes_and_exports():
    import numpy as np
    import pandas as pd

    wavelength = np.linspace(499.0, 501.0, 101)
    signal = 1e-20 * (
        1000.0 * np.exp(-0.5 * ((wavelength - 500.0) / 0.18) ** 2)
        + 12.0 + 0.8 * (wavelength - 500.0)
    )
    frame = pd.DataFrame({"wavelength_nm": wavelength, "signal": signal})
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_thomson_spectrum.csv"
    app.session_state["tis_raw"] = b"synthetic Thomson spectrum"
    app.session_state["tis_sha"] = "synthetic-thomson-spectrum"
    app.run()

    app.radio[0].set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "Thomson spectral distribution"
    ).run()
    next(item for item in app.selectbox if item.label == "Calibrated Thomson wavelength (nm)").set_value(
        "wavelength_nm"
    ).run()
    next(item for item in app.selectbox if item.label == "Thomson spectral signal").set_value(
        "signal"
    ).run()
    next(item for item in app.button if item.label == "Analyze Thomson spectral distribution").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("spectral_centroid_nm" in str(item.value) for item in app.json)
    assert any(
        item.label == "Export Thomson spectral-distribution results"
        for item in app.get("download_button")
    )



def test_cxrs_dashboard_executes_line_fit_and_exports_result():
    """Exercise the CXRS workflow through the real Streamlit dashboard."""
    import numpy as np
    import pandas as pd

    wavelength = np.linspace(529.8, 530.2, 401)
    rest_wavelength = 530.0
    expected_velocity = 2.5e4
    center = rest_wavelength * (1.0 + expected_velocity / 299792458.0)
    intrinsic_sigma = 0.025
    instrument_sigma = 0.008
    observed_sigma = np.hypot(intrinsic_sigma, instrument_sigma)
    intensity = (
        120.0 * np.exp(-0.5 * ((wavelength - center) / observed_sigma) ** 2)
        + 4.0 + 0.2 * (wavelength - center)
    )
    frame = pd.DataFrame({"wavelength_nm": wavelength, "intensity": intensity})
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_cxrs_spectrum.csv"
    app.session_state["tis_raw"] = b"synthetic CXRS spectrum"
    app.session_state["tis_sha"] = "synthetic-cxrs-spectrum"
    app.run()

    app.radio[0].set_value("Spectral Analysis").run()
    next(item for item in app.radio if item.label == "Workflow").set_value(
        "CXRS line profile"
    ).run()
    next(item for item in app.selectbox if item.label == "Calibrated wavelength (nm)").set_value(
        "wavelength_nm"
    ).run()
    next(item for item in app.selectbox if item.label == "Spectral intensity").set_value(
        "intensity"
    ).run()
    next(item for item in app.button if item.label == "Analyze CXRS line").click().run()

    assert not app.exception, [str(error.value) for error in app.exception]
    assert not app.error, [str(item.value) for item in app.error]
    assert any("ion_temperature_eV" in str(item.value) for item in app.json)
    assert any("line_of_sight_velocity_m_s" in str(item.value) for item in app.json)
    assert any(
        item.label == "Export CXRS line-profile result"
        for item in app.get("download_button")
    )

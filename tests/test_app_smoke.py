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
    app = _loaded_app()
    app.run()
    app.radio[0].set_value("Plasma Parameters").run()
    mode = next(item for item in app.selectbox if item.label == "ECE workflow")
    mode.set_value("Channel fluctuations / PSD").run()
    time_column = next(item for item in app.selectbox if item.label == "ECE time column (s)")
    time_column.set_value("time_s").run()
    channels = next(item for item in app.multiselect if item.label.startswith("ECE channel signal columns"))
    channels.set_value(["mirnov0", "mirnov1"]).run()
    next(item for item in app.button if item.label == "Analyze ECE fluctuations / PSD").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("fluctuation_psd" in str(item.value) for item in app.json)
    assert any(item.label == "Export ECE fluctuation / PSD result" for item in app.download_button)

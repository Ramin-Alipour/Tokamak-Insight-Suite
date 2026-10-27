"""Interactive Streamlit acceptance tests for Faraday-rotation polarimetry."""
from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest


def test_polarimetry_dashboard_executes_and_exports_results():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    assert not app.exception, "\n".join(str(item.value) for item in app.exception)
    app.radio(key="workspace_page").set_value("Advanced Analysis")
    app.run()
    app.selectbox(label="Select workflow").set_value("Faraday-rotation polarimetry")
    app.run()
    assert not app.exception, "\n".join(str(item.value) for item in app.exception)
    # Choose a wavelength/rotation pair with a finite, non-zero line integral.
    app.text_input(key="pol_wavelength").set_value("1.06e-5,1.06e-5")
    app.text_input(key="pol_rotation").set_value("0.001,-0.002")
    app.text_input(key="pol_density").set_value("1e19")
    app.text_input(key="pol_path").set_value("0.5")
    app.text_input(key="pol_rotation_unc").set_value("1e-5")
    app.run()
    next(button for button in app.button if button.label == "Analyze Faraday rotation").click()
    app.run()
    assert not app.exception, "\n".join(str(item.value) for item in app.exception)
    assert any("Faraday-rotation polarimetry analysis completed" in item.value for item in app.success)
    rendered = "\n".join(str(item.value) for item in app.json)
    assert "line_integrated_ne_B_parallel_m-2_T" in rendered
    assert "average_B_parallel_T" in rendered
    assert any(item.label == "Export polarimetry results CSV" for item in app.get("download_button"))
    assert any(item.label == "Export polarimetry result and limitations JSON" for item in app.get("download_button"))


def test_polarimetry_dashboard_rejects_mismatched_arrays():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py", default_timeout=120).run()
    app.radio(key="workspace_page").set_value("Advanced Analysis")
    app.run()
    app.selectbox(label="Select workflow").set_value("Faraday-rotation polarimetry")
    app.run()
    app.text_input(key="pol_wavelength").set_value("1e-6,2e-6")
    app.text_input(key="pol_rotation").set_value("0.1")
    app.run()
    next(button for button in app.button if button.label == "Analyze Faraday rotation").click()
    app.run()
    assert not app.exception, "\n".join(str(item.value) for item in app.exception)
    assert any("same number of values" in item.value for item in app.error)

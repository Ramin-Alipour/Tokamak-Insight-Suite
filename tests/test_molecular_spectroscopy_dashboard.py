"""Interactive acceptance tests for molecular spectroscopy dashboard workflows."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _app(frame):
    app = AppTest.from_file(APP_PATH, default_timeout=120)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_molecular.csv"
    app.session_state["tis_raw"] = frame.to_csv(index=False).encode("utf-8")
    app.session_state["tis_sha"] = "synthetic-molecular-regression"
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    app.radio(key="workspace_page").set_value("Spectral Analysis").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    next(item for item in app.radio if item.label == "Workflow").set_value("Molecular spectroscopy").run()
    assert not app.exception, [str(error.value) for error in app.exception]
    selectors = {item.label: item for item in app.selectbox}
    selectors["Calibrated wavelength (nm)"].set_value("wavelength_nm")
    selectors["Molecular spectral intensity"].set_value("intensity")
    app.run()
    assert not app.exception, [str(error.value) for error in app.exception]
    return app


def _spectrum():
    x = np.linspace(400.0, 410.0, 501)
    y = 2.0 + 10.0 * np.exp(-0.5 * ((x - 405.0) / 0.12) ** 2)
    return pd.DataFrame({"wavelength_nm": x, "intensity": y})


def test_molecular_spectrum_dashboard_runs_and_exports_features():
    app = _app(_spectrum())
    app.text_input(key="mol_refs").set_value("405.02:synthetic band")
    app.text_input(key="mol_ranges").set_value("404.5:405.5")
    next(button for button in app.button if button.label == "Analyze molecular spectrum").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert any("Molecular spectrum feature analysis completed" in str(item.value) for item in app.success)
    result = app.session_state["tis_molecular_spectrum_result"]["result"]
    assert result["peak_count"] == 1
    assert result["reference_matches"][0]["reference_label"] == "synthetic band"
    assert result["band_integrals"][0]["integrated_signal_intensity_nm"] > 0
    assert any(item.label == "Export molecular spectroscopy JSON" for item in app.get("download_button"))


def test_molecular_dashboard_hides_stale_result_when_analysis_settings_change():
    app = _app(_spectrum())
    app.button(label="Analyze molecular spectrum").click().run()
    assert "tis_molecular_spectrum_result" in app.session_state
    app.slider(key="mol_prominence").set_value(0.2).run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert not any("Molecular spectrum feature analysis completed" in str(item.value) for item in app.success)
    assert not any(item.label == "Export molecular spectroscopy JSON" for item in app.get("download_button"))


def test_molecular_dashboard_failed_rerun_clears_previous_result():
    app = _app(_spectrum())
    app.button(label="Analyze molecular spectrum").click().run()
    assert "tis_molecular_spectrum_result" in app.session_state
    frame = app.session_state["tis_df"].copy()
    frame["intensity"] = 1.0
    app.session_state["tis_df"] = frame
    app.run()
    app.button(label="Analyze molecular spectrum").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    assert "tis_molecular_spectrum_result" not in app.session_state
    assert any("no positive contrast" in str(item.value).lower() for item in app.error)


def test_molecular_boltzmann_dashboard_fits_supplied_transition_table_and_exports():
    energy = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    wavelength = np.array([400.0, 405.0, 410.0, 415.0, 420.0, 425.0])
    einstein_a = np.array([1.0e7, 1.2e7, 1.4e7, 1.6e7, 1.8e7, 2.0e7])
    degeneracy = np.array([1, 3, 5, 3, 5, 7], dtype=float)
    expected_temperature_eV = 0.8
    intensity = 1.0e-12 * einstein_a * degeneracy / wavelength * np.exp(-energy / expected_temperature_eV)
    frame = pd.DataFrame({
        "upper_energy_eV": energy, "integrated_intensity": intensity,
        "einstein_A_s": einstein_a, "degeneracy": degeneracy,
        "wavelength_nm": wavelength, "intensity": intensity,
        "intensity_sigma": 0.02 * intensity,
    })
    app = _app(frame)
    next(item for item in app.radio if item.label == "Workflow").set_value("Molecular Boltzmann temperature").run()
    selectors = {item.label: item for item in app.selectbox}
    selectors["Upper-state energy (eV)"].set_value("upper_energy_eV")
    selectors["Integrated line intensity (energy units)"].set_value("integrated_intensity")
    selectors["Einstein A coefficient (s⁻¹)"].set_value("einstein_A_s")
    selectors["Upper-state statistical weight"].set_value("degeneracy")
    selectors["Transition wavelength (nm)"].set_value("wavelength_nm")
    selectors["Integrated-intensity uncertainty σ (optional)"].set_value("intensity_sigma")
    app.run()
    app.button(key="mbt_fit_button").click().run()
    assert not app.exception, [str(error.value) for error in app.exception]
    result = app.session_state["tis_molecular_boltzmann_fit"]["result"]
    assert result["excitation_temperature_eV"] == pytest.approx(expected_temperature_eV, rel=1e-9)
    assert result["line_count"] == 6
    assert any(item.label == "Export molecular Boltzmann temperature" for item in app.get("download_button"))

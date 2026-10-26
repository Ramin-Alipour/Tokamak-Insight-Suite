"""Physics and live Streamlit acceptance tests for poloidal-field probes."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tokamak_analysis.poloidal_field_probes import analyze_poloidal_field_probes


def test_multichannel_calibration_derivative_and_exportable_quantities():
    t = np.linspace(0.0, 1.0, 101)
    raw = np.column_stack((2.0 * t + 3.0, -4.0 * t + 2.0))
    result = analyze_poloidal_field_probes(
        t, raw,
        calibration_gain_T_per_unit=[0.5, 0.25],
        calibration_offset_T=[0.1, -0.2],
        signal_uncertainty=np.full_like(raw, 0.02),
        probe_names=["inner", "outer"],
    )
    np.testing.assert_allclose(result["B_poloidal_T"][:, 0], t + 1.6)
    np.testing.assert_allclose(result["B_poloidal_T"][:, 1], -t + 0.3)
    np.testing.assert_allclose(result["dB_poloidal_dt_T_s"][1:-1, 0], 1.0, atol=1e-11)
    np.testing.assert_allclose(result["dB_poloidal_dt_T_s"][1:-1, 1], -1.0, atol=1e-11)
    assert result["B_poloidal_uncertainty_T"].shape == raw.shape
    assert result["dB_poloidal_dt_uncertainty_T_s"].shape == raw.shape
    assert result["sampling_rate_median_Hz"] == pytest.approx(100.0)
    assert result["probe_names"] == ["inner", "outer"]
    assert "not a spatial derivative" in result["physical_relations"]["time_derivative"]


@pytest.mark.parametrize("kwargs", [
    {"calibration_gain_T_per_unit": 0.0},
    {"calibration_gain_T_per_unit": [1.0, np.nan]},
    {"calibration_offset_T": [0.0, np.inf]},
    {"baseline_interval_s": (2.0, 1.0)},
    {"probe_names": ["only-one"]},
])
def test_invalid_calibration_baseline_or_channel_metadata_is_rejected(kwargs):
    t = np.arange(5.0)
    signal = np.column_stack((t, t))
    with pytest.raises(ValueError):
        analyze_poloidal_field_probes(t, signal, **kwargs)


def test_probe_dashboard_runs_multichannel_analysis_and_renders_export():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    t = np.linspace(0.0, 1.0, 101)
    app.session_state["tis_df"] = pd.DataFrame({
        "time_s": t, "probe_inner": 2.0 * t, "probe_outer": -3.0 * t,
    })
    app.session_state["tis_filename"] = "synthetic_poloidal_probes.csv"
    app.session_state["tis_sha"] = "synthetic-poloidal-test"
    app.radio(key="workspace_page").set_value("Probe Analysis")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    app.selectbox(key="pf_probe_time").set_value("time_s")
    app.run()
    app.multiselect(key="pf_probe_channels").set_value(["probe_inner", "probe_outer"])
    app.text_input(key="pf_gain").set_value("0.5,0.25")
    app.text_input(key="pf_offset").set_value("0.1,-0.2")
    app.text_input(key="pf_uncertainty").set_value("0.02")
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    next(button for button in app.button if button.label == "Analyze poloidal-field probes").click()
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    assert any("Poloidal-field probe analysis completed" in item.value for item in app.success)
    rendered = "\\n".join(str(item.value) for item in app.json)
    assert "B_poloidal_T" in rendered
    assert "dB_poloidal_dt_T_s" in rendered
    assert any(item.label == "Export poloidal-field analysis JSON" for item in app.get("download_button"))


def test_probe_dashboard_reports_missing_channel_selection_without_crashing():
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    app = AppTest.from_file(app_path, default_timeout=120).run()
    app.session_state["tis_df"] = pd.DataFrame({
        "time_s": np.arange(5.0), "probe": np.arange(5.0),
    })
    app.session_state["tis_filename"] = "missing_channel.csv"
    app.session_state["tis_sha"] = "synthetic-empty-selection"
    app.radio(key="workspace_page").set_value("Probe Analysis")
    app.run()
    next(button for button in app.button if button.label == "Analyze poloidal-field probes").click()
    app.run()
    assert not app.exception, "\\n".join(str(item.value) for item in app.exception)
    assert any("Select at least one magnetic-probe signal channel" in item.value for item in app.error)

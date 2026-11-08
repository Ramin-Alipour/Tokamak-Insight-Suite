import numpy as np
import pandas as pd
from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_ece_dashboard_runs_fluctuation_and_profile_interactions():
    root = Path(__file__).resolve().parents[1]
    sample_count = 128
    time_s = np.arange(sample_count, dtype=float) * 1e-6
    frame = pd.DataFrame({
        "time_s": time_s,
        "sig_a": np.sin(2 * np.pi * 10_000.0 * time_s),
        "sig_b": 0.5 * np.sin(2 * np.pi * 20_000.0 * time_s),
        "tb_a": np.full(sample_count, 25.0),
        "tb_b": np.full(sample_count, 45.0),
        "tb_c": np.full(sample_count, 65.0),
    })

    app = AppTest.from_file(str(root / "app.py"), default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_ece.csv"
    app.session_state["tis_active_source"] = "test_fixture"
    app.session_state["workspace_page"] = "Spectral Analysis"
    app.run()
    assert not app.exception

    workflow = next(widget for widget in app.radio if widget.label == "Workflow")
    workflow.set_value("ECE channel/profile analysis")
    app.run()
    assert not app.exception

    app.selectbox(key="ece_fluct_time_col").set_value("time_s")
    app.multiselect(key="ece_fluct_cols").set_value(["sig_a", "sig_b"])
    app.text_input(key="ece_fluct_frequencies").set_value("80e9,90e9")
    app.button(key="ece_fluct_run").click().run()
    assert not app.exception
    assert any("ECE fluctuation and spectral analysis completed." in item.value for item in app.success)

    app.selectbox(key="ece_time_col").set_value("time_s")
    app.multiselect(key="ece_channel_cols").set_value(["tb_a", "tb_b", "tb_c"])
    app.text_input(key="ece_frequency_text").set_value("100e9,80e9,90e9")
    app.button(key="ece_analyze").click().run()
    assert not app.exception
    assert any("ECE profile reconstructed within the declared approximate model." in item.value for item in app.success)
    assert any("reordered together into ascending frequency order" in item.value for item in app.info)

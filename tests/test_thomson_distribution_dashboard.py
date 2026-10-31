"""The Thomson spectral-distribution subanalysis must run physical inference."""

from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_thomson_spectral_distribution_routes_to_canonical_physical_inference():
    wavelength = np.linspace(531.0, 533.0, 9)
    signal = np.exp(-0.5 * ((wavelength - 532.0) / 0.2) ** 2)
    frame = pd.DataFrame({
        "channel_id": ["ch0"] * wavelength.size + ["ch1"] * wavelength.size,
        "radius_m": [0.1] * wavelength.size + [0.2] * wavelength.size,
        "wavelength_nm": np.tile(wavelength, 2),
        "intensity": np.tile(signal, 2),
    })
    app = AppTest.from_file(APP_PATH, default_timeout=60)
    app.session_state["tis_df"] = frame
    app.session_state["tis_filename"] = "synthetic_thomson_spectrum.csv"
    app.session_state["tis_raw"] = frame.to_csv(index=False).encode("utf-8")
    app.session_state["tis_sha"] = "synthetic-thomson-sha256"
    app.run()
    app.radio[0].set_value("Spectral Analysis").run()
    workflow = next(item for item in app.radio if item.label == "Workflow")
    workflow.set_value("Thomson spectral distribution").run()

    assert not app.exception, [str(error.value) for error in app.exception]
    labels = {item.label for item in app.selectbox}
    assert "Spatial channel identifier" in labels
    assert "Calibrated channel coordinate (m)" in labels
    assert "Calibrated wavelength (nm)" in labels
    assert "Calibrated spectral intensity" in labels
    assert "Physical spectral model" in labels
    assert "Calibrated Thomson wavelength (nm)" not in labels

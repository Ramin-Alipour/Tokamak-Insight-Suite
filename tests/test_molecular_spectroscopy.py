import numpy as np
import pytest

from tokamak_analysis.molecular_spectroscopy import (
    analyze_molecular_spectrum,
    fit_custom_molecular_model,
)


def test_molecular_spectrum_detects_features_and_integrates():
    x = np.linspace(400.0, 410.0, 1001)
    y = 2.0 + 10.0 * np.exp(-0.5 * ((x - 405.0) / 0.08) ** 2)
    out = analyze_molecular_spectrum(
        x, y, reference_bandheads=[(405.02, "band A")],
        band_ranges_nm=[(404.5, 405.5)],
    )
    assert out["peak_count"] == 1
    assert abs(out["peaks"][0]["wavelength_nm"] - 405.0) < 0.02
    assert out["integrated_signal_intensity_nm"] > 0
    assert out["reference_matches"][0]["reference_label"] == "band A"
    assert out["band_integrals"][0]["integrated_signal_intensity_nm"] > 0


def test_molecular_spectrum_rejects_bad_inputs():
    with pytest.raises(ValueError):
        analyze_molecular_spectrum([400, 402, 401, 403, 404], [1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        analyze_molecular_spectrum(np.arange(5) + 400, np.ones(5))


def test_custom_molecular_model_fit():
    x = np.linspace(400, 410, 100)
    y = 3.0 * np.exp(-0.5 * ((x - 405.0) / 0.3) ** 2) + 1.0
    def model(w, amplitude, center, sigma, baseline):
        return amplitude * np.exp(-0.5 * ((w - center) / sigma) ** 2) + baseline
    out = fit_custom_molecular_model(
        x, y, model, parameter_names=("amplitude", "center", "sigma", "baseline"),
        initial_values=(2.5, 404.8, 0.4, 0.8),
    )
    assert out["parameters"]["center"] == pytest.approx(405.0, abs=1e-3)
    assert out["parameters"]["amplitude"] == pytest.approx(3.0, rel=1e-3)

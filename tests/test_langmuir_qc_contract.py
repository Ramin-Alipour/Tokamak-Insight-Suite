import numpy as np

from tis.probe_analysis import langmuir_iv_analysis


def test_langmuir_analysis_reports_preprocessing_quality_and_duplicate_aggregation():
    voltage = np.r_[np.linspace(-5.0, 5.0, 11), 0.0, np.nan]
    current = np.r_[np.linspace(-2e-3, 2e-3, 11), 1e-3, 0.0]

    result = langmuir_iv_analysis(voltage, current)

    quality = result["input_quality"]
    assert quality["input_pair_count"] == 13
    assert quality["finite_pair_count"] == 12
    assert quality["discarded_nonfinite_pair_count"] == 1
    assert quality["duplicate_voltage_pair_count_aggregated"] == 1
    assert quality["processed_unique_voltage_count"] == 11
    assert quality["current_convention_normalized"] == "electron_positive"


def test_langmuir_analysis_exposes_electron_fit_residual_diagnostics():
    voltage = np.linspace(-10.0, 10.0, 81)
    current = -1e-3 + 2e-4 * np.exp((voltage + 10.0) / 3.0)

    result = langmuir_iv_analysis(
        voltage,
        current,
        electron_fit_range=(-8.0, 0.0),
    )

    assert result["electron_temperature_eV"] is not None
    assert result["electron_retarding_fit_points"] >= 4
    assert np.isfinite(result["electron_retarding_fit_R2"])
    assert np.isfinite(result["electron_retarding_fit_residual_rms_A"])
    assert np.isfinite(result["electron_retarding_fit_normalized_rmse"])
    assert result["electron_retarding_fit_status"] == "fit_completed_review_R2_and_residuals"

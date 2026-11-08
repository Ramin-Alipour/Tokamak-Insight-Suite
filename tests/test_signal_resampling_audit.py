import numpy as np

from tis.signals import prepare_time_series_robust, resample_uniform


def test_resample_uniform_reports_nonfinite_input_samples():
    time = np.arange(12.0)
    signal = np.sin(time)
    signal[4] = np.nan

    result = resample_uniform(time, signal)

    assert result["n_input_raw"] == 12
    assert result["n_input"] == 11
    assert result["n_nonfinite_dropped"] == 1


def test_robust_preprocessing_preserves_resampling_loss_in_audit():
    time = np.arange(12.0)
    signal = np.sin(time)
    signal[4] = np.nan

    _, _, _, report = prepare_time_series_robust(
        time, signal, gap_policy="interpolate"
    )

    assert report["resampling"]["input_samples_raw"] == 12
    assert report["resampling"]["finite_samples_used"] == 11
    assert report["resampling"]["nonfinite_samples_dropped"] == 1

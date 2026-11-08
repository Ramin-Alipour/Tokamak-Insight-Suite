import numpy as np
import pytest

from tis.events import current_quench_analysis, detect_elms, detect_sawteeth


@pytest.mark.parametrize("bad", [True, "5", np.nan, np.inf, 0, -1])
def test_elm_detector_rejects_invalid_threshold_scalars(bad):
    time = np.arange(64, dtype=float)
    signal = np.sin(time)

    with pytest.raises(ValueError, match="n_sigmas"):
        detect_elms(time, signal, n_sigmas=bad)


@pytest.mark.parametrize("bad", [True, "5", np.nan, np.inf, 0, -1])
def test_sawtooth_detector_rejects_invalid_threshold_scalars(bad):
    time = np.arange(64, dtype=float)
    signal = np.sin(time)

    with pytest.raises(ValueError, match="n_sigmas"):
        detect_sawteeth(time, signal, n_sigmas=bad)


@pytest.mark.parametrize("bad", [True, "0.8", np.nan, np.inf, 0, 1])
def test_quench_rejects_invalid_fraction_scalars(bad):
    time = np.arange(16, dtype=float)
    current = np.linspace(1.0, 0.0, 16)

    with pytest.raises(ValueError):
        current_quench_analysis(time, current, upper_fraction=bad)


def test_quench_rejects_malformed_reference_window():
    time = np.arange(16, dtype=float)
    current = np.linspace(1.0, 0.0, 16)

    with pytest.raises(ValueError, match="reference_window_s"):
        current_quench_analysis(time, current, reference_window_s=("bad", "window"))

@pytest.mark.parametrize("bad_window", [("bad", "window"), (0.0,), (np.nan, 2.0)])
def test_elm_detector_rejects_malformed_analysis_window(bad_window):
    time = np.arange(64, dtype=float)
    signal = np.sin(time)

    with pytest.raises(ValueError, match="window_s"):
        detect_elms(time, signal, window_s=bad_window)

import numpy as np
import pytest

from tis.signals import sampling_audit


def test_sampling_audit_reports_regular_timestamps():
    result = sampling_audit([0.0, 1.0, 2.0, 3.0])
    assert result["monotonic"] is True
    assert result["n_samples"] == 4
    assert result["n_gaps"] == 0


@pytest.mark.parametrize("timestamps", [
    [0.0, np.nan, 2.0, 3.0],
    [0.0, np.inf, 2.0, 3.0],
    [0.0, -np.inf, 2.0, 3.0],
])
def test_sampling_audit_rejects_nonfinite_timestamps(timestamps):
    with pytest.raises(ValueError, match="finite timestamps"):
        sampling_audit(timestamps)


def test_resample_uniform_rejects_unbounded_output_grid():
    from tis.signals import resample_uniform

    time_s = np.linspace(0.0, 1.0, 8)
    values = np.sin(time_s)
    with pytest.raises(ValueError, match="exceeding the configured limit"):
        resample_uniform(time_s, values, fs=1e8, max_output_samples=1000)


def test_resample_uniform_accepts_output_within_limit():
    from tis.signals import resample_uniform

    time_s = np.linspace(0.0, 1.0, 8)
    values = np.sin(time_s)
    result = resample_uniform(time_s, values, fs=10.0, max_output_samples=100)
    assert result["n_output"] <= 100

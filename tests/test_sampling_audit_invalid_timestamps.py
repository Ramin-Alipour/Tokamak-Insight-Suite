import numpy as np
import pytest

from tis.signals import sampling_audit


@pytest.mark.parametrize("timestamps", [
    [0.0, 1.0, np.nan, 2.0, 3.0],
    [0.0, 1.0, np.inf, 2.0, 3.0],
    [0.0, 1.0, -np.inf, 2.0, 3.0],
])
def test_sampling_audit_rejects_nonfinite_timestamps(timestamps):
    with pytest.raises(ValueError, match="non-finite"):
        sampling_audit(timestamps)


def test_sampling_audit_reports_regular_finite_timestamps():
    result = sampling_audit([0.0, 1.0, 2.0, 3.0])
    assert result["monotonic"] is True
    assert result["n_gaps"] == 0
    assert result["median_sampling_rate_Hz"] == pytest.approx(1.0)

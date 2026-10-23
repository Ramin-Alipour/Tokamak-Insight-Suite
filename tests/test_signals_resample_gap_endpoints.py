import numpy as np

from tis.signals import resample_uniform


def test_resample_uniform_preserves_measured_endpoints_of_long_gaps():
    # Most source intervals are 0.1 s, but the interval 0.6 -> 1.6 s is a
    # genuine long gap. Its measured endpoints must not be masked as missing.
    time_s = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                       1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2])
    values = 2.0 * time_s + 1.0

    result = resample_uniform(time_s, values, fs=10.0, max_interp_gap_s=0.15)
    grid = result["time_s"]
    output = result["values"]

    left = np.flatnonzero(np.isclose(grid, 0.6))[0]
    right = np.flatnonzero(np.isclose(grid, 1.6))[0]
    assert np.isfinite(output[left])
    assert output[left] == values[6]
    assert np.isfinite(output[right])
    assert output[right] == values[7]

    interior = (grid > 0.6) & (grid < 1.6)
    assert np.all(np.isnan(output[interior]))
    assert result["n_nan_in_long_gaps"] == int(interior.sum())

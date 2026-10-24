import numpy as np
import pytest

from tis.signals import resample_uniform


def test_resampling_epoch_timestamps_does_not_extend_endpoint_by_absolute_time_tolerance():
    # Unix-like timestamps expose tolerances accidentally scaled by |t| rather
    # than local floating-point precision / sampling interval.
    time_s = 1_700_000_000.0 + np.arange(12) * 0.1
    values = np.arange(12, dtype=float)

    result = resample_uniform(time_s, values, fs=10.0)

    assert len(result["time_s"]) <= 12
    assert result["time_s"][-1] <= time_s[-1] + 8 * abs(np.spacing(time_s[-1]))


def test_resampling_rejects_unbounded_grid_before_allocating():
    time_s = np.linspace(0.0, 1.0, 8)
    values = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="safety limit"):
        resample_uniform(time_s, values, fs=20_000_000.0, max_output_samples=100_000)


@pytest.mark.parametrize("limit", [True, 7, 3.5, "100"])
def test_resampling_validates_output_sample_limit(limit):
    time_s = np.arange(8, dtype=float)
    values = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="max_output_samples"):
        resample_uniform(time_s, values, max_output_samples=limit)

@pytest.mark.parametrize("bad_fs", [True, "10", np.nan, np.inf, 0, -1])
def test_resampling_rejects_invalid_sampling_rate(bad_fs):
    time_s = np.arange(8, dtype=float)
    values = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="fs"):
        resample_uniform(time_s, values, fs=bad_fs)


@pytest.mark.parametrize("bad_gap", [True, "bad", np.nan, np.inf, 0, -1])
def test_resampling_rejects_invalid_gap_threshold(bad_gap):
    time_s = np.arange(8, dtype=float)
    values = np.arange(8, dtype=float)

    with pytest.raises(ValueError, match="max_interp_gap_s"):
        resample_uniform(time_s, values, max_interp_gap_s=bad_gap)

def test_resampling_large_epoch_does_not_snap_distinct_grid_points_to_duplicates():
    # At Unix-epoch magnitudes, several ULPs can exceed a high-rate sample
    # interval. Snapping by a broad ULP multiple can collapse distinct grid
    # points onto the same source timestamp.
    time_s = 1_700_000_000.0 + np.arange(100) * 1e-6
    values = np.sin(np.arange(100, dtype=float))

    result = resample_uniform(time_s, values, fs=1.05e6)

    assert np.all(np.diff(result["time_s"]) > 0)

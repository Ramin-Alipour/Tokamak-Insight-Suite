"""Regression tests for numeric argument validation in public data helpers."""
import numpy as np
import pytest

from tis.data_sources import _open_zarr_with_retry, anti_alias_decimate


BAD_NUMERIC_SCALARS = [True, np.bool_(True), "3", np.nan, np.inf, -np.inf, 2**10000]


@pytest.mark.parametrize("bad_attempts", BAD_NUMERIC_SCALARS)
def test_zarr_retry_rejects_invalid_attempts_before_network_io(bad_attempts):
    class NeverOpen:
        def open_zarr(self, *args, **kwargs):
            raise AssertionError("open_zarr must not be called for invalid attempts")

    with pytest.raises(ValueError, match="attempts"):
        _open_zarr_with_retry(NeverOpen(), "https://example.invalid/data.zarr", "group", attempts=bad_attempts)


@pytest.mark.parametrize("bad_timeout", BAD_NUMERIC_SCALARS)
def test_zarr_retry_rejects_invalid_timeout_before_network_io(bad_timeout):
    class NeverOpen:
        def open_zarr(self, *args, **kwargs):
            raise AssertionError("open_zarr must not be called for invalid timeout")

    with pytest.raises(ValueError, match="timeout_s"):
        _open_zarr_with_retry(NeverOpen(), "https://example.invalid/data.zarr", "group", timeout_s=bad_timeout)


@pytest.mark.parametrize("bad_max_samples", [True, np.bool_(True), "1024", np.nan, np.inf, -np.inf, 2**10000, 1023])
def test_decimator_rejects_invalid_sample_limits(bad_max_samples):
    time_s = np.arange(1024, dtype=float) * 1e-6
    values = np.sin(np.arange(1024, dtype=float) * 0.1)
    with pytest.raises(ValueError, match="max_samples"):
        anti_alias_decimate(time_s, values, max_samples=bad_max_samples)

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


from tis.data_sources import fetch_fair_mast_efit_snapshot, fetch_public_table_bytes


@pytest.mark.parametrize("timeout", [True, np.bool_(True), "30", np.nan, np.inf, 10**1000])
def test_public_download_rejects_invalid_timeout_before_network(timeout):
    with pytest.raises(ValueError, match="Timeout"):
        fetch_public_table_bytes("https://example.com/data.csv", timeout=timeout)


@pytest.mark.parametrize("max_bytes", [True, np.bool_(True), 1023, 1_000_000_001, 10**1000])
def test_public_download_enforces_reasonable_size_limit(max_bytes):
    with pytest.raises(ValueError, match="Maximum download size"):
        fetch_public_table_bytes("https://example.com/data.csv", max_bytes=max_bytes)


@pytest.mark.parametrize("time_s", [True, np.bool_(True), "0.2", np.nan, np.inf, 10**1000])
def test_efit_snapshot_rejects_invalid_requested_time(time_s):
    with pytest.raises(ValueError, match="Requested EFIT time"):
        fetch_fair_mast_efit_snapshot(time_s=time_s)

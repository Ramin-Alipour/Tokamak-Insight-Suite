"""Regression tests for invalid numeric parameters in public-data helpers."""
import numpy as np
import pytest

from tis.data_sources import _open_zarr_with_retry, anti_alias_decimate


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 2.5, "3"])
def test_anti_alias_decimate_invalid_max_samples_raise_value_error(value):
    time = np.arange(1024, dtype=float) * 1e-6
    signal = np.sin(np.arange(1024, dtype=float))
    with pytest.raises(ValueError, match="max_samples"):
        anti_alias_decimate(time, signal, max_samples=value)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, True, 2.5, "3"])
def test_zarr_retry_invalid_attempts_raise_value_error_before_io(value):
    class NeverOpen:
        def open_zarr(self, *args, **kwargs):
            raise AssertionError("open_zarr must not be called for invalid attempts")

    with pytest.raises(ValueError, match="attempts"):
        _open_zarr_with_retry(NeverOpen(), "https://example.invalid/data.zarr", "group", attempts=value)

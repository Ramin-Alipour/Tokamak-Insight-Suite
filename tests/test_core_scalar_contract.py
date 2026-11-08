"""Regression tests for scalar contracts in time-series helpers."""
import numpy as np
import pytest

from tis.core import integrate_time_series, prepare_time_series


@pytest.mark.parametrize("bad_rate", ["1000", True, np.bool_(True), np.nan, np.inf, 10**1000])
def test_prepare_time_series_rejects_invalid_sample_rate_scalars(bad_rate):
    with pytest.raises(ValueError, match="sample rate"):
        prepare_time_series(None, np.arange(8, dtype=float), sample_rate=bad_rate)


@pytest.mark.parametrize("bad_scale", ["1.0", True, np.bool_(False), np.nan, np.inf])
def test_integrator_rejects_invalid_scale_scalars(bad_scale):
    with pytest.raises(ValueError, match="Scale and initial value"):
        integrate_time_series([0.0, 1.0], [1.0, 1.0], scale=bad_scale)


def test_integrator_rejects_non_numeric_baseline_window_without_type_error():
    with pytest.raises(ValueError, match="Baseline interval"):
        integrate_time_series(
            [0.0, 1.0, 2.0], [1.0, 1.0, 1.0],
            baseline_start_s="0", baseline_end_s=1.0,
        )

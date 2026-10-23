import numpy as np
import pytest

from tis.events import interval_statistics


def test_interval_statistics_rejects_nonchronological_event_times():
    with pytest.raises(ValueError, match="strictly increasing"):
        interval_statistics([1.0, 0.5, 2.0])


def test_interval_statistics_rejects_nonfinite_event_times():
    with pytest.raises(ValueError, match="finite"):
        interval_statistics([1.0, np.nan])

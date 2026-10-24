import numpy as np
import pytest

from tis.signals import longest_contiguous_segment


@pytest.mark.parametrize("gap_factor", [0, 1, -1, np.nan, np.inf, True, "1.5"])
def test_longest_contiguous_segment_rejects_invalid_gap_factor(gap_factor):
    with pytest.raises(ValueError, match="gap_factor"):
        longest_contiguous_segment(np.arange(8.0), np.arange(8.0), gap_factor=gap_factor)


def test_longest_contiguous_segment_accepts_valid_gap_factor():
    time = np.r_[np.arange(8.0), np.arange(10.0, 18.0)]
    values = np.arange(time.size, dtype=float)
    result = longest_contiguous_segment(time, values, gap_factor=1.5)
    assert result["n_samples_kept"] == 8
    assert result["n_segments"] == 2

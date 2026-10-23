import pytest

from tis.plasma import lh_threshold_martin08


@pytest.mark.parametrize("density", [0.0, -1.0, float("nan"), float("inf")])
def test_martin_lh_threshold_rejects_nonpositive_or_nonfinite_density(density):
    with pytest.raises(ValueError, match="ne_line_avg_m3 must be positive and finite"):
        lh_threshold_martin08(density, 2.0, 10.0)


def test_martin_lh_threshold_accepts_positive_density():
    result = lh_threshold_martin08(2e19, 2.0, 10.0)
    assert result["P_LH_MW"] > 0

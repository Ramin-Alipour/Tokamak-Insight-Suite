import numpy as np
import pytest

from tis.signals import noise_floor_detection_limit


def test_noise_floor_robust_scale_and_empirical_threshold():
    rng = np.random.default_rng(42)
    background = rng.normal(loc=3.0, scale=2.0, size=20_000)

    result = noise_floor_detection_limit(background, false_alarm_probability=1e-3)

    assert abs(result["baseline"] - 3.0) < 0.1
    assert abs(result["noise_floor_sigma"] - 2.0) < 0.1
    assert result["empirical_detection_threshold_above_baseline"] > result["noise_floor_sigma"]
    assert result["gaussian_reference_threshold_above_baseline"] > result["noise_floor_sigma"]
    assert result["n_background_samples"] == 20_000
    assert result["two_sided"] is False


def test_noise_floor_supports_two_sided_threshold():
    background = np.linspace(-1.0, 1.0, 101)
    result = noise_floor_detection_limit(background, two_sided=True)

    assert result["two_sided"] is True
    assert result["empirical_detection_threshold_above_baseline"] >= 0
    assert result["gaussian_reference_threshold_above_baseline"] >= 0


def test_constant_background_reports_degenerate_noise_scale():
    result = noise_floor_detection_limit(np.ones(16))

    assert result["noise_estimate_degenerate"] is True
    assert result["noise_floor_sigma"] == 0
    assert result["empirical_detection_threshold_above_baseline"] == 0


@pytest.mark.parametrize("alpha", [True, "0.01", np.nan, np.inf, 0, 0.5, -0.1])
def test_noise_floor_rejects_invalid_false_alarm_probability(alpha):
    with pytest.raises(ValueError, match="false_alarm_probability"):
        noise_floor_detection_limit(np.arange(16, dtype=float), false_alarm_probability=alpha)


def test_noise_floor_rejects_nonfinite_or_short_background():
    with pytest.raises(ValueError, match="At least 8"):
        noise_floor_detection_limit(np.arange(7, dtype=float))
    values = np.arange(16, dtype=float)
    values[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        noise_floor_detection_limit(values)

@pytest.mark.parametrize("two_sided", [1, 0, "true", None])
def test_noise_floor_rejects_non_boolean_two_sided_flag(two_sided):
    with pytest.raises(ValueError, match="two_sided"):
        noise_floor_detection_limit(np.arange(16, dtype=float), two_sided=two_sided)

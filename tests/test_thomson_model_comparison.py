import numpy as np
import pytest

from tokamak_analysis.thomson_model_comparison import compare_thomson_forward_models


def test_model_comparison_ranks_same_data_predictions_with_full_covariance():
    x = np.linspace(-1.0, 1.0, 31)
    measured = np.exp(-x**2) + 0.002 * np.sin(7 * x)
    result = compare_thomson_forward_models(
        measured,
        {"close": measured.copy(), "poor": measured + 0.03},
        {"close": 3, "poor": 2},
        spectral_covariance=np.eye(x.size) * 1e-4,
    )
    assert result["models"]["close"]["weighted_residual_sum_squares"] == pytest.approx(0.0)
    assert result["models"]["close"]["delta_AICc"] == pytest.approx(0.0)
    assert result["models"]["poor"]["delta_BIC"] > 0
    assert result["noise_model"] == "supplied_full_Gaussian_covariance"


def test_model_comparison_profiles_common_unknown_variance():
    y = np.linspace(0.0, 1.0, 20)
    result = compare_thomson_forward_models(
        y, {"m1": y + 0.01, "m2": y + 0.03}, {"m1": 2, "m2": 1}
    )
    assert result["models"]["m1"]["residual_variance_MLE"] is not None
    assert result["models"]["m1"]["effective_parameter_count"] == 3
    assert result["ranking_metric"] == "AICc"


def test_model_comparison_rejects_incomparable_uncertainty_contracts():
    y = np.linspace(0.0, 1.0, 10)
    with pytest.raises(ValueError, match="only one"):
        compare_thomson_forward_models(
            y, {"a": y, "b": y + 0.1}, {"a": 1, "b": 1},
            spectral_covariance=np.eye(10), pointwise_uncertainty=np.ones(10)
        )

import numpy as np
import pytest

from tis.products.thomson.physics.thomson_distribution import (
    _model_averaged_projected_evdf_moments,
    _piecewise_linear_moment_weights,
    _trapz,
)


def _normal_pdf(velocity, mean, sigma):
    values = np.exp(-0.5 * ((velocity - mean) / sigma) ** 2)
    return values / _trapz(values, velocity)


def test_model_averaged_evdf_moments_are_computed_from_the_mixture_not_averaged_moments():
    velocity = np.linspace(-7.0, 7.0, 701)
    first = _normal_pdf(velocity, -1.2, 0.8)
    second = _normal_pdf(velocity, 1.7, 1.1)
    weights = {"narrow_left": 0.35, "broad_right": 0.65}
    mixture = weights["narrow_left"] * first + weights["broad_right"] * second
    delta_first = first - mixture
    delta_second = second - mixture
    between = (
        weights["narrow_left"] * np.outer(delta_first, delta_first)
        + weights["broad_right"] * np.outer(delta_second, delta_second)
    )
    total = between + np.eye(velocity.size) * 1.0e-14

    result = _model_averaged_projected_evdf_moments(
        velocity,
        {
            "model_averaged_projected_evdf_s_per_m": mixture,
            "projected_evdf_velocity_grid_m_s": velocity,
            "projected_evdf_candidate_weights": weights,
            "between_model_projected_evdf_covariance_s2_per_m2": between,
            "model_averaged_projected_evdf_covariance_s2_per_m2": total,
        },
    )

    moments = result["physical_moments"]
    normalization = _trapz(mixture, velocity)
    expected_mean = (
        _piecewise_linear_moment_weights(velocity, 1) @ mixture
    ) / normalization
    expected_variance = (
        _piecewise_linear_moment_weights(velocity, 2, expected_mean) @ mixture
    ) / normalization
    assert result["status"] == "within_and_between_candidate_family_covariance_propagated"
    assert result["evdf_probability_normalization"] == pytest.approx(1.0, abs=1e-10)
    assert moments["mean_velocity_parallel_m_s"] == pytest.approx(expected_mean, rel=1e-10)
    assert moments["velocity_parallel_variance_m2_s2"] == pytest.approx(expected_variance, rel=1e-10)
    assert moments["moment_covariance"] is not None
    assert np.linalg.eigvalsh(moments["moment_covariance"]).min() > -1e-10
    assert result["moment_covariance_scope"] == "within_and_between_candidate_family"


def test_model_average_without_within_model_covariance_is_never_labelled_total():
    velocity = np.linspace(-6.0, 6.0, 301)
    first = _normal_pdf(velocity, -0.5, 0.9)
    second = _normal_pdf(velocity, 0.8, 1.2)
    mixture = 0.5 * (first + second)
    between = 0.25 * np.outer(first - second, first - second)

    result = _model_averaged_projected_evdf_moments(
        velocity,
        {
            "model_averaged_projected_evdf_s_per_m": mixture,
            "between_model_projected_evdf_covariance_s2_per_m2": between,
        },
    )

    assert result["status"] == "between_candidate_family_covariance_only"
    assert result["within_model_covariance_available"] is False
    assert result["physical_moments"]["moment_covariance"] is not None


def test_model_average_rejects_a_mismatched_velocity_grid_and_unnormalized_mixture():
    velocity = np.linspace(-5.0, 5.0, 101)
    pdf = _normal_pdf(velocity, 0.0, 1.0)
    with pytest.raises(ValueError, match="differs from the selected inverse velocity grid"):
        _model_averaged_projected_evdf_moments(
            velocity,
            {
                "model_averaged_projected_evdf_s_per_m": pdf,
                "projected_evdf_velocity_grid_m_s": velocity + 0.1,
            },
        )
    with pytest.raises(ValueError, match="not normalized"):
        _model_averaged_projected_evdf_moments(
            velocity,
            {"model_averaged_projected_evdf_s_per_m": 2.0 * pdf},
        )

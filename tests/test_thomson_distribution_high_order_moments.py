"""Physical regression tests for high-order projected Thomson EVDF moments."""
import numpy as np
from scipy.integrate import trapezoid
import pytest

from tis.products.thomson.physics.thomson_distribution import (
    _piecewise_linear_moment_weights,
    _projected_evdf_grid_truncation_sensitivity,
    _projected_evdf_physical_moments,
)


def test_piecewise_linear_sixth_order_moment_weights_integrate_exactly():
    velocity = np.array([-2.0, -0.7, 0.1, 1.4, 3.0])
    density = np.array([0.2, 1.1, 0.8, 0.4, 0.1])
    weights = _piecewise_linear_moment_weights(velocity, 6, center_m_s=0.25)

    # Independent interval-by-interval 8-point quadrature of the same
    # piecewise-linear interpolant times (v-center)^6.
    nodes, quadrature = np.polynomial.legendre.leggauss(8)
    expected = 0.0
    for left, right, f_left, f_right in zip(
        velocity[:-1], velocity[1:], density[:-1], density[1:]
    ):
        points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
        fraction = (points - left) / (right - left)
        interpolated = (1.0 - fraction) * f_left + fraction * f_right
        expected += 0.5 * (right - left) * np.sum(
            quadrature * (points - 0.25) ** 6 * interpolated
        )

    assert weights @ density == pytest.approx(expected, rel=2e-13, abs=1e-13)


def test_gaussian_projected_evdf_has_expected_fifth_and_sixth_moments():
    sigma = 2.0e5
    velocity = np.linspace(-8.0 * sigma, 8.0 * sigma, 321)
    density = np.exp(-0.5 * (velocity / sigma) ** 2)
    density /= trapezoid(density, velocity)
    covariance = np.eye(velocity.size) * 1.0e-24

    result = _projected_evdf_physical_moments(
        velocity, density, covariance
    )

    assert result["central_fifth_velocity_moment_m5_s5"] == pytest.approx(
        0.0, abs=2.0e-4 * sigma**5
    )
    assert result["central_sixth_velocity_moment_m6_s6"] == pytest.approx(
        15.0 * sigma**6, rel=2.0e-3
    )
    assert result["standardized_fifth_central_moment"] == pytest.approx(
        0.0, abs=2.0e-4
    )
    assert result["excess_sixth_standardized_central_moment"] == pytest.approx(
        0.0, abs=2.0e-3
    )
    assert result["moment_covariance"].shape == (14, 14)
    assert np.allclose(
        result["moment_covariance"],
        result["moment_covariance"].T,
        rtol=1e-10,
        atol=1e-30,
    )
    assert result["random_energy_moment_covariance"].shape == (2, 2)
    sensitivity = _projected_evdf_grid_truncation_sensitivity(velocity, density)
    assert "central_fifth_velocity_moment_m5_s5" in sensitivity["full_grid_moments"]
    assert "central_sixth_velocity_moment_m6_s6" in sensitivity["full_grid_moments"]
    assert "excess_sixth_standardized_central_moment" in sensitivity["interior_minus_full_moments"]

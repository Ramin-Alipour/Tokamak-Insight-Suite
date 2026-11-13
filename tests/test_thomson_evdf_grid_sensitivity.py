"""Physics checks for projected Thomson EVDF velocity-grid truncation diagnostics."""
import numpy as np
import pytest

from tis.products.thomson.physics.thomson_distribution import (
    _projected_evdf_grid_truncation_sensitivity,
)


def test_symmetric_maxwellian_like_evdf_has_small_edge_mass_and_zero_skewness():
    velocity = np.linspace(-5.0e6, 5.0e6, 401)
    thermal_speed = 1.0e6
    density = np.exp(-0.5 * (velocity / thermal_speed) ** 2)
    result = _projected_evdf_grid_truncation_sensitivity(velocity, density)

    assert result["status"] == "computed_one_edge_cell_truncation_sensitivity"
    assert result["boundary_interval_probability_fraction"] < 1.0e-5
    assert result["boundary_interval_bounds_m_s"] == [[-5.0e6, -4.975e6], [4.975e6, 5.0e6]]
    assert result["boundary_node_density_fraction_of_peak"] < 1.0e-5
    assert abs(result["full_grid_moments"]["mean_velocity_parallel_m_s"]) < 1.0e-7
    assert abs(result["full_grid_moments"]["skewness"]) < 1.0e-10
    assert result["full_grid_moments"]["excess_kurtosis"] == pytest.approx(0.0, abs=0.02)


def test_asymmetric_heavy_tail_reports_distinct_high_order_grid_sensitivity():
    velocity = np.linspace(-4.0e6, 8.0e6, 301)
    core = np.exp(-0.5 * (velocity / 1.0e6) ** 2)
    tail = 0.08 * np.exp(-np.maximum(velocity, 0.0) / 3.0e6)
    density = core + tail
    result = _projected_evdf_grid_truncation_sensitivity(velocity, density)

    assert result["boundary_interval_probability_fraction"] > 0.0
    assert result["boundary_node_density_fraction_of_peak"] > 0.0
    assert result["interior_minus_full_moments"]["central_fourth_velocity_moment_m4_s4"] != 0.0
    assert result["interpretation"].startswith("The interior-minus-full differences")
    assert "confidence" in result["interpretation"]


def test_grid_truncation_diagnostic_rejects_invalid_velocity_grid():
    with pytest.raises(ValueError, match="at least five nodes"):
        _projected_evdf_grid_truncation_sensitivity(
            [0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 0.0]
        )


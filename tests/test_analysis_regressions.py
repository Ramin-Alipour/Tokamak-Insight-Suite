import numpy as np
import pytest

from tokamak_analysis.safety_factor import estimate_safety_factor
from tokamak_analysis.neutron_flux import analyze_neutron_flux
from tokamak_analysis.spectral_line_ratio import analyze_line_ratio
from tokamak_analysis.wall_recycling import recycling_coefficient
from tokamak_analysis.measurement_uncertainty import propagate_product_uncertainty
from tokamak_analysis.channel_synchronization import synchronize_channels
from tokamak_analysis.plasma_geometry import geometry_from_boundary


def test_cylindrical_safety_factor_uses_minor_over_major_radius():
    result = estimate_safety_factor(2.0, 0.5, 1.0, 0.1)
    assert result["q_cylindrical_estimate"] == pytest.approx(2.5)


def test_neutron_flux_treats_input_as_binned_counts_by_default():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [10.0, 20.0, 30.0])
    np.testing.assert_allclose(result["count_rate_s-1"], [10.0, 20.0, 30.0])
    assert result["total_counts"] == 60.0


def test_neutron_flux_supports_cumulative_counters_explicitly():
    result = analyze_neutron_flux([0.0, 1.0, 2.0], [0.0, 10.0, 30.0], counts_are_cumulative=True)
    assert result["total_counts"] == 30.0
    np.testing.assert_allclose(result["count_rate_s-1"], [10.0, 15.0, 20.0])


def test_line_ratio_uncertainty_is_finite_for_zero_numerator():
    result = analyze_line_ratio(0.0, 2.0, uncertainty_numerator=0.1, uncertainty_denominator=0.2)
    assert result["line_ratio"] == 0.0
    assert result["line_ratio_uncertainty"] == pytest.approx(0.05)


def test_recycling_uncertainty_is_finite_for_zero_returned_flux():
    result = recycling_coefficient(2.0, 0.0, uncertainties=(0.1, 0.2))
    assert result["recycling_coefficient"] == 0.0
    assert result["uncertainty"] == pytest.approx(0.1)


def test_product_uncertainty_handles_zero_factor():
    result = propagate_product_uncertainty([0.0, 3.0], [2.0, 0.1])
    assert result["value"] == 0.0
    assert result["standard_uncertainty"] == pytest.approx(6.0)
    assert np.isnan(result["relative_uncertainty"])


def test_synchronization_rejects_extrapolation():
    with pytest.raises(ValueError, match="inside"):
        synchronize_channels([[0.0, 1.0, 2.0], [0.5, 1.0, 1.5]],
                              [[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]],
                              target_time=[0.0, 1.0])


def test_geometry_rejects_boundary_crossing_symmetry_axis():
    with pytest.raises(ValueError, match="symmetry axis"):
        geometry_from_boundary([-0.1, 0.5, 0.5, -0.1, -0.1],
                               [0.0, 0.0, 1.0, 1.0, 0.0])

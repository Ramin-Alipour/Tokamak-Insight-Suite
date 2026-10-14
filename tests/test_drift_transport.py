"""Tests for paper-style E×B transport equations and product/covariance distinction."""
import numpy as np
import pytest

from tis.core import drift_transport_from_fields


def test_drift_transport_recovers_known_e_cross_b_quantities():
    density = np.array([1.0, 2.0, 3.0]) * 1e19
    er = np.array([100.0, 200.0, 300.0])
    ep = np.array([200.0, 400.0, 600.0])
    result = drift_transport_from_fields(density, er, ep, 0.5)
    np.testing.assert_allclose(result["radial_velocity_m_s"], [400, 800, 1200])
    np.testing.assert_allclose(result["poloidal_velocity_m_s"], [200, 400, 600])
    np.testing.assert_allclose(result["radial_particle_flux_m-2_s-1"], [4e21, 16e21, 36e21])
    np.testing.assert_allclose(result["reynolds_stress_velocity_product_m2_s-2"], [8e4, 3.2e5, 7.2e5])


def test_drift_transport_keeps_mean_product_separate_from_fluctuation_covariance():
    density = np.array([1.0, 2.0, 3.0]) * 1e19
    er = np.array([100.0, 200.0, 300.0])
    ep = np.array([200.0, 400.0, 600.0])
    result = drift_transport_from_fields(density, er, ep, 0.5)
    assert result["mean_radial_particle_flux_m-2_s-1"] != pytest.approx(
        result["radial_particle_flux_fluctuation_covariance_m-2_s-1"]
    )


def test_drift_transport_requires_explicit_valid_sign_and_field():
    with pytest.raises(ValueError, match="nonzero"):
        drift_transport_from_fields([1, 2], [1, 2], [3, 4], 0.0)
    with pytest.raises(ValueError, match="sign"):
        drift_transport_from_fields([1, 2], [1, 2], [3, 4], 1.0, radial_velocity_sign=0)


def test_drift_transport_uses_common_finite_mask():
    result = drift_transport_from_fields([1, np.nan, 3], [2, 100, 6], [4, 200, 8], 2.0)
    assert result["n_common_finite_samples"] == 2
    np.testing.assert_allclose(result["radial_velocity_m_s"], [2, 4])

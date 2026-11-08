import numpy as np
import pytest

from tokamak_analysis.thomson_gap_closure import (
    map_thomson_channels_to_normalized_poloidal_flux,
)


def test_normalized_poloidal_flux_uses_axis_to_separatrix_definition():
    result = map_thomson_channels_to_normalized_poloidal_flux(
        [0.10, 0.225, 0.60],
        magnetic_axis_flux_wb=0.10,
        separatrix_flux_wb=0.60,
    )
    np.testing.assert_allclose(result["normalized_poloidal_flux"], [0.0, 0.25, 1.0])
    np.testing.assert_allclose(result["rho_pol"], [0.0, 0.5, 1.0])
    assert result["outside_lcfs_channel_indices"].size == 0
    assert result["rho_pol_covariance"] is None


def test_flux_coordinate_propagates_joint_channel_axis_and_edge_covariance():
    psi = np.array([0.16, 0.30, 0.46])
    axis, edge = 0.10, 0.50
    covariance = np.diag([1e-6, 2e-6, 1e-6, 3e-6, 4e-6])
    result = map_thomson_channels_to_normalized_poloidal_flux(
        psi, magnetic_axis_flux_wb=axis, separatrix_flux_wb=edge,
        joint_flux_covariance=covariance,
    )
    rho = result["rho_pol"]
    assert np.all(np.isfinite(result["rho_pol_covariance"]))
    assert np.all(np.linalg.eigvalsh(result["rho_pol_covariance"]) >= -1e-14)
    assert np.all(result["rho_pol_standard_uncertainty"] > 0.0)
    # Independent central finite-difference check of the axis-flux Jacobian.
    eps = 1e-7
    plus = np.sqrt((psi[1] - (axis + eps)) / (edge - (axis + eps)))
    minus = np.sqrt((psi[1] - (axis - eps)) / (edge - (axis - eps)))
    derivative = (plus - minus) / (2.0 * eps)
    expected_variance = derivative**2 * covariance[3, 3]
    assert result["rho_pol_covariance"][1, 1] >= expected_variance
    assert result["uncertainty_model"] == "full_joint_flux_axis_separatrix_covariance_delta_method"


def test_flux_coordinate_reports_sol_channels_without_clipping_rho():
    result = map_thomson_channels_to_normalized_poloidal_flux(
        [0.2, 0.6, 0.8],
        magnetic_axis_flux_wb=0.1,
        separatrix_flux_wb=0.6,
    )
    assert result["rho_pol"][2] > 1.0
    np.testing.assert_array_equal(result["outside_lcfs_channel_indices"], [2])


def test_flux_coordinate_rejects_wrong_flux_orientation_and_invalid_covariance():
    with pytest.raises(ValueError, match="opposite side"):
        map_thomson_channels_to_normalized_poloidal_flux(
            [0.05, 0.2], magnetic_axis_flux_wb=0.1, separatrix_flux_wb=0.6
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        map_thomson_channels_to_normalized_poloidal_flux(
            [0.2, 0.3], magnetic_axis_flux_wb=0.1, separatrix_flux_wb=0.6,
            joint_flux_covariance=np.diag([1.0, 1.0, 1.0, 1.0]) * -1.0,
        )


def test_advanced_physics_dispatch_exposes_flux_coordinate_mapping():
    from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics

    result = run_advanced_thomson_physics(
        "thomson_normalized_poloidal_flux",
        poloidal_flux_wb=[0.2, 0.3],
        magnetic_axis_flux_wb=0.1,
        separatrix_flux_wb=0.5,
    )
    np.testing.assert_allclose(result["rho_pol"], np.sqrt([0.25, 0.5]))

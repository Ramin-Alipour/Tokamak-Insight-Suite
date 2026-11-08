import numpy as np
import pytest

from tokamak_analysis.thomson_workflow import _reconstruct_inferred_parameter_profile


def test_profile_reconstruction_jointly_samples_coordinate_value_cross_covariance():
    radius = np.linspace(0.1, 0.5, 5)
    values = np.linspace(2.0, 6.0, 5)
    coordinate_covariance = np.eye(5) * 1.0e-8
    value_covariance = np.eye(5) * 4.0e-4
    cross_covariance = np.eye(5) * 5.0e-7

    result = _reconstruct_inferred_parameter_profile(
        radius,
        values,
        covariance=value_covariance,
        coordinate_covariance=coordinate_covariance,
        coordinate_value_cross_covariance=cross_covariance,
        positive=True,
        method="pchip",
        points=40,
        mc_samples=80,
        random_seed=12,
    )

    assert result["coordinate_value_cross_covariance_used"] is True
    assert result["coordinate_uncertainty_model"] == (
        "joint_researcher_supplied_coordinate_and_profile_value_covariance"
    )
    profile = result["methods"]["pchip"]
    assert profile["uncertainty_available"] is True
    assert profile["uncertainty_mc_successful_samples"] >= 20


def test_profile_reconstruction_rejects_non_psd_joint_coordinate_value_covariance():
    radius = np.linspace(0.1, 0.5, 5)
    values = np.linspace(2.0, 6.0, 5)
    coordinate_covariance = np.eye(5) * 1.0e-8
    value_covariance = np.eye(5) * 4.0e-4
    # This cross block is too large for the two supplied marginal covariances.
    cross_covariance = np.eye(5) * 1.0e-4

    with pytest.raises(ValueError, match="joint coordinate/profile-value covariance"):
        _reconstruct_inferred_parameter_profile(
            radius,
            values,
            covariance=value_covariance,
            coordinate_covariance=coordinate_covariance,
            coordinate_value_cross_covariance=cross_covariance,
            positive=True,
            method="pchip",
            points=40,
            mc_samples=80,
        )


def test_profile_coordinate_value_cross_covariance_requires_both_marginals():
    radius = np.linspace(0.1, 0.5, 5)
    values = np.linspace(2.0, 6.0, 5)

    with pytest.raises(ValueError, match="requires both profile_coordinate_covariance"):
        _reconstruct_inferred_parameter_profile(
            radius,
            values,
            coordinate_value_cross_covariance=np.zeros((5, 5)),
            method="pchip",
            points=40,
        )

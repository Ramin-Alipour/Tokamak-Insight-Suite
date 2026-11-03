import numpy as np
import pytest

from tokamak_analysis.thomson_advanced_closure import (
    assemble_joint_covariance,
    profile_regularization_sensitivity,
    propagate_pca_reference_uncertainty,
)


def test_joint_covariance_preserves_supplied_cross_blocks_and_reports_missing_ones():
    a = np.array([[2.0, 0.2], [0.2, 1.0]])
    b = np.array([[1.5]])
    cross = np.array([[0.1], [0.05]])
    result = assemble_joint_covariance([a, b, np.array([[0.7]])], {(0, 1): cross})
    expected = np.block([
        [a, cross, np.zeros((2, 1))],
        [cross.T, b, np.zeros((1, 1))],
        [np.zeros((1, 2)), np.zeros((1, 1)), np.array([[0.7]])],
    ])
    np.testing.assert_allclose(result["covariance"], expected)
    assert result["supplied_cross_blocks"] == ((0, 1),)
    assert result["zero_assumed_cross_blocks"] == ((0, 2), (1, 2))


def test_joint_covariance_rejects_non_psd_joint_block():
    with pytest.raises(ValueError, match="assembled_joint_covariance"):
        assemble_joint_covariance(
            [np.array([[1.0]]), np.array([[1.0]])],
            {(0, 1): np.array([[1.2]])},
        )


def test_reference_pca_propagates_query_and_reference_covariance():
    rng = np.random.default_rng(4)
    reference = rng.normal(size=(8, 5))
    query = reference.mean(axis=0) + np.array([0.2, -0.1, 0.3, 0.0, 0.1])
    ref_cov = np.eye(reference.size) * 1.0e-5
    query_cov = np.eye(query.size) * 2.0e-4
    result = propagate_pca_reference_uncertainty(
        reference, query, reference_covariance=ref_cov,
        query_covariance=query_cov, n_components=3,
    )
    assert result["scores"].shape == (3,)
    assert result["components"].shape == (5, 3)
    assert result["score_covariance"].shape == (3, 3)
    assert np.linalg.eigvalsh(result["score_covariance"]).min() >= -1e-12
    assert result["reference_uncertainty_propagated"]
    assert result["query_uncertainty_propagated"]


def test_profile_regularization_reports_data_identifiability_separately():
    x = np.linspace(0.0, 1.0, 7)
    matrix = np.column_stack((np.ones(7), x, x**2, x**3))
    truth = np.array([1.0, 2.0, 0.5, 0.2])
    y = matrix @ truth
    result = profile_regularization_sensitivity(
        y, matrix, np.eye(7) * 0.01,
        strengths=np.array([0.0, 0.01, 1.0]),
        methods=("identity", "second_difference"),
    )
    assert result["data_only_identifiable"]
    assert result["data_only_rank"] == 4
    assert len(result["candidates"]) == 6
    assert result["selection"] == "none; compare residual/roughness and use external physical constraints"


def test_profile_regularization_does_not_hide_forward_null_space():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    result = profile_regularization_sensitivity(
        np.array([1.0, 2.0, 3.0]), matrix, np.eye(3),
        strengths=np.array([0.1, 1.0]),
    )
    assert result["data_only_rank"] == 2
    assert not result["data_only_identifiable"]
    assert "null space" in result["null_space_warning"]

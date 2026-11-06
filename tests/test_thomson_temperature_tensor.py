import numpy as np
import pytest

from tokamak_analysis.thomson_temperature_tensor import (
    infer_full_temperature_tensor_from_thomson_projections,
)


def _views():
    q = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ], dtype=float)
    q /= np.linalg.norm(q, axis=1)[:, None]
    tensor = np.array([[800.0, 90.0, -35.0],
                       [90.0, 420.0, 55.0],
                       [-35.0, 55.0, 260.0]])
    projected = np.einsum("vi,ij,vj->v", q, tensor, q)
    return q, tensor, projected


def test_full_temperature_tensor_recovers_non_gyrotropic_tensor_with_gls_covariance():
    q, expected, projected = _views()
    covariance = np.eye(q.shape[0]) * 0.04
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected, q, projected_temperature_covariance_eV2=covariance,
    )
    np.testing.assert_allclose(result["electron_temperature_tensor_eV"], expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result["principal_temperature_eigenvalues_eV"], np.linalg.eigvalsh(expected))
    assert result["view_dyad_rank"] == 6
    assert result["parameter_covariance_eV2"].shape == (6, 6)
    assert result["reportable_as_inferred_parameter"]



def test_exactly_six_unweighted_views_do_not_claim_parameter_uncertainty():
    q, _, projected = _views()
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected[:6], q[:6],
    )
    assert result["degrees_of_freedom"] == 0
    assert result["parameter_covariance_eV2"] is None
    assert result["parameter_covariance_status"] == (
        "not_reported_without_absolute_covariance_or_redundant_views"
    )
    assert result["reportable_as_inferred_parameter"] is True

def test_full_temperature_tensor_rejects_rank_deficient_view_geometry():
    q = np.tile([1.0, 0.0, 0.0], (6, 1))
    with pytest.raises(ValueError, match="not identifiable"):
        infer_full_temperature_tensor_from_thomson_projections(
            np.full(6, 100.0), q,
        )


def test_full_temperature_tensor_rejects_non_positive_definite_measurement_covariance():
    q, _, projected = _views()
    covariance = np.eye(q.shape[0])
    covariance[-1, -1] = 0.0
    with pytest.raises(ValueError, match="positive definite"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, projected_temperature_covariance_eV2=covariance,
        )


def test_unweighted_tensor_fit_does_not_claim_a_chi_square_probability():
    q, _, projected = _views()
    result = infer_full_temperature_tensor_from_thomson_projections(projected, q)
    assert result["consistency_p_value"] is None
    assert result["consistency_status"] == "not_testable_without_absolute_covariance_or_redundant_views"


def test_full_temperature_tensor_enforces_positive_semidefinite_boundary():
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    # The unconstrained six-view solution has eigenvalue -50 eV.
    projected = np.array([100.0, 100.0, 100.0, 250.0, 100.0, 100.0])
    result = infer_full_temperature_tensor_from_thomson_projections(projected, q)
    assert np.min(np.linalg.eigvalsh(result["electron_temperature_tensor_eV"])) >= -1e-10
    assert result["positive_semidefinite_constraint_active"] is True
    assert result["parameter_covariance_eV2"] is None
    assert result["parameter_covariance_status"] == "not_reported_psd_boundary"


def test_full_temperature_tensor_derives_pressure_tensor_from_independent_density():
    q, expected, projected = _views()
    density = 2.0e19
    result = infer_full_temperature_tensor_from_thomson_projections(
        projected, q,
        projected_temperature_covariance_eV2=0.04 * np.eye(q.shape[0]),
        electron_density_m3=density,
        electron_density_uncertainty_m3=0.1e19,
    )
    np.testing.assert_allclose(
        result["electron_pressure_tensor_Pa"],
        density * 1.602176634e-19 * expected,
        rtol=1e-10, atol=1e-10,
    )
    assert result["electron_pressure_parameter_covariance_Pa2"].shape == (6, 6)
    assert result["electron_pressure_covariance_status"] == (
        "independent_temperature_gls_and_density_uncertainty_delta_method"
    )


def test_full_temperature_tensor_does_not_infer_pressure_without_density():
    q, _, projected = _views()
    result = infer_full_temperature_tensor_from_thomson_projections(projected, q)
    assert result["electron_pressure_tensor_Pa"] is None
    assert result["electron_pressure_parameter_covariance_Pa2"] is None


def test_full_temperature_tensor_rejects_density_uncertainty_without_density():
    q, _, projected = _views()
    with pytest.raises(ValueError, match="requires electron_density_m3"):
        infer_full_temperature_tensor_from_thomson_projections(
            projected, q, electron_density_uncertainty_m3=1.0e17,
        )

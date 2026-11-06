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

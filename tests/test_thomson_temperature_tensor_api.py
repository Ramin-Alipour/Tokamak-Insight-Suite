import numpy as np

from tokamak_analysis.cts_full_physics import run_advanced_thomson_physics


def test_full_temperature_tensor_is_reachable_through_public_analysis1_dispatcher():
    q = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    q /= np.linalg.norm(q, axis=1)[:, None]
    expected = np.array([[800.0, 90.0, -35.0],
                         [90.0, 420.0, 55.0],
                         [-35.0, 55.0, 260.0]])
    projected = np.einsum("vi,ij,vj->v", q, expected, q)
    result = run_advanced_thomson_physics(
        "thomson_full_temperature_tensor",
        projected, q,
        projected_temperature_covariance_eV2=0.04*np.eye(q.shape[0]),
    )
    np.testing.assert_allclose(result["electron_temperature_tensor_eV"], expected, rtol=1e-10, atol=1e-10)
    assert result["public_physics_entrypoint"] == "thomson_full_temperature_tensor"
    assert result["reportable_as_inferred_parameter"] is True

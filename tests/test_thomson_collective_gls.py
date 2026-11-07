import numpy as np
import pytest

from tokamak_analysis.thomson_inference import _collective_spectral_whitener


def test_collective_whitener_uses_cholesky_gls_metric():
    signal = np.array([1.0, 2.0, 3.0])
    covariance = np.array([
        [4.0, 1.0, 0.0],
        [1.0, 3.0, 0.5],
        [0.0, 0.5, 2.0],
    ])
    whiten, mode = _collective_spectral_whitener(signal, covariance=covariance)
    residual = np.array([0.5, -1.0, 2.0])
    transformed = whiten(residual)
    assert mode == "full_covariance"
    assert transformed @ transformed == pytest.approx(
        residual @ np.linalg.solve(covariance, residual)
    )


def test_collective_whitener_pointwise_uncertainty_is_diagonal_gls():
    signal = np.array([1.0, 2.0, 3.0])
    sigma = np.array([2.0, 3.0, 4.0])
    whiten, mode = _collective_spectral_whitener(signal, uncertainty=sigma)
    residual = np.array([2.0, 3.0, 8.0])
    assert mode == "pointwise_uncertainty"
    assert whiten(residual) == pytest.approx(np.ones(3))


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"covariance": np.eye(3)}, "match the wavelength grid"),
        ({"covariance": np.array([[1.0, 2.0], [2.0, 1.0]])}, "positive definite"),
        ({"uncertainty": np.array([1.0, 0.0])}, "strictly positive|positive"),
    ],
)
def test_collective_whitener_rejects_invalid_noise_models(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _collective_spectral_whitener(np.array([1.0, 2.0]), **kwargs)


def test_collective_whitener_rejects_ambiguous_noise_inputs():
    with pytest.raises(ValueError, match="uncertainty or covariance"):
        _collective_spectral_whitener(
            np.array([1.0, 2.0]),
            uncertainty=np.ones(2),
            covariance=np.eye(2),
        )

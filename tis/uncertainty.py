"""Reusable uncertainty propagation primitives for diagnostic analysis chains."""
from __future__ import annotations
from typing import Callable, Any
import numpy as np


class UncertaintyError(ValueError):
    """Invalid covariance, model output or uncertainty configuration."""


def propagate_covariance(jacobian: Any, covariance: Any) -> dict[str, np.ndarray]:
    """First-order covariance propagation: C_y = J C_x J^T."""
    J = np.asarray(jacobian, dtype=float)
    C = np.asarray(covariance, dtype=float)
    if J.ndim != 2 or C.ndim != 2 or C.shape[0] != C.shape[1] or J.shape[1] != C.shape[0]:
        raise UncertaintyError("jacobian must be 2-D and covariance square with matching input dimension")
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(C)):
        raise UncertaintyError("jacobian and covariance must contain only finite values")
    covariance_scale = max(float(np.max(np.abs(C))), np.finfo(float).tiny)
    if not np.allclose(C, C.T, rtol=1e-10, atol=1e-12 * covariance_scale):
        raise UncertaintyError("covariance must be symmetric")
    C = 0.5 * (C + C.T)
    eigenvalues = np.linalg.eigvalsh(C)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise UncertaintyError("covariance must be positive semidefinite")
    result = J @ C @ J.T
    result = 0.5 * (result + result.T)
    diagonal = np.diag(result)
    if np.any(diagonal < -1e-10 * max(float(np.max(np.abs(diagonal))), np.finfo(float).tiny)):
        raise UncertaintyError("propagated covariance has negative diagonal values")
    return {"covariance": result, "standard_uncertainty": np.sqrt(np.maximum(diagonal, 0.0)),
            "method": "first-order covariance propagation; local linearization"}


def propagate_monte_carlo(model: Callable[[np.ndarray], Any], mean: Any, covariance: Any, *,
                          n_samples: int = 10000, seed: int | None = None,
                          confidence_level: float = 0.95) -> dict[str, Any]:
    """Propagate a multivariate normal input uncertainty through a nonlinear model.

    model receives an array shaped (n_samples, n_inputs) and must return a numeric
    array whose first dimension is n_samples. This function propagates the supplied
    Gaussian covariance only; non-Gaussian input distributions need a custom sampler.
    """
    mu = np.asarray(mean, dtype=float)
    C = np.asarray(covariance, dtype=float)
    if mu.ndim != 1 or mu.size < 1 or not np.all(np.isfinite(mu)):
        raise UncertaintyError("mean must be a finite 1-D vector")
    if isinstance(n_samples, bool) or int(n_samples) != n_samples or n_samples < 100:
        raise UncertaintyError("n_samples must be an integer of at least 100")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise UncertaintyError("confidence_level must lie strictly between 0 and 1")
    if C.shape != (mu.size, mu.size) or not np.all(np.isfinite(C)):
        raise UncertaintyError("covariance shape must match mean and contain only finite values")
    covariance_scale = max(float(np.max(np.abs(C))), np.finfo(float).tiny)
    if not np.allclose(C, C.T, rtol=1e-10, atol=1e-12 * covariance_scale):
        raise UncertaintyError("covariance must be symmetric")
    C = 0.5 * (C + C.T)
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise UncertaintyError("covariance must be positive semidefinite")
    factor = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    rng = np.random.default_rng(seed)
    samples = mu + rng.standard_normal((int(n_samples), mu.size)) @ factor.T
    try:
        outputs = np.asarray(model(samples), dtype=float)
    except Exception as exc:
        raise UncertaintyError(f"model failed for Monte Carlo samples: {exc}") from exc
    if outputs.ndim == 0 or outputs.shape[0] != int(n_samples):
        raise UncertaintyError("model output must have first dimension equal to n_samples")
    if not np.all(np.isfinite(outputs)):
        raise UncertaintyError("model produced non-finite outputs; handle invalid samples explicitly")
    flat = outputs.reshape(int(n_samples), -1)
    output_mean = np.mean(flat, axis=0)
    output_covariance = np.atleast_2d(np.cov(flat, rowvar=False, ddof=1))
    alpha = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(flat, [alpha, 1.0 - alpha], axis=0)
    output_shape = outputs.shape[1:]
    return {
        "mean": output_mean.reshape(output_shape),
        "standard_uncertainty": np.sqrt(np.maximum(np.diag(output_covariance), 0.0)).reshape(output_shape),
        "covariance": output_covariance,
        "confidence_interval": np.stack([lower, upper], axis=-1).reshape((*output_shape, 2)),
        "confidence_level": float(confidence_level),
        "sample_count": int(n_samples),
        "method": "Monte Carlo propagation with multivariate normal input model",
        "limitation": "Includes only the supplied covariance model; systematic, model-selection and non-Gaussian uncertainty are not inferred."
    }

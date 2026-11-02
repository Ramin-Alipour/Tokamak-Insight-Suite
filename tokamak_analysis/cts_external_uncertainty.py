from __future__ import annotations

from collections.abc import Sequence
import numpy as np


SUPPORTED_LOCAL_COMPONENTS = tuple(
    [f"magnetic_field_T_{axis}" for axis in "xyz"]
    + [f"incident_direction_{axis}" for axis in "xyz"]
    + [f"scattered_direction_{axis}" for axis in "xyz"]
    + [f"{kind}_polarization_{part}_{axis}"
       for kind in ("incident", "scattered")
       for part in ("real", "imag") for axis in "xyz"]
)


def _validated_covariance(covariance, size, name="covariance"):
    matrix = np.asarray(covariance)
    if matrix.dtype.kind not in "iuf" or matrix.ndim != 2:
        raise ValueError(f"{name} must be a real numeric matrix")
    matrix = matrix.astype(float, copy=False)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite and match the ordered input vector")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14):
        raise ValueError(f"{name} must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T


def validate_external_input_covariance(covariance, parameter_order):
    """Validate ordered covariance for local Full-EM CTS field/geometry inputs.

    Frequency-resolved ray/equilibrium and composition uncertainties are not
    represented by this local-vector contract.
    """
    if (not isinstance(parameter_order, Sequence)
            or isinstance(parameter_order, (str, bytes)) or not parameter_order):
        raise ValueError("parameter_order must be a non-empty ordered sequence")
    order = tuple(parameter_order)
    if any(not isinstance(name, str) or name not in SUPPORTED_LOCAL_COMPONENTS
           for name in order):
        raise ValueError("parameter_order contains an unsupported local Full-EM CTS component")
    if len(set(order)) != len(order):
        raise ValueError("parameter_order entries must be unique")
    return order, _validated_covariance(covariance, len(order))


def sample_external_input_vectors(base_values, covariance, *, samples=100, seed=84621):
    """Draw reproducible joint local-input perturbations from a supplied covariance."""
    base = np.asarray(base_values)
    if base.dtype.kind not in "iuf" or base.ndim != 1 or base.size == 0 or not np.all(np.isfinite(base)):
        raise ValueError("base_values must be a finite non-empty real vector")
    cov = _validated_covariance(covariance, base.size)
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 20:
        raise ValueError("samples must be an integer >= 20")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    values, vectors = np.linalg.eigh(cov)
    root = vectors * np.sqrt(np.maximum(values, 0.0))
    rng = np.random.default_rng(int(seed))
    return base[None, :] + rng.standard_normal((int(samples), base.size)) @ root.T


def propagate_cts_external_uncertainty(forward_or_inverse_model, input_mean, covariance, *,
                                       input_names=None, relative_step=None):
    """Propagate correlated external Full-EM CTS inputs through a supplied model.

    Pass a callable that returns either a forward-model observable vector or
    an inversion's fitted-parameter vector. This deliberately does not merge
    external-input uncertainty with spectral measurement-noise covariance.
    """
    from .thomson_external_uncertainty import propagate_external_covariance
    names = tuple(input_names) if input_names is not None else None
    if names is not None:
        validate_external_input_covariance(covariance, names)
    return propagate_external_covariance(
        forward_or_inverse_model, input_mean, covariance,
        input_names=names, relative_step=relative_step,
    )


def monte_carlo_cts_external_uncertainty(forward_or_inverse_model, input_mean, covariance, *,
                                         samples=1000, seed=84621, input_names=None):
    """Joint Monte-Carlo propagation for correlated external Full-EM CTS inputs."""
    from .thomson_external_uncertainty import monte_carlo_external_covariance
    names = tuple(input_names) if input_names is not None else None
    if names is not None:
        validate_external_input_covariance(covariance, names)
    return monte_carlo_external_covariance(
        forward_or_inverse_model, input_mean, covariance,
        samples=samples, seed=seed, input_names=names,
    )

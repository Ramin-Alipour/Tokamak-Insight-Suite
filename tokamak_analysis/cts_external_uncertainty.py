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
    matrix = np.asarray(covariance)
    if matrix.dtype.kind not in "iuf" or matrix.ndim != 2:
        raise ValueError("covariance must be a real numeric matrix")
    matrix = matrix.astype(float, copy=False)
    if matrix.shape != (len(order), len(order)) or not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite and match parameter_order")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-14):
        raise ValueError("covariance must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(np.min(eigenvalues)) < -1e-10 * scale:
        raise ValueError("covariance must be positive semidefinite")
    matrix = (eigenvectors * np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return order, matrix


def sample_external_input_vectors(base_values, covariance, *, samples=100, seed=84621):
    """Draw reproducible joint local-input perturbations from a supplied covariance."""
    base = np.asarray(base_values)
    if base.dtype.kind not in "iuf" or base.ndim != 1 or not np.all(np.isfinite(base)):
        raise ValueError("base_values must be a finite real vector")
    cov = np.asarray(covariance)
    if cov.shape != (base.size, base.size):
        raise ValueError("covariance shape must match base_values")
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 20:
        raise ValueError("samples must be an integer >= 20")
    values, vectors = np.linalg.eigh(0.5 * (cov + cov.T))
    root = vectors * np.sqrt(np.maximum(values, 0.0))
    rng = np.random.default_rng(seed)
    return base[None, :] + rng.standard_normal((int(samples), base.size)) @ root.T

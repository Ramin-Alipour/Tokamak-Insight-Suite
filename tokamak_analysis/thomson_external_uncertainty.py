"""Correlated external-input uncertainty propagation for Thomson and CTS inversions.

External geometry/field/response uncertainty is kept separate from the
measurement-noise covariance of the observed spectrum.
"""
from __future__ import annotations
import numpy as np


def _vector(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    a = a.astype(float, copy=False)
    if a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite non-empty vector")
    return a


def _covariance(value, size):
    c = np.asarray(value)
    if c.dtype.kind not in "iuf":
        raise ValueError("external_input_covariance must be real numeric")
    c = c.astype(float, copy=False)
    if c.shape != (size, size) or not np.all(np.isfinite(c)):
        raise ValueError("external_input_covariance must be finite and match the ordered input vector")
    if not np.allclose(c, c.T, rtol=1e-10, atol=1e-14):
        raise ValueError("external_input_covariance must be symmetric")
    c = (c + c.T) / 2
    eig, vec = np.linalg.eigh(c)
    scale = max(float(np.max(np.abs(eig))), np.finfo(float).tiny)
    if eig.min() < -1e-10 * scale:
        raise ValueError("external_input_covariance must be positive semidefinite")
    return (vec * np.maximum(eig, 0.0)) @ vec.T


def propagate_external_covariance(forward_model, input_mean, external_input_covariance, *,
                                   input_names=None, relative_step=None):
    """Propagate correlated external inputs through a deterministic forward model.

    The model receives the ordered input vector and must return a finite vector.
    C_out = J C_external J.T. The numerical Jacobian uses central differences.
    """
    x = _vector(input_mean, "input_mean")
    c = _covariance(external_input_covariance, x.size)
    if input_names is not None:
        names = tuple(input_names)
        if len(names) != x.size or len(set(names)) != len(names) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError("input_names must be unique non-empty names matching input_mean")
    else:
        names = tuple(f"input_{i}" for i in range(x.size))
    hscale = np.cbrt(np.finfo(float).eps) if relative_step is None else float(relative_step)
    if not np.isfinite(hscale) or hscale <= 0:
        raise ValueError("relative_step must be positive and finite")
    y = _vector(forward_model(x.copy()), "forward_model output")
    jac = np.empty((y.size, x.size))
    for j in range(x.size):
        h = hscale * max(abs(float(x[j])), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        yp, ym = _vector(forward_model(xp), "forward_model output"), _vector(forward_model(xm), "forward_model output")
        if yp.shape != y.shape or ym.shape != y.shape:
            raise ValueError("forward_model output dimension must remain fixed")
        jac[:, j] = (yp - ym) / (2*h)
    out = jac @ c @ jac.T
    return {
        "output_mean": y, "input_names": names, "jacobian": jac,
        "external_input_covariance": c,
        "external_output_covariance": (out + out.T)/2,
        "covariance_semantics": "external_input_uncertainty_only; distinct_from_measurement_noise_covariance",
        "method": "central_finite_difference_jacobian",
    }


def monte_carlo_external_covariance(forward_model, input_mean, external_input_covariance, *,
                                    samples=1000, seed=84621, input_names=None):
    """Joint Monte-Carlo propagation; off-diagonal input correlations are retained."""
    x = _vector(input_mean, "input_mean")
    c = _covariance(external_input_covariance, x.size)
    if isinstance(samples, (bool, np.bool_)) or not isinstance(samples, (int, np.integer)) or samples < 20:
        raise ValueError("samples must be an integer >= 20")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if input_names is not None and len(tuple(input_names)) != x.size:
        raise ValueError("input_names must match input_mean")
    eig, vec = np.linalg.eigh(c)
    root = vec * np.sqrt(np.maximum(eig, 0.0))
    rng = np.random.default_rng(int(seed))
    draws = x + rng.standard_normal((int(samples), x.size)) @ root.T
    outputs = np.asarray([_vector(forward_model(row), "forward_model output") for row in draws])
    if outputs.ndim != 2 or not np.all(np.isfinite(outputs)):
        raise ValueError("forward_model must return a fixed-length finite vector")
    return {
        "input_samples": draws, "output_samples": outputs,
        "output_mean": outputs.mean(axis=0),
        "external_output_covariance": np.cov(outputs, rowvar=False, ddof=1),
        "samples": int(samples), "seed": int(seed),
        "covariance_semantics": "external_input_uncertainty_only; distinct_from_measurement_noise_covariance",
        "method": "joint_monte_carlo",
    }

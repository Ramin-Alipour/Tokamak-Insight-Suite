"""Joint uncertainty utilities for Analysis 1 Thomson/CTS inference.

Preserves channel, physical-parameter, geometry and beam cross-covariances.
Uncertain inputs must be supplied explicitly; missing covariance is never inferred.
"""
from __future__ import annotations
from collections.abc import Callable, Mapping
import numpy as np

def _finite_matrix(value, name):
    a = np.asarray(value, dtype=float)
    if a.ndim != 2 or not a.size or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be a finite, non-empty 2-D matrix")
    return a

def validate_covariance(covariance, *, size=None, name="covariance", positive_definite=False, rtol=1e-9):
    """Validate a real symmetric PSD covariance without silently repairing it."""
    c = _finite_matrix(covariance, name)
    if c.shape[0] != c.shape[1] or (size is not None and c.shape != (size, size)):
        raise ValueError(f"{name} must be square" + (f" with shape ({size},{size})" if size is not None else ""))
    scale = max(float(np.max(np.abs(c))), np.finfo(float).tiny)
    if not np.allclose(c, c.T, rtol=rtol, atol=rtol * scale):
        raise ValueError(f"{name} must be symmetric")
    c = 0.5 * (c + c.T)
    eig = np.linalg.eigvalsh(c)
    if eig[0] < -rtol * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    if positive_definite and eig[0] <= rtol * scale:
        raise ValueError(f"{name} must be positive definite")
    return c

def assemble_joint_covariance(blocks, cross_blocks=None, *, name="joint_covariance"):
    """Assemble a joint covariance with explicit off-diagonal cross blocks."""
    if not isinstance(blocks, Mapping) or not blocks:
        raise ValueError("blocks must be a non-empty mapping")
    labels = list(blocks)
    checked, sizes = {}, []
    for label in labels:
        c = validate_covariance(blocks[label], name=f"blocks[{label!r}]")
        checked[label] = c
        sizes.append(c.shape[0])
    offsets = np.cumsum([0] + sizes)
    out = np.zeros((offsets[-1], offsets[-1]), dtype=float)
    for i, label in enumerate(labels):
        out[offsets[i]:offsets[i+1], offsets[i]:offsets[i+1]] = checked[label]
    supplied = set()
    for (left, right), raw in (cross_blocks or {}).items():
        if left not in checked or right not in checked or left == right:
            raise ValueError("cross-block labels must be distinct known labels")
        i, j = labels.index(left), labels.index(right)
        x = _finite_matrix(raw, f"cross_blocks[{left!r},{right!r}]")
        if x.shape != (sizes[i], sizes[j]):
            raise ValueError("cross-block shape does not match marginal blocks")
        out[offsets[i]:offsets[i+1], offsets[j]:offsets[j+1]] = x
        out[offsets[j]:offsets[j+1], offsets[i]:offsets[i+1]] = x.T
        supplied.add((left, right))
    out = validate_covariance(out, name=name)
    return {
        "covariance": out,
        "block_order": labels,
        "block_slices": {label: (int(offsets[i]), int(offsets[i+1])) for i, label in enumerate(labels)},
        "unspecified_cross_covariance_assumed_zero": [
            (labels[i], labels[j]) for i in range(len(labels)) for j in range(i+1, len(labels))
            if (labels[i], labels[j]) not in supplied and (labels[j], labels[i]) not in supplied
        ],
    }

def propagate_joint_covariance(jacobian, input_covariance, *, output_noise_covariance=None):
    """First-order propagation C_y = J C_x J.T + C_extra."""
    j = _finite_matrix(jacobian, "jacobian")
    cx = validate_covariance(input_covariance, size=j.shape[1], name="input_covariance")
    cy = j @ cx @ j.T
    if output_noise_covariance is not None:
        cy += validate_covariance(output_noise_covariance, size=j.shape[0], name="output_noise_covariance")
    return validate_covariance(cy, name="propagated_covariance")

def finite_difference_jacobian(model: Callable, parameters, *, relative_step=1e-5):
    """Central-difference Jacobian with parameter-scaled steps."""
    x = np.asarray(parameters, dtype=float)
    if x.ndim != 1 or not x.size or not np.all(np.isfinite(x)):
        raise ValueError("parameters must be a finite non-empty vector")
    if not np.isfinite(relative_step) or relative_step <= 0:
        raise ValueError("relative_step must be positive and finite")
    y0 = np.asarray(model(x), dtype=float)
    if y0.ndim != 1 or not np.all(np.isfinite(y0)):
        raise ValueError("model must return a finite one-dimensional vector")
    j = np.empty((y0.size, x.size), dtype=float)
    for k in range(x.size):
        h = relative_step * max(abs(x[k]), 1.0)
        xp, xm = x.copy(), x.copy()
        xp[k] += h
        xm[k] -= h
        yp, ym = np.asarray(model(xp), dtype=float), np.asarray(model(xm), dtype=float)
        if yp.shape != y0.shape or ym.shape != y0.shape or not np.all(np.isfinite(yp)) or not np.all(np.isfinite(ym)):
            raise ValueError("model output changed shape or became non-finite during differentiation")
        j[:, k] = (yp - ym) / (2.0 * h)
    return j

def propagate_reference_ensemble(reference_spectra, transform: Callable, *, n_draws=1000, seed=0):
    """Bootstrap reference realizations through a PCA/basis transform."""
    samples = _finite_matrix(reference_spectra, "reference_spectra")
    if samples.shape[0] < 3:
        raise ValueError("at least three reference realizations are required")
    if not callable(transform) or isinstance(n_draws, bool) or int(n_draws) < 100:
        raise ValueError("transform must be callable and n_draws must be at least 100")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, samples.shape[0], size=int(n_draws))
    values = []
    for index in indices:
        value = np.asarray(transform(samples[index]), dtype=float)
        if value.ndim != 1 or not np.all(np.isfinite(value)):
            raise ValueError("transform must return a finite one-dimensional vector")
        values.append(value)
    values = np.asarray(values)
    return {
        "draws": values, "mean": values.mean(axis=0),
        "covariance": validate_covariance(np.atleast_2d(np.cov(values, rowvar=False)), name="reference_ensemble_covariance"),
        "standard_deviation": values.std(axis=0, ddof=1),
        "method": "nonparametric_reference_realization_bootstrap",
        "n_draws": int(n_draws), "seed": int(seed),
        "reference_realizations": int(samples.shape[0]),
    }

def regularization_sensitivity(solver: Callable, strengths, *, profile_extractor: Callable | None = None):
    """Evaluate a reconstruction over a declared regularization grid."""
    lam = np.asarray(strengths, dtype=float)
    if lam.ndim != 1 or lam.size < 2 or not np.all(np.isfinite(lam)) or np.any(lam <= 0) or np.any(np.diff(lam) <= 0):
        raise ValueError("strengths must be strictly increasing positive finite values")
    results, profiles = [], []
    for value in lam:
        result = solver(float(value))
        p = np.asarray(profile_extractor(result) if profile_extractor else result, dtype=float)
        if p.ndim != 1 or not np.all(np.isfinite(p)):
            raise ValueError("each extracted profile must be a finite one-dimensional vector")
        results.append(result)
        profiles.append(p)
    if len({p.shape for p in profiles}) != 1:
        raise ValueError("profile shape must remain constant across regularization strengths")
    matrix = np.asarray(profiles)
    return {
        "regularization_strength": lam, "profiles": matrix,
        "profile_mean": matrix.mean(axis=0),
        "profile_standard_deviation": matrix.std(axis=0, ddof=1),
        "profile_covariance_across_regularization": np.atleast_2d(np.cov(matrix, rowvar=False)),
        "solver_results": results,
        "interpretation": "regularization-choice sensitivity; not measurement uncertainty",
    }

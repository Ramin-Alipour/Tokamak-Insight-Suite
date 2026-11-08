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


def _bounds(value, size, point):
    if value is None:
        return np.full(size, -np.inf), np.full(size, np.inf), "unbounded"
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.shape != (size, 2):
        raise ValueError("input_bounds must be a real (n_inputs, 2) array of lower/upper bounds")
    limits = raw.astype(float, copy=False)
    lower, upper = limits[:, 0], limits[:, 1]
    if np.any(np.isnan(limits)) or np.any(lower >= upper):
        raise ValueError("input_bounds must contain ordered lower < upper limits")
    if np.any(point < lower) or np.any(point > upper):
        raise ValueError("input_mean must lie within input_bounds")
    return lower, upper, "researcher_supplied"


def propagate_external_covariance(forward_model, input_mean, external_input_covariance, *,
                                   input_names=None, relative_step=None, input_bounds=None):
    """Propagate correlated external inputs with bound-aware numerical derivatives.

    Uses central differences in the interior and second-order one-sided
    differences at an active lower/upper bound. Fixed-width parameters are
    intentionally unsupported; use an input vector without fixed parameters.
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
    lower, upper, bounds_status = _bounds(input_bounds, x.size, x)
    y = _vector(forward_model(x.copy()), "forward_model output")
    jac = np.empty((y.size, x.size))
    derivative_methods = []
    for j in range(x.size):
        h = hscale * max(abs(float(x[j])), 1.0)
        plus_room, minus_room = upper[j] - x[j], x[j] - lower[j]
        use_central = plus_room >= h and minus_room >= h
        xp, xm = x.copy(), x.copy()
        if use_central:
            xp[j] += h
            xm[j] -= h
            yp = _vector(forward_model(xp), "forward_model output")
            ym = _vector(forward_model(xm), "forward_model output")
            if yp.shape != y.shape or ym.shape != y.shape:
                raise ValueError("forward_model output dimension must remain fixed")
            jac[:, j] = (yp - ym) / (2.0 * h)
            derivative_methods.append("central")
        elif plus_room > 0:
            step = min(h, plus_room / 2.0)
            xp1, xp2 = x.copy(), x.copy()
            xp1[j] += step
            xp2[j] += 2.0 * step
            y1 = _vector(forward_model(xp1), "forward_model output")
            y2 = _vector(forward_model(xp2), "forward_model output")
            if y1.shape != y.shape or y2.shape != y.shape:
                raise ValueError("forward_model output dimension must remain fixed")
            jac[:, j] = (-3.0 * y + 4.0 * y1 - y2) / (2.0 * step)
            derivative_methods.append("second_order_forward")
        elif minus_room > 0:
            step = min(h, minus_room / 2.0)
            xm1, xm2 = x.copy(), x.copy()
            xm1[j] -= step
            xm2[j] -= 2.0 * step
            y1 = _vector(forward_model(xm1), "forward_model output")
            y2 = _vector(forward_model(xm2), "forward_model output")
            if y1.shape != y.shape or y2.shape != y.shape:
                raise ValueError("forward_model output dimension must remain fixed")
            jac[:, j] = (3.0 * y - 4.0 * y1 + y2) / (2.0 * step)
            derivative_methods.append("second_order_backward")
        else:
            raise ValueError(f"input {names[j]} has no room for a finite-difference derivative within input_bounds")
    if not np.all(np.isfinite(jac)):
        raise ValueError("external-input Jacobian contains non-finite values")
    out = jac @ c @ jac.T
    return {
        "output_mean": y, "input_names": names, "jacobian": jac,
        "external_input_covariance": c,
        "external_output_covariance": (out + out.T) / 2.0,
        "input_bounds": np.column_stack((lower, upper)),
        "input_bounds_status": bounds_status,
        "finite_difference_methods": derivative_methods,
        "covariance_semantics": "external_input_uncertainty_only; distinct_from_measurement_noise_covariance",
        "method": "bound_aware_finite_difference_jacobian",
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

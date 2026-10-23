""""Extensible user-defined spectral models for Thomson-scattering analysis.

The registry fits user-supplied model callables; it does not interpret fitted
parameters as plasma quantities. Registration is explicit and never dynamically
imports external plugin files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class SpectralModelSpec:
    name: str
    function: Callable
    parameter_names: tuple[str, ...]
    initial_values: tuple[float, ...]
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]
    description: str


_CUSTOM_MODELS: dict[str, SpectralModelSpec] = {}


def register_spectral_model(
    name: str,
    function: Callable,
    *,
    parameter_names: Sequence[str],
    initial_values: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    description: str = "",
    replace: bool = False,
) -> SpectralModelSpec:
    """Register a model callable accepting function(wavelength_nm, *parameters)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string")
    key = name.strip()
    if not callable(function):
        raise TypeError("Model function must be callable")
    raw_names = tuple(parameter_names)
    if not raw_names or any(not isinstance(v, str) or not v.strip() for v in raw_names):
        raise ValueError("Parameter names must be non-empty strings")
    names = tuple(v.strip() for v in raw_names)
    if len(set(names)) != len(names):
        raise ValueError("Parameter names must be unique after trimming whitespace")
    initial = tuple(float(v) for v in initial_values)
    if len(initial) != len(names) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial values must be finite and match parameter names")

    if bounds is None:
        lower = tuple(-np.inf for _ in names)
        upper = tuple(np.inf for _ in names)
    else:
        if len(bounds) != 2:
            raise ValueError("bounds must be a (lower, upper) pair")
        lower = tuple(float(v) for v in bounds[0])
        upper = tuple(float(v) for v in bounds[1])
        if len(lower) != len(names) or len(upper) != len(names):
            raise ValueError("Bounds must match parameter names")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("Bounds cannot contain NaN")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("Each lower bound must be less than its upper bound")
        if any(v < lo or v > hi for v, lo, hi in zip(initial, lower, upper)):
            raise ValueError("Initial values must lie within bounds")

    if key in _CUSTOM_MODELS and not replace:
        raise ValueError(f"Spectral model {key!r} is already registered; use replace=True")
    spec = SpectralModelSpec(key, function, names, initial, lower, upper, str(description))
    _CUSTOM_MODELS[key] = spec
    return spec


def list_spectral_models() -> list[dict[str, object]]:
    """List metadata for registered custom models in this process."""
    return [
        {"name": spec.name, "parameter_names": list(spec.parameter_names),
         "description": spec.description}
        for spec in sorted(_CUSTOM_MODELS.values(), key=lambda item: item.name.casefold())
    ]


def unregister_spectral_model(name: str) -> bool:
    """Unregister a custom model from this process."""
    return _CUSTOM_MODELS.pop(name, None) is not None


def fit_custom_spectral_model(
    wavelength_nm,
    signal,
    *,
    model_name: str,
    signal_uncertainty=None,
    absolute_sigma: bool = True,
) -> dict[str, object]:
    """Fit a registered model; no plasma-parameter interpretation is inferred."""
    if model_name not in _CUSTOM_MODELS:
        raise ValueError(f"Unknown custom spectral model {model_name!r}; register it first")
    spec = _CUSTOM_MODELS[model_name]
    x = np.asarray(wavelength_nm, dtype=float)
    y = np.asarray(signal, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("Wavelength and signal must be 1-D arrays of equal length")
    if x.size <= len(spec.parameter_names):
        raise ValueError("Need more spectral samples than fitted parameters")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Wavelength and signal must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("Wavelength axis must be strictly increasing")
    sigma = None
    if signal_uncertainty is not None:
        sigma = np.asarray(signal_uncertainty, dtype=float)
        if sigma.ndim != 1 or sigma.size != y.size:
            raise ValueError("Signal uncertainty must match the signal shape")
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("Signal uncertainties must be finite and positive")

    def checked_model(axis, *parameters):
        prediction = np.asarray(spec.function(axis, *parameters), dtype=float)
        if prediction.shape != axis.shape or not np.all(np.isfinite(prediction)):
            raise ValueError("Custom model must return finite values matching wavelength shape")
        return prediction

    parameters, covariance = curve_fit(
        checked_model, x, y, p0=spec.initial_values,
        bounds=(spec.lower_bounds, spec.upper_bounds), sigma=sigma,
        absolute_sigma=bool(absolute_sigma) if sigma is not None else False,
        maxfev=50_000,
    )
    fitted = checked_model(x, *parameters)
    residual = y - fitted
    dof = int(y.size - parameters.size)
    chi_square = None if sigma is None else float(np.sum((residual / sigma) ** 2))
    return {
        "model_name": spec.name,
        "parameter_names": list(spec.parameter_names),
        "parameters": {n: float(v) for n, v in zip(spec.parameter_names, parameters)},
        "parameter_uncertainties": {
            n: float(v) for n, v in zip(
                spec.parameter_names, np.sqrt(np.maximum(np.diag(covariance), 0.0)))
        },
        "covariance": covariance,
        "wavelength_nm": x.copy(),
        "fitted_spectrum": fitted,
        "residual": residual,
        "degrees_of_freedom": dof,
        "chi_square": chi_square,
        "reduced_chi_square": None if chi_square is None or dof <= 0 else float(chi_square / dof),
        "interpretation_note": (
            "Generic fitted parameters only. Electron temperature, density, and other "
            "physical quantities are not inferred automatically."
        ),
    }

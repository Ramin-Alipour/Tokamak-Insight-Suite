"""Extensible user-defined spectral models for Thomson-scattering analysis.

The registry fits user-supplied model callables; it does not interpret fitted
parameters as plasma quantities. Registration is explicit and never dynamically
imports external plugin files.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Mapping

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
    physical_parameter_mapper: Callable | None = None
    physical_contract: dict | None = None


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
    physical_parameter_mapper: Callable | None = None,
    physical_contract: Mapping | None = None,
) -> SpectralModelSpec:
    """Register a callable accepting function(wavelength_nm, *parameters)."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Model name must be a non-empty string")
    key = name.strip()
    if not callable(function):
        raise TypeError("Model function must be callable")
    if isinstance(parameter_names, (str, bytes)):
        raise ValueError("parameter_names must be a sequence of distinct names, not a string")
    try:
        raw_names = tuple(parameter_names)
    except TypeError as exc:
        raise ValueError("parameter_names must be a sequence of names") from exc
    if not raw_names or any(not isinstance(v, str) or not v.strip() for v in raw_names):
        raise ValueError("Parameter names must be non-empty strings")
    names = tuple(v.strip() for v in raw_names)
    if len(set(names)) != len(names):
        raise ValueError("Parameter names must be unique after trimming whitespace")
    try:
        initial = tuple(float(v) for v in initial_values)
    except (TypeError, ValueError) as exc:
        raise ValueError("Initial values must be numeric and match parameter names") from exc
    if len(initial) != len(names) or not np.all(np.isfinite(initial)):
        raise ValueError("Initial values must be finite and match parameter names")

    if bounds is None:
        lower = tuple(-np.inf for _ in names)
        upper = tuple(np.inf for _ in names)
    else:
        try:
            if len(bounds) != 2:
                raise ValueError("bounds must be a (lower, upper) pair")
            lower = tuple(float(v) for v in bounds[0])
            upper = tuple(float(v) for v in bounds[1])
        except (TypeError, ValueError, IndexError) as exc:
            if isinstance(exc, ValueError) and str(exc) == "bounds must be a (lower, upper) pair":
                raise
            raise ValueError("bounds must contain numeric lower and upper sequences") from exc
        if len(lower) != len(names) or len(upper) != len(names):
            raise ValueError("Bounds must match parameter names")
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("Bounds cannot contain NaN")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("Each lower bound must be less than its upper bound")
        if any(v < lo or v > hi for v, lo, hi in zip(initial, lower, upper)):
            raise ValueError("Initial values must lie within bounds")

    if physical_parameter_mapper is not None and not callable(physical_parameter_mapper):
        raise TypeError("physical_parameter_mapper must be callable")
    contract = None
    if physical_parameter_mapper is not None:
        if not isinstance(physical_contract, Mapping):
            raise ValueError("A physical_contract mapping is required whenever a physical_parameter_mapper is registered")
        required_contract = {"output_units", "formula_references", "assumptions", "limitations", "parameter_provenance"}
        missing = required_contract - set(physical_contract)
        if missing:
            raise ValueError(f"physical_contract is missing required fields: {sorted(missing)}")
        if not all(physical_contract[k] for k in required_contract):
            raise ValueError("Every physical_contract field must be non-empty")
        contract = dict(physical_contract)
    elif physical_contract is not None:
        raise ValueError("physical_contract cannot be supplied without a physical_parameter_mapper")
    if key in _CUSTOM_MODELS and not replace:
        raise ValueError(f"Spectral model {key!r} is already registered; use replace=True")
    spec = SpectralModelSpec(key, function, names, initial, lower, upper, str(description),
                             physical_parameter_mapper, contract)
    _CUSTOM_MODELS[key] = spec
    return spec


def list_spectral_models() -> list[dict[str, object]]:
    """List metadata for registered custom models in this process."""
    return [
        {
            "name": spec.name,
            "parameter_names": list(spec.parameter_names),
            "description": spec.description,
        }
        for spec in sorted(_CUSTOM_MODELS.values(), key=lambda item: item.name.casefold())
    ]


def unregister_spectral_model(name: str) -> bool:
    """Unregister a custom model from this process."""
    if not isinstance(name, str) or not name.strip():
        return False
    return _CUSTOM_MODELS.pop(name.strip(), None) is not None


def fit_custom_spectral_model(
    wavelength_nm,
    signal,
    *,
    model_name: str,
    signal_uncertainty=None,
    absolute_sigma: bool = True,
) -> dict[str, object]:
    """Fit a registered model; no plasma-parameter interpretation is inferred."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty registered model name")
    key = model_name.strip()
    if key not in _CUSTOM_MODELS:
        raise ValueError(f"Unknown custom spectral model {key!r}; register it first")
    if not isinstance(absolute_sigma, (bool, np.bool_)):
        raise ValueError("absolute_sigma must be boolean")
    spec = _CUSTOM_MODELS[key]
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
    covariance_diagonal = np.diag(covariance)
    parameter_uncertainties = {
        name: float(np.sqrt(max(value, 0.0))) if np.isfinite(value) else float("inf")
        for name, value in zip(spec.parameter_names, covariance_diagonal)
    }
    parameter_map = {n: float(v) for n, v in zip(spec.parameter_names, parameters)}
    physical_inference = None
    if spec.physical_parameter_mapper is not None:
        mapped = spec.physical_parameter_mapper(dict(parameter_map))
        if not isinstance(mapped, Mapping) or not mapped:
            raise ValueError("physical_parameter_mapper must return a non-empty mapping")
        numeric_outputs={k:float(v) for k,v in mapped.items()
                        if isinstance(v,(int,float,np.integer,np.floating)) and np.isfinite(v)}
        physical_uncertainties={}
        physical_covariance=None
        output_order=list(numeric_outputs)
        if output_order:
            mapper_jacobian=np.zeros((len(output_order),len(parameters)))
            for j in range(len(parameters)):
                step=np.cbrt(np.finfo(float).eps)*max(abs(float(parameters[j])),1.0)
                plus=np.asarray(parameters,dtype=float).copy()
                minus=np.asarray(parameters,dtype=float).copy()
                plus[j]+=step; minus[j]-=step
                plus_map={n:float(v) for n,v in zip(spec.parameter_names,plus)}
                minus_map={n:float(v) for n,v in zip(spec.parameter_names,minus)}
                plus_values=spec.physical_parameter_mapper(plus_map)
                minus_values=spec.physical_parameter_mapper(minus_map)
                for row,key in enumerate(output_order):
                    mapper_jacobian[row,j]=(float(plus_values[key])-float(minus_values[key]))/(2.0*step)
            physical_covariance=mapper_jacobian@covariance@mapper_jacobian.T
            physical_uncertainties={key:float(np.sqrt(max(physical_covariance[i,i],0.0)))
                                    for i,key in enumerate(output_order)}
        physical_inference = {
            "values": dict(mapped),
            "uncertainties": physical_uncertainties,
            "covariance": physical_covariance,
            "covariance_output_order": output_order,
            "contract": dict(spec.physical_contract),
            "status": "model_contract_mapped_with_local_covariance_not_independently_validated",
        }
    return {
        "model_name": spec.name,
        "parameter_names": list(spec.parameter_names),
        "parameters": parameter_map,
        "physical_inference": physical_inference,
        "parameter_uncertainties": parameter_uncertainties,
        "covariance": covariance,
        "wavelength_nm": x.copy(),
        "fitted_spectrum": fitted,
        "residual": residual,
        "degrees_of_freedom": dof,
        "chi_square": chi_square,
        "reduced_chi_square": None if chi_square is None or dof <= 0 else float(chi_square / dof),
        "interpretation_note": (
            "Physical quantities are emitted only through the explicitly registered mapper and physical contract; this is not independent physical validation."
            if physical_inference is not None else
            "Generic fitted parameters only. Electron temperature, density, and other physical quantities are not inferred automatically."
        ),
    }

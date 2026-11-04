"""Shared, strict contracts for Thomson/CTS inference paths.

These validators do not add physical models. They prevent implicit background,
instrument-response, covariance, regularization, and identifiability assumptions
from being silently introduced at API boundaries.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np


_BACKGROUND_STATES = {
    "assumed_zero",
    "reference_pca",
    "measured_spectrum",
    "uncharacterized_constant_offset",
}


def resolve_physical_background_state(
    background_state: str | None,
    *,
    reference_spectra=None,
    measured_background_spectrum=None,
) -> dict:
    """Resolve background semantics without interpreting omission as zero."""
    if background_state is None:
        state = "uncharacterized_constant_offset"
    elif not isinstance(background_state, str) or background_state not in _BACKGROUND_STATES:
        raise ValueError(f"background_state must be one of {sorted(_BACKGROUND_STATES)}")
    else:
        state = background_state

    if state == "assumed_zero":
        if reference_spectra is not None or measured_background_spectrum is not None:
            raise ValueError("assumed_zero conflicts with supplied background measurements")
        return {
            "background_state": state,
            "physical_background_zero": True,
            "background_identifiability": "fixed_by_explicit_user_assumption",
        }
    if state == "reference_pca":
        if reference_spectra is None:
            raise ValueError("reference_pca requires reference_spectra")
        x = _finite_real_array(reference_spectra, "reference_spectra", ndim=2)
        if min(x.shape) < 2:
            raise ValueError("reference_spectra must contain at least two spectra and two channels")
        return {
            "background_state": state,
            "physical_background_zero": None,
            "background_identifiability": "conditional_on_supplied_reference_basis",
            "reference_spectra_shape": tuple(int(v) for v in x.shape),
        }
    if state == "measured_spectrum":
        if measured_background_spectrum is None:
            raise ValueError("measured_spectrum requires measured_background_spectrum")
        x = _finite_real_array(measured_background_spectrum, "measured_background_spectrum", ndim=1)
        if x.size < 2:
            raise ValueError("measured_background_spectrum must contain at least two channels")
        return {
            "background_state": state,
            "physical_background_zero": None,
            "background_identifiability": "conditioned_on_independent_measured_background",
            "background_channel_count": int(x.size),
        }
    if reference_spectra is not None or measured_background_spectrum is not None:
        raise ValueError("uncharacterized_constant_offset cannot silently consume background measurements")
    return {
        "background_state": state,
        "physical_background_zero": None,
        "background_identifiability": "not_identified_as_physical_zero",
    }


def _finite_real_array(value, name: str, *, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or (ndim is not None and raw.ndim != ndim):
        raise ValueError(f"{name} must contain real numeric values" + (f" with ndim={ndim}" if ndim is not None else ""))
    out = raw.astype(float, copy=False)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must be finite and non-empty")
    return out


def validate_spectral_response_operator(response, channel_count: int, *, atol: float = 1e-8) -> np.ndarray:
    """Validate a measured column-stochastic response; never renormalize it."""
    if isinstance(channel_count, bool) or not isinstance(channel_count, (int, np.integer)) or channel_count < 2:
        raise ValueError("channel_count must be an integer >= 2")
    r = _finite_real_array(response, "instrument_response", ndim=2)
    if r.shape != (channel_count, channel_count):
        raise ValueError("instrument_response must be square and match the measured spectral grid")
    if np.any(r < 0):
        raise ValueError("instrument_response must be non-negative")
    sums = r.sum(axis=0)
    if np.any(sums <= 0) or not np.allclose(sums, 1.0, rtol=0.0, atol=atol):
        raise ValueError("measured instrument_response columns must already sum to one; renormalization is forbidden")
    return r.copy()


def _symmetric_psd(value, name: str, n: int) -> np.ndarray:
    x = _finite_real_array(value, name, ndim=2)
    if x.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n})")
    scale = max(float(np.max(np.abs(x))), np.finfo(float).tiny)
    if not np.allclose(x, x.T, rtol=1e-9, atol=1e-11 * scale):
        raise ValueError(f"{name} must be symmetric")
    x = 0.5 * (x + x.T)
    eigenvalues = np.linalg.eigvalsh(x)
    eig_scale = max(float(np.max(np.abs(eigenvalues))), np.finfo(float).tiny)
    if float(eigenvalues[0]) < -1e-10 * eig_scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return x


def compose_spectral_covariance(
    channel_count: int,
    *,
    measurement_covariance=None,
    measurement_sigma=None,
    model_discrepancy_covariance=None,
) -> dict:
    """Combine independent measurement and discrepancy covariance exactly once."""
    if (measurement_covariance is None) == (measurement_sigma is None):
        raise ValueError("supply exactly one of measurement_covariance or measurement_sigma")
    if measurement_covariance is not None:
        measurement = _symmetric_psd(measurement_covariance, "measurement_covariance", channel_count)
    else:
        sigma = _finite_real_array(measurement_sigma, "measurement_sigma", ndim=1)
        if sigma.shape != (channel_count,) or np.any(sigma <= 0):
            raise ValueError("measurement_sigma must be positive and match channel_count")
        measurement = np.diag(sigma**2)

    discrepancy = None
    total = measurement.copy()
    if model_discrepancy_covariance is not None:
        discrepancy = _symmetric_psd(
            model_discrepancy_covariance, "model_discrepancy_covariance", channel_count
        )
        total += discrepancy
    try:
        np.linalg.cholesky(total)
    except np.linalg.LinAlgError as exc:
        raise ValueError("combined spectral covariance must be positive definite; no silent jitter is applied") from exc
    return {
        "measurement_covariance": measurement,
        "model_discrepancy_covariance": discrepancy,
        "total_covariance": 0.5 * (total + total.T),
        "covariance_components_independent": True,
        "model_discrepancy_included": discrepancy is not None,
    }


def validate_profile_inversion_contract(
    profile_bounds_relative,
    regularization_strength,
    active_profiles: Sequence[str],
    *,
    regularization_scope: str,
) -> dict:
    """Require explicit support and per-profile regularization for profile fits."""
    if not isinstance(active_profiles, Sequence) or isinstance(active_profiles, (str, bytes)) or not active_profiles:
        raise ValueError("active_profiles must be a non-empty sequence")
    names = tuple(str(x) for x in active_profiles)
    if any(not x for x in names) or len(set(names)) != len(names):
        raise ValueError("active_profiles must contain unique non-empty names")
    bounds = _finite_real_array(profile_bounds_relative, "profile_bounds_relative", ndim=1)
    if bounds.shape != (2,) or bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("profile_bounds_relative must be an explicit positive (lower, upper) pair")
    if not isinstance(regularization_scope, str) or not regularization_scope.strip():
        raise ValueError("regularization_scope must explicitly describe which profile components are regularized")
    if isinstance(regularization_strength, Mapping):
        if set(regularization_strength) != set(names):
            raise ValueError("regularization_strength mapping must define exactly every active profile")
        strengths = {name: regularization_strength[name] for name in names}
    else:
        strengths = {name: regularization_strength for name in names}
    parsed = {}
    for name, value in strengths.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise ValueError(f"regularization strength for {name} must be an explicit positive real scalar")
        strength = float(value)
        if not np.isfinite(strength) or strength < 0:
            raise ValueError(f"regularization strength for {name} must be finite and >= 0")
        parsed[name] = strength
    if not any(value > 0 for value in parsed.values()):
        raise ValueError("at least one active profile must have a positive regularization strength")
    return {
        "profile_bounds_relative": (float(bounds[0]), float(bounds[1])),
        "regularization_strength": parsed,
        "regularization_scope": regularization_scope.strip(),
        "prior_semantics": "explicit_smoothness_penalty_not_measurement_information",
    }


def diagnose_local_identifiability(jacobian, parameter_names: Sequence[str], *, rtol: float | None = None) -> dict:
    """SVD-based local identifiability using column-scaled Jacobian geometry."""
    j = _finite_real_array(jacobian, "jacobian", ndim=2)
    names = tuple(str(x) for x in parameter_names)
    if j.shape[1] != len(names) or not names or any(not x for x in names):
        raise ValueError("parameter_names must name every Jacobian column")
    if len(set(names)) != len(names):
        raise ValueError("parameter_names must be unique")
    norms = np.linalg.norm(j, axis=0)
    active = norms > np.finfo(float).tiny
    scaled = np.zeros_like(j)
    scaled[:, active] = j[:, active] / norms[active]
    _, singular, vt = np.linalg.svd(scaled, full_matrices=True)
    if rtol is None:
        tolerance = max(j.shape) * np.finfo(float).eps
    else:
        if isinstance(rtol, (bool, np.bool_)) or not isinstance(rtol, (int, float, np.integer, np.floating)) or not np.isfinite(rtol) or rtol <= 0:
            raise ValueError("rtol must be finite and > 0")
        tolerance = float(rtol)
    threshold = tolerance * (float(singular[0]) if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > threshold))
    null_basis = vt[rank:, :]
    condition = float(singular[0] / singular[rank - 1]) if rank and rank == len(names) else float("inf")
    status = "locally_identifiable" if rank == len(names) else "structurally_or_practically_rank_deficient"
    if rank == len(names) and condition > 1e8:
        status = "full_rank_but_ill_conditioned"
    return {
        "status": status,
        "jacobian_rank": rank,
        "parameter_count": len(names),
        "singular_values_column_scaled": singular,
        "condition_number_column_scaled": condition,
        "unconstrained_parameter_directions_scaled": null_basis,
        "zero_sensitivity_parameters": [names[i] for i in np.flatnonzero(~active)],
        "rank_tolerance_relative": tolerance,
        "scope": "local_jacobian_diagnostic_not_global_identifiability_or_posterior_inference",
    }

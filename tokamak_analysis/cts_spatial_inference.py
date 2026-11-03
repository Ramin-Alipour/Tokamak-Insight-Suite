"""Profile-varying Full-EM CTS inverse problem over a resolved scattering volume.

Positive node-wise relative profiles are fitted on top of explicit reference
plasma parameters. Every optimizer evaluation calls the magnetized Full-EM
kinetic forward solver at every measured frequency and scattering-volume
quadrature point; no spatially averaged surrogate is substituted.
"""
from __future__ import annotations
from collections.abc import Mapping
import numpy as np
from scipy.optimize import least_squares
from .cts_full_em import full_em_cts_spectral_density, _apply_frequency_response


def _vector(value, name, positive=False):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf" or raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"{name} must be a non-empty real vector")
    out = raw.astype(float, copy=False)
    if not np.all(np.isfinite(out)) or (positive and np.any(out <= 0)):
        raise ValueError(f"{name} must contain finite {'positive ' if positive else ''}values")
    return out


def _covariance(value, n):
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError("spectral_covariance must be real")
    cov = raw.astype(float, copy=False)
    if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
        raise ValueError("spectral_covariance must be finite and match the measured spectrum")
    scale = max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    if not np.allclose(cov, cov.T, rtol=1e-9, atol=1e-11 * scale):
        raise ValueError("spectral_covariance must be symmetric")
    cov = 0.5 * (cov + cov.T)
    eig = np.linalg.eigvalsh(cov)
    if eig[0] <= np.finfo(float).eps * n * max(float(eig[-1]), np.finfo(float).tiny):
        raise ValueError("spectral_covariance must be positive definite; no silent regularization is applied")
    return cov


def _second_difference(n):
    if n < 3:
        raise ValueError("at least three scattering-volume nodes are required")
    return np.diff(np.eye(n), n=2, axis=0)


def _geometry_nodes(geometry, n_frequency):
    if not isinstance(geometry, Mapping):
        raise ValueError("scattering_geometry must be a mapping")
    records = geometry.get("frequency_resolved_geometry")
    if not isinstance(records, (list, tuple)) or len(records) != n_frequency:
        raise ValueError("frequency_resolved_geometry must contain one record per measured bin")
    node_ids = None
    for fi, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"frequency_resolved_geometry[{fi}] must be a mapping")
        samples = record.get("scattering_samples")
        if not isinstance(samples, (list, tuple)) or len(samples) < 3:
            raise ValueError(f"frequency_resolved_geometry[{fi}] needs at least three scattering samples")
        ids = [str(s.get("sample_id", "")) for s in samples if isinstance(s, Mapping)]
        if len(ids) != len(samples) or any(not x for x in ids) or len(set(ids)) != len(ids):
            raise ValueError("each scattering sample needs a unique non-empty sample_id")
        weights = np.asarray([s.get("sample_weight", np.nan) for s in samples], dtype=float)
        if np.any(~np.isfinite(weights)) or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError(f"scattering sample weights at frequency index {fi} must be non-negative and sum to one")
        if node_ids is None:
            node_ids = ids
        elif set(ids) != set(node_ids):
            raise ValueError("all frequency bins must use the same scattering sample_id set")
        overlap = float(record.get("beam_overlap_relative", np.nan))
        if not np.isfinite(overlap) or overlap <= 0:
            raise ValueError("beam_overlap_relative must be positive and finite")
    return node_ids


def analyze_full_em_cts_spatial_profiles(
    frequency_shift_Hz, measured_spectrum, *, probe_frequency_Hz,
    scattering_geometry, forward_model_kwargs, spectral_covariance=None,
    signal_uncertainty=None, instrument_response=None,
    fixed_background_spectrum=None, fit_electron_density_profile=True,
    fit_electron_temperature_profile=True, fit_ion_temperature_profile=False,
    initial_profiles=None, regularization_matrix=None,
    regularization_strength=1.0, regularization_sensitivity_strengths=None,
    profile_bounds_relative=(0.05, 20.0), max_nfev=300,
):
    """Infer node-wise relative profiles using repeated local Full-EM solves.

    forward_model_kwargs must provide all physical Full-EM inputs including
    reference density, electron/ion temperatures, species, fractions, field
    and polarization. Exactly one spectral covariance representation is
    required. A supplied response matrix is applied once after volume integration.
    """
    f = _vector(frequency_shift_Hz, "frequency_shift_Hz")
    y = _vector(measured_spectrum, "measured_spectrum")
    if f.shape != y.shape or f.size < 8 or np.any(np.diff(f) <= 0):
        raise ValueError("frequency and spectrum must be matching vectors with >=8 increasing bins")
    if (spectral_covariance is None) == (signal_uncertainty is None):
        raise ValueError("supply exactly one of spectral_covariance or signal_uncertainty")
    if spectral_covariance is None:
        sigma = _vector(signal_uncertainty, "signal_uncertainty", positive=True)
        if sigma.shape != y.shape:
            raise ValueError("signal_uncertainty must match measured_spectrum")
        covariance = np.diag(sigma**2)
    else:
        covariance = _covariance(spectral_covariance, y.size)
    chol = np.linalg.cholesky(covariance)
    node_ids = _geometry_nodes(scattering_geometry, f.size)
    n_nodes = len(node_ids)
    flags = {
        "electron_density_relative": bool(fit_electron_density_profile),
        "electron_temperature_relative": bool(fit_electron_temperature_profile),
        "ion_temperature_relative": bool(fit_ion_temperature_profile),
    }
    active = [name for name, enabled in flags.items() if enabled]
    if not active:
        raise ValueError("select at least one spatial profile for inversion")
    if not isinstance(forward_model_kwargs, Mapping):
        raise ValueError("forward_model_kwargs must be a mapping")
    required = {"electron_density_m3", "electron_temperature_eV", "ion_temperature_eV",
                "ion_species", "ion_fractions", "magnetic_field_T"}
    missing = sorted(required - set(forward_model_kwargs))
    if missing:
        raise ValueError(f"missing required Full-EM inputs: {missing}")
    lower, upper = map(float, profile_bounds_relative)
    if not np.isfinite(lower + upper) or lower <= 0 or upper <= lower:
        raise ValueError("profile bounds must be finite positive (lower, upper)")
    if isinstance(regularization_strength, Mapping):
        lambdas = {name: float(regularization_strength.get(name, 0.0)) for name in active}
    else:
        lambdas = {name: float(regularization_strength) for name in active}
    if any(not np.isfinite(v) or v < 0 for v in lambdas.values()) or not any(v > 0 for v in lambdas.values()):
        raise ValueError("regularization strengths must be finite, non-negative, and at least one positive")
    penalty = _second_difference(n_nodes) if regularization_matrix is None else np.asarray(regularization_matrix, dtype=float)
    if penalty.ndim != 2 or penalty.shape[1] != n_nodes or not np.all(np.isfinite(penalty)):
        raise ValueError("regularization_matrix must be finite with one column per spatial node")
    response = np.eye(f.size) if instrument_response is None else np.asarray(instrument_response, dtype=float)
    if response.shape != (f.size, f.size) or not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("instrument_response must be a finite non-negative square matrix")
    background = np.zeros_like(y) if fixed_background_spectrum is None else _vector(fixed_background_spectrum, "fixed_background_spectrum")
    if background.shape != y.shape:
        raise ValueError("fixed_background_spectrum must match measured_spectrum")
    initial = initial_profiles or {}
    x_parts = []
    for name in active:
        values = np.asarray(initial.get(name, np.ones(n_nodes)), dtype=float)
        if values.ndim == 0:
            values = np.full(n_nodes, float(values))
        if values.shape != (n_nodes,) or np.any(~np.isfinite(values)) or np.any(values <= 0):
            raise ValueError(f"initial_profiles[{name}] must be a positive scalar or node vector")
        if np.any(values < lower) or np.any(values > upper):
            raise ValueError(f"initial_profiles[{name}] must lie within profile_bounds_relative")
        x_parts.append(np.log(values))
    x0 = np.concatenate(x_parts)
    lo, hi = np.full(x0.size, np.log(lower)), np.full(x0.size, np.log(upper))
    base = dict(forward_model_kwargs)
    records = scattering_geometry["frequency_resolved_geometry"]
    node_index = {node_id: i for i, node_id in enumerate(node_ids)}

    def unpack(x):
        return {name: np.exp(x[k*n_nodes:(k+1)*n_nodes]) for k, name in enumerate(active)}

    def forward(x):
        profiles = unpack(x)
        local_geometry = {"frequency_resolved_geometry": []}
        for record in records:
            samples = []
            for sample in record["scattering_samples"]:
                item = dict(sample)
                j = node_index[str(item["sample_id"])]
                for name in active:
                    item[name] = float(profiles[name][j])
                samples.append(item)
            local_geometry["frequency_resolved_geometry"].append({
                "beam_overlap_relative": float(record["beam_overlap_relative"]),
                "scattering_samples": samples,
            })
        local = full_em_cts_spectral_density(f, probe_frequency_Hz, local_geometry, **base)
        return _apply_frequency_response(response, np.asarray(local["spectral_density"], dtype=float), f)

    def residual(x):
        spectral = np.linalg.solve(chol, forward(x) + background - y)
        p = unpack(x)
        prior = [np.sqrt(lambdas[name]) * (penalty @ np.log(p[name]))
                 for name in active if lambdas[name] > 0]
        return np.concatenate([spectral, *prior]) if prior else spectral

    fit = least_squares(residual, x0, bounds=(lo, hi), method="trf",
                        max_nfev=int(max_nfev), x_scale="jac")
    profiles = unpack(fit.x)
    predicted = forward(fit.x)
    residual_spectrum = y - predicted - background
    singular = np.linalg.svd(np.asarray(fit.jac), compute_uv=False)
    tol = np.finfo(float).eps * max(fit.jac.shape) * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tol))
    identifiable = rank == fit.x.size
    log_cov = np.linalg.pinv(fit.jac.T @ fit.jac, rcond=1e-12) if identifiable else None
    scale = np.diag(np.exp(fit.x))
    profile_cov = scale @ log_cov @ scale if log_cov is not None else None
    whitened = np.linalg.solve(chol, residual_spectrum)
    chi2 = float(whitened @ whitened)
    dof = max(y.size - rank, 1)
    sensitivity = None
    if regularization_sensitivity_strengths is not None:
        grid = _vector(regularization_sensitivity_strengths, "regularization_sensitivity_strengths", positive=True)
        if grid.size < 2 or np.any(np.diff(grid) <= 0):
            raise ValueError("regularization_sensitivity_strengths must be strictly increasing with at least two values")
        repeated = []
        for strength in grid:
            repeated.append(analyze_full_em_cts_spatial_profiles(
                frequency_shift_Hz, measured_spectrum,
                probe_frequency_Hz=probe_frequency_Hz,
                scattering_geometry=scattering_geometry,
                forward_model_kwargs=forward_model_kwargs,
                spectral_covariance=covariance,
                instrument_response=response,
                fixed_background_spectrum=background,
                fit_electron_density_profile=fit_electron_density_profile,
                fit_electron_temperature_profile=fit_electron_temperature_profile,
                fit_ion_temperature_profile=fit_ion_temperature_profile,
                initial_profiles=initial,
                regularization_matrix=penalty,
                regularization_strength=float(strength),
                regularization_sensitivity_strengths=None,
                profile_bounds_relative=profile_bounds_relative,
                max_nfev=max_nfev,
            ))
        sensitivity = {
            "regularization_strengths": grid.copy(),
            "profiles_relative_to_reference": {
                name: np.stack([item["profiles_relative_to_reference"][name] for item in repeated])
                for name in active
            },
            "profile_standard_deviation_across_regularization": {
                name: np.std(np.stack([item["profiles_relative_to_reference"][name] for item in repeated]), axis=0, ddof=1)
                for name in active
            },
            "interpretation": "regularization-choice sensitivity; not measurement uncertainty",
            "measurement_covariance_unchanged": True,
        }
    return {
        "model": "spatially_varying_magnetized_full_electromagnetic_CTS",
        "optimizer_success": bool(fit.success), "optimizer_message": str(fit.message),
        "node_ids": node_ids, "profile_parameter_names": active,
        "parameter_order": [{"parameter": name, "sample_id": node_id}
                            for name in active for node_id in node_ids],
        "profiles_relative_to_reference": {k: v.copy() for k, v in profiles.items()},
        "profile_covariance_relative": profile_cov,
        "log_profile_covariance": log_cov,
        "profile_identifiability": {
            "jacobian_rank": rank, "parameter_count": int(fit.x.size),
            "locally_identifiable": bool(identifiable), "singular_values": singular,
            "condition_number": float(singular[0]/singular[-1]) if singular.size and singular[-1] > 0 else float("inf"),
        },
        "frequency_shift_Hz": f.copy(), "measured_spectrum": y.copy(),
        "fitted_spectrum": predicted + background, "physical_signal_spectrum": predicted,
        "fixed_background_spectrum": background, "residual": residual_spectrum,
        "spectral_covariance": covariance.copy(), "chi_squared": chi2,
        "degrees_of_freedom": int(dof), "reduced_chi_squared": chi2/dof,
        "regularization_matrix": penalty.copy(), "regularization_strength": dict(lambdas),
        "regularization_is_measurement_covariance": False,
        "regularization_sensitivity": sensitivity,
        "reference_bulk_parameters": {k: base[k] for k in ("electron_density_m3", "electron_temperature_eV", "ion_temperature_eV")},
        "forward_model_evaluations_use_local_full_em_kinetics": True,
        "limitations": [
            "Profiles are relative to supplied reference density and temperatures.",
            "Spatial nodes, ray geometry, beam weights and polarizations are fixed supplied inputs.",
            "Profile covariance is omitted when the augmented Jacobian is rank deficient.",
            "No conserving collision operator, non-paraxial propagation or experimental validation is implied.",
        ],
    }

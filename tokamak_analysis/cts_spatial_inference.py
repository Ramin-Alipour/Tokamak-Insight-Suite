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
    signal_uncertainty=None, model_discrepancy_covariance=None,
    instrument_response=None,
    fixed_background_spectrum=None, background_mode=None, fit_electron_density_profile=True,
    fit_electron_temperature_profile=True, fit_ion_temperature_profile=False,
    initial_profiles=None, regularization_matrix=None,
    regularization_strength=None, regularization_sensitivity_strengths=None,
    profile_bounds_relative=None, max_nfev=300,
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
        measurement_covariance = np.diag(sigma**2)
    else:
        measurement_covariance = _covariance(spectral_covariance, y.size)

    discrepancy_covariance = None
    if model_discrepancy_covariance is not None:
        raw_discrepancy = np.asarray(model_discrepancy_covariance)
        if raw_discrepancy.dtype.kind not in "iuf":
            raise ValueError("model_discrepancy_covariance must be real")
        discrepancy_covariance = raw_discrepancy.astype(float, copy=False)
        if (discrepancy_covariance.shape != (y.size, y.size)
                or not np.all(np.isfinite(discrepancy_covariance))):
            raise ValueError("model_discrepancy_covariance must be finite and match the measured spectrum")
        discrepancy_scale = max(
            float(np.max(np.abs(discrepancy_covariance))), np.finfo(float).tiny
        )
        if not np.allclose(
            discrepancy_covariance, discrepancy_covariance.T,
            rtol=1e-9, atol=1e-11 * discrepancy_scale,
        ):
            raise ValueError("model_discrepancy_covariance must be symmetric")
        discrepancy_covariance = 0.5 * (
            discrepancy_covariance + discrepancy_covariance.T
        )
        discrepancy_eigenvalues = np.linalg.eigvalsh(discrepancy_covariance)
        eigen_scale = max(
            float(np.max(np.abs(discrepancy_eigenvalues))), np.finfo(float).tiny
        )
        if float(discrepancy_eigenvalues[0]) < -1e-10 * eigen_scale:
            raise ValueError("model_discrepancy_covariance must be positive semidefinite")

    covariance = measurement_covariance.copy()
    if discrepancy_covariance is not None:
        covariance += discrepancy_covariance
    try:
        chol = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "measurement covariance plus model discrepancy must be positive definite; "
            "no silent regularization is applied"
        ) from exc
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
    if profile_bounds_relative is None:
        raise ValueError("profile_bounds_relative must be supplied explicitly; no scientific profile range is assumed")
    raw_bounds = np.asarray(profile_bounds_relative)
    if raw_bounds.dtype.kind not in "iuf" or raw_bounds.shape != (2,):
        raise ValueError("profile_bounds_relative must be a real numeric (lower, upper) pair; booleans and strings are invalid")
    lower, upper = raw_bounds.astype(float, copy=False)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower <= 0 or upper <= lower:
        raise ValueError("profile bounds must be finite positive (lower, upper)")
    if regularization_strength is None:
        raise ValueError("regularization_strength must be supplied explicitly; no default smoothness prior is assumed")
    if isinstance(regularization_strength, Mapping):
        missing_lambdas = sorted(set(active) - set(regularization_strength))
        extra_lambdas = sorted(set(regularization_strength) - set(active))
        if missing_lambdas:
            raise ValueError(
                "regularization_strength mapping must explicitly define every fitted profile; "
                f"missing: {missing_lambdas}"
            )
        if extra_lambdas:
            raise ValueError(
                "regularization_strength mapping contains profiles that are not fitted: "
                f"{extra_lambdas}"
            )
        raw_lambdas = {name: regularization_strength[name] for name in active}
    else:
        raw_lambdas = {name: regularization_strength for name in active}
    lambdas = {}
    for name, raw_value in raw_lambdas.items():
        if isinstance(raw_value, (bool, np.bool_)) or not isinstance(
            raw_value, (int, float, np.integer, np.floating)
        ):
            raise ValueError(
                f"regularization strength for {name} must be a real numeric scalar; booleans and strings are invalid"
            )
        value = float(raw_value)
        if not np.isfinite(value) or value < 0:
            raise ValueError(
                f"regularization strength for {name} must be finite and non-negative"
            )
        lambdas[name] = value
    if not any(value > 0 for value in lambdas.values()):
        raise ValueError("regularization strengths must be finite, non-negative, and at least one positive")
    penalty = _second_difference(n_nodes) if regularization_matrix is None else np.asarray(regularization_matrix, dtype=float)
    if penalty.ndim != 2 or penalty.shape[1] != n_nodes or not np.all(np.isfinite(penalty)):
        raise ValueError("regularization_matrix must be finite with one column per spatial node")
    response = np.eye(f.size) if instrument_response is None else np.asarray(instrument_response, dtype=float)
    if response.shape != (f.size, f.size) or not np.all(np.isfinite(response)) or np.any(response < 0):
        raise ValueError("instrument_response must be a finite non-negative square matrix")
    if not np.allclose(response.sum(axis=0), 1.0, rtol=1e-8, atol=1e-10):
        raise ValueError(
            "instrument_response columns must already sum to one; the calibrated response is applied unchanged"
        )
    if background_mode not in {"zero", "spectrum"}:
        raise ValueError("background_mode must be explicitly set to 'zero' or 'spectrum'; zero background is never assumed")
    if background_mode == "zero" and fixed_background_spectrum is not None:
        raise ValueError("fixed_background_spectrum must be omitted when background_mode='zero'")
    if background_mode == "spectrum" and fixed_background_spectrum is None:
        raise ValueError("fixed_background_spectrum is required when background_mode='spectrum'")
    background = (
        np.zeros_like(y) if background_mode == "zero"
        else _vector(fixed_background_spectrum, "fixed_background_spectrum")
    )
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
    # Identifiability must be assessed from measured spectral information
    # alone.  The augmented least-squares Jacobian also contains regularization
    # rows; counting those rows as data can make an underdetermined profile
    # appear identifiable and produce a falsely finite covariance.
    def data_residual(x):
        return np.linalg.solve(chol, forward(x) + background - y)

    data_jacobian = np.empty((y.size, fit.x.size), dtype=float)
    for column in range(fit.x.size):
        step = 1.0e-5 * max(abs(float(fit.x[column])), 1.0)
        plus = fit.x.copy()
        minus = fit.x.copy()
        if fit.x[column] - step < lo[column]:
            plus[column] += step
            data_jacobian[:, column] = (
                data_residual(plus) - data_residual(fit.x)
            ) / step
        elif fit.x[column] + step > hi[column]:
            minus[column] -= step
            data_jacobian[:, column] = (
                data_residual(fit.x) - data_residual(minus)
            ) / step
        else:
            plus[column] += step
            minus[column] -= step
            data_jacobian[:, column] = (
                data_residual(plus) - data_residual(minus)
            ) / (2.0 * step)

    data_singular = np.linalg.svd(data_jacobian, compute_uv=False)
    data_tol = (
        np.finfo(float).eps * max(data_jacobian.shape)
        * (data_singular[0] if data_singular.size else 0.0)
    )
    data_rank = int(np.count_nonzero(data_singular > data_tol))
    augmented_jacobian = np.asarray(fit.jac, dtype=float)
    augmented_singular = np.linalg.svd(augmented_jacobian, compute_uv=False)
    augmented_tol = (
        np.finfo(float).eps * max(augmented_jacobian.shape)
        * (augmented_singular[0] if augmented_singular.size else 0.0)
    )
    augmented_rank = int(np.count_nonzero(augmented_singular > augmented_tol))
    identifiable = data_rank == fit.x.size
    log_cov = (
        np.linalg.pinv(augmented_jacobian.T @ augmented_jacobian, rcond=1e-12)
        if identifiable and augmented_rank == fit.x.size else None
    )
    scale = np.diag(np.exp(fit.x))
    profile_cov = scale @ log_cov @ scale if log_cov is not None else None
    whitened = np.linalg.solve(chol, residual_spectrum)
    chi2 = float(whitened @ whitened)
    dof = max(y.size - data_rank, 1)
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
                spectral_covariance=measurement_covariance,
                model_discrepancy_covariance=discrepancy_covariance,
                instrument_response=response,
                fixed_background_spectrum=(background if background_mode == 'spectrum' else None),
                background_mode=background_mode,
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
            "jacobian_rank": data_rank,
            "data_only_jacobian_rank": data_rank,
            "augmented_jacobian_rank": augmented_rank,
            "parameter_count": int(fit.x.size),
            "locally_identifiable": bool(identifiable),
            "data_only_singular_values": data_singular,
            "augmented_singular_values": augmented_singular,
            "condition_number": (
                float(data_singular[0]/data_singular[-1])
                if data_singular.size and data_singular[-1] > 0 else float("inf")
            ),
            "regularization_contributed_to_identifiability": bool(
                augmented_rank > data_rank
            ),
        },
        "frequency_shift_Hz": f.copy(), "measured_spectrum": y.copy(),
        "fitted_spectrum": predicted + background, "physical_signal_spectrum": predicted,
        "fixed_background_spectrum": background, "residual": residual_spectrum,
        "spectral_covariance": covariance.copy(),
        "measurement_spectral_covariance": measurement_covariance.copy(),
        "model_discrepancy_covariance": (
            None if discrepancy_covariance is None else discrepancy_covariance.copy()
        ),
        "spectral_covariance_status": (
            "measurement_covariance_plus_independent_model_discrepancy"
            if discrepancy_covariance is not None
            else "measurement_covariance_only"
        ),
        "chi_squared": chi2,
        "degrees_of_freedom": int(dof), "reduced_chi_squared": chi2/dof,
        "regularization_matrix": penalty.copy(), "regularization_strength": dict(lambdas),
        "regularization_is_measurement_covariance": False,
        "regularization_sensitivity": sensitivity,
        "reference_bulk_parameters": {k: base[k] for k in ("electron_density_m3", "electron_temperature_eV", "ion_temperature_eV")},
        "forward_model_evaluations_use_local_full_em_kinetics": True,
        "limitations": [
            "Profiles are relative to supplied reference density and temperatures.",
            "Spatial nodes, ray geometry, beam weights and polarizations are fixed supplied inputs.",
            "Profile covariance is omitted whenever the data-only spectral Jacobian is rank deficient; regularization cannot establish data identifiability.",
            "No conserving collision operator, non-paraxial propagation or experimental validation is implied.",
        ],
    }

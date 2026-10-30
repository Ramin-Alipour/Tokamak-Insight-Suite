"""Advanced calibrated Langmuir-probe analysis extensions.

All functions consume already-calibrated, ordered or orderable voltage/current
samples. They do not perform instrument calibration or experimental validation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import curve_fit


def _prepare_advanced_iv(voltage_V: Any, current_A: Any, current_convention: str) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Apply the canonical calibrated-I/V preprocessing contract to advanced fits."""
    v0, i0 = np.asarray(voltage_V), np.asarray(current_A)
    if v0.ndim != 1 or i0.ndim != 1 or v0.shape != i0.shape:
        raise ValueError("Voltage and current must be equal-length one-dimensional arrays.")
    if v0.dtype.kind not in "iuf" or i0.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must be real numeric arrays.")
    if current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    v, i = v0.astype(float), i0.astype(float)
    if current_convention == "ion_positive":
        i = -i
    finite = np.isfinite(v) & np.isfinite(i)
    input_count = int(v.size)
    v, i = v[finite], i[finite]
    if v.size < 8:
        raise ValueError("At least eight finite paired samples are required.")
    order = np.argsort(v, kind="mergesort")
    v, i = v[order], i[order]
    unique, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
    if unique.size < 8:
        raise ValueError("At least eight distinct finite voltage values are required.")
    if unique.size != v.size:
        i = np.bincount(inverse, weights=i) / counts
        v = unique
    return v, i, {
        "input_pair_count": input_count,
        "finite_pair_count": int(np.count_nonzero(finite)),
        "discarded_nonfinite_pair_count": int(input_count - np.count_nonzero(finite)),
        "duplicate_voltage_pair_count_aggregated": int(np.count_nonzero(counts - 1)),
        "processed_unique_voltage_count": int(v.size),
    }


def _local_polynomial_smooth_nonuniform(voltage: np.ndarray, current: np.ndarray,
                                        window_points: int, polynomial_order: int):
    """Local polynomial smoothing and derivatives in the actual voltage coordinate."""
    smoothed = np.empty_like(current, dtype=float)
    first = np.empty_like(current, dtype=float)
    second = np.empty_like(current, dtype=float)
    half = window_points // 2
    for center in range(voltage.size):
        start = max(0, min(center - half, voltage.size - window_points))
        stop = start + window_points
        dx = voltage[start:stop] - voltage[center]
        scale = float(np.max(np.abs(dx)))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Local polynomial window has a degenerate voltage span.")
        design = np.vander(dx / scale, N=polynomial_order + 1, increasing=True)
        coeff, _, rank, _ = np.linalg.lstsq(design, current[start:stop], rcond=None)
        if rank < polynomial_order + 1:
            raise ValueError("Local polynomial smoothing is rank-deficient on the supplied voltage grid.")
        smoothed[center] = coeff[0]
        first[center] = coeff[1] / scale
        second[center] = 2.0 * coeff[2] / scale**2 if polynomial_order >= 2 else np.nan
    return smoothed, first, second


def fit_langmuir_electron_models(
    voltage_V: Any,
    current_A: Any,
    fit_range_V: tuple[float, float],
    *,
    model: str = "single_maxwellian",
    current_convention: str = "electron_positive",
) -> dict[str, Any]:
    """Fit a single-temperature or two-temperature exponential electron branch.

    The two-temperature model is an empirical mixture of two exponentials. It is
    not proof of a physical bi-Maxwellian distribution; component identifiability
    must be assessed from residuals, covariance, and the selected voltage range.
    """
    v, i, input_quality = _prepare_advanced_iv(voltage_V, current_A, current_convention)
    if not isinstance(fit_range_V, (tuple, list)) or len(fit_range_V) != 2:
        raise ValueError("fit_range_V must contain two increasing voltage bounds.")
    lo, hi = map(float, fit_range_V)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo >= v[0] and hi <= v[-1]):
        raise ValueError("fit_range_V must be finite, increasing, and inside the measured voltage range.")
    mask = (v >= lo) & (v <= hi)
    x, y = v[mask], i[mask]
    if model not in {"single_maxwellian", "two_temperature"}:
        raise ValueError("model must be 'single_maxwellian' or 'two_temperature'.")
    min_count = 4 if model == "single_maxwellian" else 8
    if np.unique(x).size < min_count:
        raise ValueError(f"The selected model requires at least {min_count} distinct voltage values in the fit window.")
    xref = float(x[0])
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), np.finfo(float).tiny)
    ys = y / scale

    if model == "single_maxwellian":
        def fn(xx, offset, amp, slope):
            return offset + amp * np.exp(np.clip(slope * (xx - xref), -700, 700))
        p0 = (float(np.min(ys)), max(float(np.ptp(ys)), 1e-8), 1.0 / max(float(np.ptp(x)), 1e-6))
        pars, cov = curve_fit(fn, x, ys, p0=p0,
                              bounds=([-np.inf, 0.0, 1e-12], [np.inf, np.inf, np.inf]),
                              maxfev=30000)
        fitted = fn(x, *pars) * scale
        slopes = [float(pars[2])]
        amps = [float(pars[1] * scale)]
    else:
        def fn(xx, offset, amp_cold, slope_cold, amp_hot, slope_hot):
            z = np.asarray(xx) - xref
            return (offset + amp_cold * np.exp(np.clip(slope_cold * z, -700, 700))
                    + amp_hot * np.exp(np.clip(slope_hot * z, -700, 700)))
        span = max(float(np.ptp(x)), 1e-6)
        amp0 = max(float(np.ptp(ys)) * 0.5, 1e-8)
        # Ordered slopes reduce label switching; the optimizer still may find
        # weakly identifiable components, which are explicitly flagged below.
        pars, cov = curve_fit(
            fn, x, ys,
            p0=(float(np.min(ys)), amp0, 2.0 / span, amp0, 0.5 / span),
            bounds=([-np.inf, 0.0, 1e-12, 0.0, 1e-12],
                    [np.inf, np.inf, np.inf, np.inf, np.inf]),
            maxfev=60000,
        )
        # Sort temperatures so component 1 is the colder component (larger slope).
        if pars[2] < pars[4]:
            permutation = [0, 3, 4, 1, 2]
            pars = np.asarray(pars)[permutation]
            cov = np.asarray(cov)[np.ix_(permutation, permutation)]
        fitted = fn(x, *pars) * scale
        slopes = [float(pars[2]), float(pars[4])]
        amps = [float(pars[1] * scale), float(pars[3] * scale)]

    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residual ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
    rms = float(np.sqrt(np.mean(residual ** 2)))
    nrmse = rms / float(np.ptp(y)) if np.ptp(y) > 0 else None
    temperatures = [1.0 / s for s in slopes]
    covariance_valid = bool(
        np.asarray(cov).ndim == 2 and cov.shape[0] == cov.shape[1]
        and np.all(np.isfinite(cov)) and np.all(np.diag(cov) >= 0.0)
        and float(np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T)))) > 1e-12 * max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    )
    parameter_names = (["offset_scaled", "amplitude_scaled", "slope_per_V"]
                       if model == "single_maxwellian" else
                       ["offset_scaled", "amplitude_cold_scaled", "slope_cold_per_V",
                        "amplitude_hot_scaled", "slope_hot_per_V"])
    parameter_standard_errors = (np.sqrt(np.maximum(np.diag(cov), 0.0)).tolist()
                                  if covariance_valid else None)
    temperature_standard_uncertainties = (
        [float(np.sqrt(cov[2, 2]) / slopes[0]**2)] if covariance_valid and model == "single_maxwellian"
        else ([float(np.sqrt(cov[2, 2]) / slopes[0]**2),
               float(np.sqrt(cov[4, 4]) / slopes[1]**2)]
              if covariance_valid and model == "two_temperature" else None)
    )
    parameter_correlation = None
    covariance_condition_number = None
    if covariance_valid:
        standard = np.sqrt(np.diag(cov))
        denom = np.outer(standard, standard)
        parameter_correlation = np.divide(cov, denom, out=np.zeros_like(cov), where=denom > 0).tolist()
        covariance_condition_number = float(np.linalg.cond(cov))
    return {
        "model": model,
        "model_class": ("published_single_maxwellian_exponential_relation" if model == "single_maxwellian"
                        else "empirical_two_exponential_mixture_not_physical_population_identification"),
        "physical_interpretation_status": ("conditional_single_maxwellian_fit_requires_regime_review" if model == "single_maxwellian"
                                           else "empirical_decomposition_only_not_evidence_of_two_physical_populations"),
        "fit_range_V": [lo, hi],
        "fit_points": int(len(x)),
        "input_quality": input_quality,
        "preprocessing_contract": "finite_pair_filter_sort_duplicate_voltage_current_mean",
        "electron_temperature_components_eV": temperatures,
        "component_amplitudes_A": amps,
        "fit_R2": r2,
        "fit_residual_rms_A": rms,
        "fit_normalized_rmse": nrmse,
        "covariance_finite": bool(np.all(np.isfinite(cov))),
        "covariance_valid": covariance_valid,
        "parameter_names": parameter_names,
        "parameter_standard_errors": parameter_standard_errors,
        "parameter_correlation_matrix": parameter_correlation,
        "covariance_condition_number": covariance_condition_number,
        "temperature_component_standard_uncertainties_eV": temperature_standard_uncertainties,
        "temperature_component_uncertainty_status": (
            "conditional_covariance_standard_errors_available" if temperature_standard_uncertainties is not None
            else "unavailable_invalid_or_singular_fit_covariance"
        ),
        "current_convention_input": current_convention,
        "current_convention_normalized": "electron_positive",
        "component_identifiability_warning": (
            "Two-exponential decomposition can be non-identifiable; inspect covariance, residuals, "
            "parameter stability, and physical justification before interpreting components."
            if model == "two_temperature" else None
        ),
        "physical_distribution_confirmed": False,
        "interpretation": "Numerical model fit to calibrated I-V samples; not experimental or physical-regime validation.",
    }


def analyze_langmuir_probe_series(
    voltage_V: Any,
    current_A: Any,
    *,
    time_labels: Sequence[Any] | None = None,
    channel_names: Sequence[str] | None = None,
    analysis_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze calibrated sweeps shaped (time, channels, samples) independently."""
    from .langmuir_probe import analyze_langmuir_probe

    v, i = np.asarray(voltage_V), np.asarray(current_A)
    if i.ndim != 3:
        raise ValueError("current_A must have shape (time, channels, samples).")
    if v.ndim == 1:
        if v.size != i.shape[2]:
            raise ValueError("Shared voltage grid length must match the sample dimension.")
        v = np.broadcast_to(v, i.shape)
    elif v.shape != i.shape:
        raise ValueError("voltage_V must be a shared 1-D grid or match current_A shape.")
    nt, nc, ns = i.shape
    if nt < 1 or nc < 1 or ns < 8:
        raise ValueError("Series requires at least one time, one channel, and eight samples per sweep.")
    times = list(range(nt)) if time_labels is None else list(time_labels)
    names = [f"channel_{j + 1}" for j in range(nc)] if channel_names is None else [str(x) for x in channel_names]
    if len(times) != nt or len(names) != nc or len(set(names)) != nc:
        raise ValueError("time_labels and channel_names must match their dimensions; channel names must be unique.")
    options = dict(analysis_options or {})
    results = []
    for t in range(nt):
        row = {}
        for c, name in enumerate(names):
            row[name] = analyze_langmuir_probe(v[t, c], i[t, c], **options)
        results.append({"time_label": times[t], "channels": row})
    return {
        "analysis_scope": "calibrated_time_series_multi_channel_independent_IV_sweeps",
        "time_count": nt,
        "channel_count": nc,
        "sample_count_per_sweep": ns,
        "time_slices": results,
        "temporal_trend_analysis_performed": False,
        "cross_channel_inference_performed": False,
    }


def estimate_multispecies_ion_density(
    ion_saturation_current_A: float, electron_temperature_eV: float, probe_area_m2: float,
    species: Sequence[Mapping[str, float]], *, current_convention: str = "electron_positive",
    fraction_basis: str = "number_density",
) -> dict[str, Any]:
    """Use the registered species-summed Bohm/sheath density relation."""
    from tokamak_analysis.formula_library import langmuir_multispecies_ion_density
    if current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    if fraction_basis != "number_density":
        raise ValueError("fraction_basis must be 'number_density'; mass and current fractions are not interchangeable.")
    normalized_current = -float(ion_saturation_current_A) if current_convention == "ion_positive" else float(ion_saturation_current_A)
    if normalized_current >= 0:
        raise ValueError("Ion current must be negative after normalization to electron_positive.")
    result = langmuir_multispecies_ion_density(
        abs(normalized_current), electron_temperature_eV, probe_area_m2, species
    )
    return {
        "ion_density_estimate_m-3": result["ion_density_m3"],
        "ion_density_standard_uncertainty_m-3": result["uncertainty_ion_density_m3"],
        "ion_density_relative_standard_uncertainty": None,
        "ion_species": result["ion_species"],
        "ion_species_fraction_sum": result["ion_species_fraction_sum"],
        "ion_species_fraction_basis": "number_density_fraction",
        "current_convention_input": current_convention,
        "current_convention_normalized": "electron_positive",
        "model": "species_summed_bohm_sheath_current",
        "formula": "n_i = |I_i,sat| / [e A_p sqrt(e T_e) sum_s(f_s C_s Z_s^(3/2)/sqrt(m_s))]",
        "assumptions": ["common electron temperature", "number-density fractions sum to one",
                        "species Bohm speed sqrt(Z_s e T_e/m_s)", "species sheath coefficients appropriate to geometry and regime",
                        "single effective collection area"],
        "physical_regime_confirmed": False, "uncertainty_propagation_performed": False,
        "uncertainty_status": result["uncertainty_status"],
    }

def langmuir_fit_window_sensitivity(
    voltage_V: Any,
    current_A: Any,
    electron_fit_range_V: tuple[float, float],
    *,
    relative_shifts: Sequence[float] = (-0.1, 0.0, 0.1),
    analysis_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-fit Te over nearby nested windows and report window-selection sensitivity."""
    from .langmuir_probe import analyze_langmuir_probe

    lo, hi = map(float, electron_fit_range_V)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("electron_fit_range_V must contain finite increasing bounds.")
    shifts = list(relative_shifts)
    if not shifts or any(not isinstance(x, (int, float, np.integer, np.floating)) or isinstance(x, (bool, np.bool_)) or not math.isfinite(float(x)) or abs(float(x)) >= 0.5 for x in shifts):
        raise ValueError("relative_shifts must be a non-empty sequence of finite values with absolute value below 0.5.")
    span = hi - lo
    runs = []
    for shift in shifts:
        shrink = float(shift) * span
        candidate = (lo + max(0.0, shrink), hi - max(0.0, shrink)) if shift >= 0 else (lo + shrink, hi - shrink)
        if candidate[0] >= candidate[1]:
            runs.append({"relative_shift": float(shift), "fit_range_V": list(candidate), "status": "invalid_window"})
            continue
        try:
            result = analyze_langmuir_probe(
                voltage_V, current_A, electron_fit_range=candidate, **dict(analysis_options or {})
            )
            te = result.get("electron_temperature_eV")
            runs.append({
                "relative_shift": float(shift),
                "fit_range_V": list(candidate),
                "electron_temperature_eV": te,
                "fit_R2": result.get("electron_retarding_fit_R2"),
                "fit_normalized_rmse": result.get("electron_retarding_fit_normalized_rmse"),
                "status": "fit_returned" if te is not None else "temperature_unavailable",
            })
        except (ValueError, RuntimeError, FloatingPointError) as exc:
            runs.append({"relative_shift": float(shift), "fit_range_V": list(candidate),
                         "status": "fit_failed", "reason": str(exc)})
    valid = [float(x["electron_temperature_eV"]) for x in runs
             if isinstance(x.get("electron_temperature_eV"), (int, float)) and math.isfinite(float(x["electron_temperature_eV"]))]
    spread = (max(valid) - min(valid)) if len(valid) >= 2 else None
    mean = float(np.mean(valid)) if valid else None
    return {
        "reference_fit_range_V": [lo, hi],
        "window_trials": runs,
        "valid_temperature_fit_count": len(valid),
        "temperature_window_sensitivity_span_eV": spread,
        "temperature_window_sensitivity_relative_to_mean": (spread / mean if spread is not None and mean not in (None, 0.0) else None),
        "interpretation": "Sensitivity diagnostic, not a statistical confidence interval or a replacement for justified physical fit-window selection.",
    }


def estimate_electron_density_from_saturation(
    electron_saturation_current_A: float, electron_temperature_eV: float, probe_area_m2: float, *,
    current_convention: str = "electron_positive",
    current_standard_uncertainty_A: float | None = None,
    temperature_relative_standard_uncertainty: float | None = None,
    area_relative_standard_uncertainty: float | None = None,
) -> dict[str, Any]:
    """Use the registered Maxwellian electron-saturation density relation."""
    from tokamak_analysis.formula_library import langmuir_electron_density_from_saturation_current
    if current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    current = -float(electron_saturation_current_A) if current_convention == "ion_positive" else float(electron_saturation_current_A)
    if current <= 0:
        raise ValueError("Electron saturation current must be positive after normalization.")
    uncertainty_values = (current_standard_uncertainty_A, temperature_relative_standard_uncertainty, area_relative_standard_uncertainty)
    supplied = all(value is not None for value in uncertainty_values)
    result = langmuir_electron_density_from_saturation_current(
        current, electron_temperature_eV, probe_area_m2,
        sigma_electron_saturation_current_A=current_standard_uncertainty_A if supplied else None,
        relative_sigma_electron_temperature=temperature_relative_standard_uncertainty if supplied else None,
        relative_sigma_probe_area=area_relative_standard_uncertainty if supplied else None,
    )
    sigma = result.get("uncertainty_electron_density_m3")
    return {
        "electron_density_estimate_m-3": result["electron_density_m3"],
        "electron_density_standard_uncertainty_m-3": sigma,
        "electron_density_uncertainty_status": "conditional_independent_input_uncertainties_propagated" if sigma is not None else "not_calculated_all_three_independent_uncertainties_required",
        "current_convention_input": current_convention,
        "current_convention_normalized": "electron_positive",
        "formula": "n_e = I_e,sat / (e A_p sqrt(e T_e / (2 pi m_e)))",
        "reference": "Merlino (2007), American Journal of Physics 75, 1078-1085, Eq. (6), DOI: 10.1119/1.2772282; Hutchinson (2002), Chapter 3, Section 3.2.5.",
        "assumption": "Maxwellian electron collection with supplied effective area and applicable electron-saturation regime.",
        "physical_regime_confirmed": False, "uncertainty_propagation_performed": sigma is not None,
    }

def estimate_ion_particle_flux(
    ion_density_m3: float, electron_temperature_eV: float, ion_mass_kg: float, *,
    sheath_factor: float, charge_state: int = 1,
    density_relative_standard_uncertainty: float | None = None,
    temperature_relative_standard_uncertainty: float | None = None,
    mass_relative_standard_uncertainty: float | None = None,
    sheath_factor_relative_standard_uncertainty: float | None = None,
) -> dict[str, Any]:
    """Use the registered species Bohm particle-flux relation."""
    from tokamak_analysis.formula_library import langmuir_bohm_particle_flux
    sigmas = (density_relative_standard_uncertainty, temperature_relative_standard_uncertainty,
              mass_relative_standard_uncertainty, sheath_factor_relative_standard_uncertainty)
    supplied = all(value is not None for value in sigmas)
    result = langmuir_bohm_particle_flux(
        ion_density_m3, electron_temperature_eV, ion_mass_kg, sheath_factor,
        charge_state=charge_state,
        relative_sigma_ion_density=density_relative_standard_uncertainty if supplied else None,
        relative_sigma_electron_temperature=temperature_relative_standard_uncertainty if supplied else None,
        relative_sigma_ion_mass=mass_relative_standard_uncertainty if supplied else None,
        relative_sigma_sheath_coefficient=sheath_factor_relative_standard_uncertainty if supplied else None,
    )
    sigma = result.get("uncertainty_ion_particle_flux_m2_s")
    return {
        "ion_sound_speed_m_s": result["ion_sound_speed_m_s"],
        "ion_particle_flux_m-2_s-1": result["ion_particle_flux_m2_s"],
        "ion_charge_state": result["ion_charge_state"],
        "ion_particle_flux_standard_uncertainty_m-2_s-1": sigma,
        "uncertainty_status": "conditional_independent_input_uncertainties_propagated" if sigma is not None else "not_calculated_all_four_independent_uncertainties_required",
        "formula": "Gamma_i = C_s n_i sqrt(Z_i e T_e / m_i)",
        "reference": "Merlino (2007), American Journal of Physics 75, 1078-1085, Eq. (3), DOI: 10.1119/1.2772282; Hutchinson (2002), Chapter 3, Section 3.2.5.",
        "assumption": "Bohm-like species flux; charge state and sheath factor are explicit and model applicability is supplied by the user.",
        "physical_regime_confirmed": False, "uncertainty_propagation_performed": sigma is not None,
    }

def estimate_smoothed_plasma_potential_candidates(
    voltage_V: Any,
    current_A: Any,
    *,
    window_length: int = 7,
    polynomial_order: int = 2,
    current_convention: str = "electron_positive",
) -> dict[str, Any]:
    """Compare raw gradients with local-polynomial derivatives on actual voltage coordinates."""
    v, i, input_quality = _prepare_advanced_iv(voltage_V, current_A, current_convention)
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 5 or window_length % 2 == 0 or window_length > len(v):
        raise ValueError("window_length must be an odd integer between 5 and the number of samples.")
    if isinstance(polynomial_order, bool) or not isinstance(polynomial_order, int) or polynomial_order < 2 or polynomial_order >= window_length:
        raise ValueError("polynomial_order must be an integer in [2, window_length) to calculate second derivatives.")
    raw_didv = np.gradient(i, v)
    raw_d2 = np.gradient(raw_didv, v)
    smooth_i, smooth_didv, smooth_d2 = _local_polynomial_smooth_nonuniform(
        v, i, window_length, polynomial_order
    )
    from tis.probe_analysis import _zero_crossing_candidates
    return {
        "current_convention_input": current_convention,
        "current_convention_normalized": "electron_positive",
        "raw_max_dIdV_candidate_V": float(v[int(np.argmax(raw_didv))]),
        "raw_max_d2IdV2_extremum_V": float(v[int(np.argmax(raw_d2))]),
        "raw_max_d2IdV2_candidate_V": float(v[int(np.argmax(raw_d2))]),
        "raw_d2IdV2_zero_crossing_candidates_V": _zero_crossing_candidates(v, raw_d2),
        "smoothed_max_dIdV_candidate_V": float(v[int(np.argmax(smooth_didv))]),
        "smoothed_max_d2IdV2_extremum_V": float(v[int(np.argmax(smooth_d2))]),
        "smoothed_max_d2IdV2_candidate_V": float(v[int(np.argmax(smooth_d2))]),
        "smoothed_d2IdV2_zero_crossing_candidates_V": _zero_crossing_candidates(v, smooth_d2),
        "second_derivative_inflection_method": "zero_crossings_of_d2IdV2; maxima_are_reported_as_separate_extrema",
        "smoothing_method": "local_polynomial_least_squares_on_actual_voltage_coordinates",
        "window_length_samples": window_length,
        "input_quality": input_quality,
        "voltage_grid_uniformity": "not_required_local_fit_uses_actual_voltage_coordinates",
        "polynomial_order": polynomial_order,
        "smoothed_current_A": smooth_i.tolist(),
        "physical_plasma_potential_confirmed": False,
        "interpretation": "Compare candidates and parameter sensitivity; smoothing is explicit and does not certify the physical plasma potential.",
    }


def fit_langmuir_stretched_exponential(
    voltage_V: Any,
    current_A: Any,
    fit_range_V: tuple[float, float],
    *,
    current_convention: str = "electron_positive",
) -> dict[str, Any]:
    """Fit an empirical stretched-exponential curve shape for comparison only.

    The shape exponent is empirical; this is not a physical EEDF or kinetic kappa-distribution model
    and must not be interpreted as a unique non-Maxwellian electron distribution.
    """
    v, i, input_quality = _prepare_advanced_iv(voltage_V, current_A, current_convention)
    lo, hi = map(float, fit_range_V)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo >= v[0] and hi <= v[-1]):
        raise ValueError("fit_range_V must be finite, increasing, and inside the measured voltage range.")
    mask = (v >= lo) & (v <= hi)
    x, y = v[mask], i[mask]
    if np.unique(x).size < 5:
        raise ValueError("Stretched-exponential fit requires at least five distinct voltage values in the selected window.")
    # Mirrored Kohlrausch-Williams-Watts form for a monotonically increasing
    # retarding-branch signal. The reference voltage is the selected window's
    # upper endpoint, not an inferred plasma potential.
    xref = float(hi)
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), np.finfo(float).tiny)
    def fn(xx, offset, amplitude, slope, shape):
        z = np.maximum(slope * (xref - np.asarray(xx)), 0.0)
        return offset + amplitude * np.exp(-np.clip(np.power(z, shape), 0.0, 700.0))
    pars, cov = curve_fit(
        fn, x, y / scale,
        p0=(float(np.min(y / scale)), max(float(np.ptp(y / scale)), 1e-8), 1.0 / max(float(np.ptp(x)), 1e-6), 1.0),
        bounds=([-np.inf, 0.0, 1e-12, 0.2], [np.inf, np.inf, np.inf, 3.0]),
        maxfev=50000,
    )
    fitted = fn(x, *pars) * scale
    residual = y - fitted
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum(residual ** 2))
    covariance_valid = bool(
        np.asarray(cov).shape == (4, 4) and np.all(np.isfinite(cov))
        and np.all(np.diag(cov) >= 0.0)
        and float(np.min(np.linalg.eigvalsh(0.5 * (cov + cov.T)))) > 1e-12 * max(float(np.max(np.abs(cov))), np.finfo(float).tiny)
    )
    parameter_se = (np.sqrt(np.maximum(np.diag(cov), 0.0)) * np.asarray([scale, scale, 1.0, 1.0])).tolist() if covariance_valid else None
    return {
        "model": "stretched_exponential_empirical_shape",
        "model_class": "empirical_curve_shape_not_a_physical_distribution_model",
        "input_quality": input_quality,
        "preprocessing_contract": "finite_pair_filter_sort_duplicate_voltage_current_mean",
        "current_convention_input": current_convention,
        "current_convention_normalized": "electron_positive",
        "fit_range_V": [lo, hi],
        "fit_points": int(len(x)),
        "reference_voltage_V": xref,
        "reference_voltage_interpretation": "upper_endpoint_of_selected_fit_window_not_a_plasma_potential_estimate",
        "shape_exponent": float(pars[3]),
        "scale_slope_per_V": float(pars[2]),
        "fit_R2": (1.0 - ss_res / ss_tot if ss_tot > 0 else None),
        "fit_residual_rms_A": float(np.sqrt(np.mean(residual ** 2))),
        "covariance_finite": bool(np.all(np.isfinite(cov))),
        "covariance_valid": covariance_valid,
        "parameter_names": ["offset_scaled", "amplitude_scaled", "scale_slope_per_V", "shape_exponent"],
        "parameter_standard_errors": parameter_se,
        "shape_exponent_standard_uncertainty": parameter_se[3] if parameter_se is not None else None,
        "scale_slope_standard_uncertainty_per_V": parameter_se[2] if parameter_se is not None else None,
        "parameter_uncertainty_status": "conditional_covariance_standard_errors_available" if parameter_se is not None else "unavailable_invalid_or_singular_fit_covariance",
        "kinetic_distribution_identified": False,
        "interpretation": "Empirical curve-shape alternative only; no unique physical distribution is identified; covariance errors are conditional on this empirical model and selected window.",
    }

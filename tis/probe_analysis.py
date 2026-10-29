"""Electrostatic probe I-V analysis and temperature estimates."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy import optimize
from .spectral_analysis import _r2


def _is_finite_real_scalar(value: Any) -> bool:
    """Accept finite real scalars while rejecting booleans and numeric strings."""
    if not isinstance(value, (int, float, np.integer, np.floating)) or isinstance(value, (bool, np.bool_)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def _floating_potential_candidates(voltage: np.ndarray, current: np.ndarray) -> list[float]:
    """Return distinct zero-current crossings for sorted, unique voltage samples."""
    exact_zero = current == 0.0
    candidates: list[float] = []
    zero_idx = np.flatnonzero(exact_zero)
    if zero_idx.size:
        groups = np.split(zero_idx, np.flatnonzero(np.diff(zero_idx) > 1) + 1)
        candidates.extend(float(0.5 * (voltage[g[0]] + voltage[g[-1]])) for g in groups)
    for k in range(len(current) - 1):
        if exact_zero[k] or exact_zero[k + 1]:
            continue
        if (current[k] < 0 < current[k + 1]) or (current[k] > 0 > current[k + 1]):
            candidates.append(float(
                voltage[k] - current[k] * (voltage[k + 1] - voltage[k]) /
                (current[k + 1] - current[k])
            ))
    candidates.sort()
    unique: list[float] = []
    for candidate in candidates:
        if not unique or not np.isclose(candidate, unique[-1], rtol=1e-10, atol=1e-12):
            unique.append(candidate)
    return unique


def probe_iv_summary(voltage: Any, current: Any) -> dict[str, Any]:
    """Descriptive I-V preprocessing only; does not infer plasma parameters from an unvalidated regime model."""
    raw_v = np.asarray(voltage)
    raw_i = np.asarray(current)
    if raw_v.ndim != 1 or raw_i.ndim != 1:
        raise ValueError("Voltage and current inputs must be one-dimensional arrays.")
    if raw_v.dtype.kind not in "iuf" or raw_i.dtype.kind not in "iuf":
        raise ValueError("Voltage and current inputs must contain real numeric values, not booleans or strings.")
    v = raw_v.astype(float, copy=False)
    i = raw_i.astype(float, copy=False)
    if len(v) != len(i):
        raise ValueError("Voltage and current arrays must have equal lengths.")
    good = np.isfinite(v) & np.isfinite(i)
    v, i = v[good], i[good]
    if len(v) < 5:
        raise ValueError("At least 5 finite I-V points are required.")
    order = np.argsort(v)
    v, i = v[order], i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v) < 5:
        raise ValueError("Voltage must contain at least 5 unique values.")
    if len(unique_v) < len(v):
        sums = np.bincount(inverse, weights=i)
        counts = np.bincount(inverse)
        v, i = unique_v, sums / counts
    dIdV = np.gradient(i, v)
    crossings = _floating_potential_candidates(v, i)
    fp = crossings[0] if len(crossings) == 1 else None
    status = ("single_zero_crossing" if len(crossings) == 1 else
              "no_zero_crossing_in_measured_sweep" if not crossings else
              "ambiguous_multiple_zero_crossings")
    return {
        "n_points": int(len(v)),
        "voltage_min_V": float(v.min()),
        "voltage_max_V": float(v.max()),
        "current_min_A": float(i.min()),
        "current_max_A": float(i.max()),
        "floating_potential_interpolated_V": fp,
        "floating_potential_candidates_V": crossings,
        "floating_potential_status": status,
        "max_dIdV_A_per_V": float(np.max(dIdV)),
        "min_dIdV_A_per_V": float(np.min(dIdV)),
        "warning": "Descriptive summary only. Physical interpretation and temperature estimates require a justified probe regime/model; density and particle flux are not inferred.",
    }


# Expanded diagnostic workflows (v0.3): calculations operate on user-supplied,
# calibrated data. They do not invent device calibration or geometry.

def langmuir_iv_analysis(voltage: Any, current: Any,
                         electron_fit_range: tuple[float, float] | None = None,
                         ion_fit_range: tuple[float, float] | None = None,
                         current_convention: str = "electron_positive",
                         probe_area_m2: float | None = None,
                         ion_mass_kg: float | None = None,
                         ion_saturation_coefficient: float | None = None,
                         sheath_model: str | None = None,
                         ion_current_uncertainty_A: float | None = None,
                         probe_area_relative_uncertainty: float | None = None,
                         ion_mass_relative_uncertainty: float | None = None,
                         ion_saturation_coefficient_relative_uncertainty: float | None = None,
                         electron_temperature_relative_uncertainty: float | None = None,
                         electron_temperature_calibration_relative_uncertainty: float | None = None,
                         electron_temperature_fit_window_relative_uncertainty: float | None = None,
                         density_model_relative_uncertainty: float | None = None) -> dict[str, Any]:
    """Analyze a calibrated Langmuir I-V sweep with optional explicit sheath-model density.

    Density requires explicit collection area, ion mass, sheath-model coefficient and
    model label, plus analyst-selected ion/electron fit windows. No coefficient is
    silently defaulted because it depends on geometry and plasma regime.
    """
    if not isinstance(current_convention, str) or current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    for name, fit_range in (("electron_fit_range", electron_fit_range), ("ion_fit_range", ion_fit_range)):
        if fit_range is not None:
            try:
                raw_limits = list(fit_range)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain exactly two finite voltage limits in strictly increasing order; limits must be real numeric scalars.") from exc
            if len(raw_limits) != 2 or not all(_is_finite_real_scalar(value) for value in raw_limits):
                raise ValueError(f"{name} must contain exactly two finite voltage limits in strictly increasing order; limits must be real numeric scalars.")
            limits = np.asarray(raw_limits, dtype=float)
            if limits[1] <= limits[0]:
                raise ValueError(f"{name} must contain exactly two finite voltage limits in strictly increasing order; limits must be real numeric scalars.")
            # Normalize one-shot iterables once: the validated limits are used
            # again below when selecting fit samples.
            if name == "electron_fit_range":
                electron_fit_range = (float(limits[0]), float(limits[1]))
            else:
                ion_fit_range = (float(limits[0]), float(limits[1]))
    raw_v=np.asarray(voltage); raw_i=np.asarray(current)
    if raw_v.ndim != 1 or raw_i.ndim != 1:
        raise ValueError("Voltage and current inputs must be one-dimensional arrays.")
    if raw_v.dtype.kind not in "iuf" or raw_i.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must contain real numeric values; booleans, strings and complex values are not accepted.")
    v=raw_v.astype(float, copy=True); i=raw_i.astype(float, copy=True)
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
    input_points = int(len(v))
    if current_convention=="ion_positive": i=-i
    good=np.isfinite(v)&np.isfinite(i)
    finite_pair_count = int(np.count_nonzero(good))
    v,i=v[good],i[good]
    if len(v)<8: raise ValueError('At least 8 valid voltage points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v)<8: raise ValueError('At least 8 distinct voltage points are required after duplicate aggregation.')
    duplicate_count = int(len(v) - len(unique_v))
    if duplicate_count:
        sums=np.bincount(inverse,weights=i)
        counts=np.bincount(inverse)
        v,i=unique_v,sums/counts
    out=probe_iv_summary(v,i)
    # Preserve the exact calibrated samples used by every downstream calculation.
    # These are processed values only: non-finite pairs are removed, duplicate
    # voltages are averaged, polarity is normalized, and rows are voltage-sorted.
    out["processed_voltage_V"] = v.tolist()
    out["processed_current_A"] = i.tolist()
    # Keep preprocessing decisions machine-readable so exported results can be
    # audited without reconstructing how many points were dropped or averaged.
    out["input_quality"] = {
        "input_pair_count": input_points,
        "finite_pair_count": finite_pair_count,
        "discarded_nonfinite_pair_count": input_points - finite_pair_count,
        "duplicate_voltage_pair_count_aggregated": duplicate_count,
        "processed_unique_voltage_count": int(len(v)),
        "voltage_span_V": float(v[-1] - v[0]),
        "current_convention_normalized": "electron_positive",
    }
    out['electron_temperature_eV'] = None
    out['electron_temperature_standard_error_eV'] = None
    out["plasma_potential_method"] = "second_derivative_maximum_candidate"
    out["plasma_potential_status"] = "numerical_candidate_requires_regime_review"
    out["fit_window_review"] = {
        "physical_regime_certified": False,
        "required_review": [
            "confirm electron-retarding interval is exponential and excludes saturation/transition regions",
            "confirm ion interval is a physically justified ion-saturation regime",
            "review fit residuals, R-squared, voltage span and probe-specific theory"
        ],
        "note": "Numerical fit quality cannot prove physical regime validity from I-V samples alone."
    }
    # Always provide explicitly labelled descriptive saturation-tail estimates.
    # These are fallbacks only; a user-supplied ion_fit_range overrides the ion tail.
    # Tail size is bounded to avoid letting a few endpoint samples dominate.
    tail_n = max(3, int(math.ceil(0.1 * len(v))))
    out["ion_saturation_current_A"] = float(np.median(i[:tail_n]))
    out["ion_saturation_current_method"] = "median_lowest_voltage_tail_fallback"
    out["electron_saturation_current_A"] = float(np.median(i[-tail_n:]))
    out["electron_saturation_current_method"] = "median_highest_voltage_tail_descriptive"
    out["ion_saturation_current_status"] = "descriptive_tail_fallback_not_physical_saturation"
    out["ion_saturation_current_physical_regime_confirmed"] = False
    # Zero-current plateaus use their midpoint; ambiguous crossings stay explicit.
    unique_candidates = _floating_potential_candidates(v, i)
    out['floating_potential_candidates_V'] = unique_candidates
    out['floating_potential_V'] = unique_candidates[0] if len(unique_candidates) == 1 else None
    if not unique_candidates:
        out['floating_potential_status'] = 'no_zero_crossing_in_measured_sweep'
    elif len(unique_candidates) > 1:
        out['floating_potential_status'] = 'ambiguous_multiple_zero_crossings'
    else:
        out['floating_potential_status'] = 'single_zero_crossing'
    didv=np.gradient(i,v)
    if not np.all(np.isfinite(didv)):
        raise ValueError("dI/dV is non-finite; voltage values must be distinct and well-conditioned.")
    out["dIdV_A_per_V"] = didv.tolist()
    d2IdV2 = np.gradient(didv, v)
    if not np.all(np.isfinite(d2IdV2)):
        raise ValueError("d2I/dV2 is non-finite; voltage values must be distinct and well-conditioned.")
    first_derivative_index = int(np.argmax(didv))
    second_derivative_index = int(np.argmax(d2IdV2))
    out["d2IdV2_A_per_V2"] = d2IdV2.tolist()
    out["max_dIdV_A_per_V"] = float(np.max(didv))
    out["plasma_potential_proxy_V"] = float(v[first_derivative_index])
    out["plasma_potential_dIdV_max_candidate_V"] = float(v[first_derivative_index])
    out["plasma_potential_d2IdV2_max_candidate_V"] = float(v[second_derivative_index])
    out["plasma_potential_estimate_V"] = float(v[second_derivative_index])
    out["plasma_potential_candidate_method"] = "maximum_second_derivative_of_current"
    out["plasma_potential_status"] = "numerical_candidate_requires_regime_review"
    out["plasma_potential_voltage_grid_spacing_median_V"] = float(np.median(np.diff(v)))
    out["plasma_potential_uncertainty_note"] = ("Voltage-grid spacing is reported as a resolution indicator, not a statistical uncertainty; "
        "physical Vp interpretation depends on probe regime and branch coverage.")
    if ion_fit_range is not None:
        m=(v>=min(ion_fit_range))&(v<=max(ion_fit_range))
        if m.sum() < 3:
            raise ValueError(
                "ion_fit_range must overlap the measured voltage sweep and contain "
                "at least 3 valid samples within the measured voltage range."
            )
        if min(ion_fit_range) < min(v) or max(ion_fit_range) > max(v):
            raise ValueError(
                f"ion_fit_range must overlap and lie inside the measured voltage range "
                f"[{min(v):g}, {max(v):g}] V."
            )
        p = np.polyfit(v[m], i[m], 1)
        ion_fitted = np.polyval(p, v[m])
        ion_residual = i[m] - ion_fitted
        ion_rms = float(np.sqrt(np.mean(ion_residual ** 2)))
        ion_span = float(np.ptp(i[m]))
        ion_r2 = _r2(i[m], ion_fitted)
        out['ion_saturation_linear_slope_A_per_V'] = float(p[0])
        out['ion_saturation_intercept_A'] = float(p[1])
        out['ion_saturation_fit_R2'] = ion_r2
        out['ion_saturation_fit_residual_rms_A'] = ion_rms
        out['ion_saturation_fit_normalized_rmse'] = ion_rms / ion_span if ion_span > 0 else None
        out['ion_saturation_fit_points'] = int(np.count_nonzero(m))
        ion_nrmse = out['ion_saturation_fit_normalized_rmse']
        ion_relative_slope = abs(float(p[0])) * float(np.ptp(v[m])) / max(abs(float(np.median(i[m]))), np.finfo(float).tiny)
        ion_numerically_flat = bool(np.isfinite(ion_relative_slope) and ion_relative_slope <= 0.1)
        ion_fit_good = bool(ion_r2 is not None and np.isfinite(ion_r2) and ion_r2 >= 0.98
                            and ion_nrmse is not None and np.isfinite(ion_nrmse) and ion_nrmse <= 0.1)
        out['ion_saturation_fit_status'] = (
            'numerical_fit_review_pass_not_physical_validation' if ion_fit_good
            else 'weak_or_undefined_numerical_fit_review_required'
        )
        out["ion_saturation_regime_review"] = {
            "numerically_flat_candidate": ion_numerically_flat,
            "relative_current_change_over_window": float(ion_relative_slope),
            "physical_ion_saturation_confirmed": False,
            "note": "Flatness is only a numerical screening indicator; physical saturation requires an appropriate probe regime and theory."
        }
        out['ion_saturation_current_A'] = float(np.median(i[m]))
        out['ion_saturation_current_method'] = 'median_user_selected_ion_fit_range'
    if electron_fit_range is not None:
        if min(electron_fit_range) < min(v) or max(electron_fit_range) > max(v):
            raise ValueError(
                f"electron_fit_range must overlap and lie inside the measured voltage range "
                f"[{min(v):g}, {max(v):g}] V."
            )
        m = (v >= min(electron_fit_range)) & (v <= max(electron_fit_range))
        if m.sum() < 4:
            raise ValueError("Electron-retarding fit range needs at least 4 points with positive current after ion-current baseline subtraction.")
        x, measured = v[m], i[m]
        x_ref = float(x[0])
        current_scale = max(float(np.ptp(measured)), float(np.max(np.abs(measured))), np.finfo(float).tiny)
        ion_baseline = float(out.get("ion_saturation_current_A", 0.0))
        baseline_scaled = ion_baseline / current_scale
        excess = measured - ion_baseline
        positive = excess > 0
        slope0 = 5.0 / max(float(np.ptp(x)), np.finfo(float).eps)
        if np.count_nonzero(positive) >= 3:
            trial_slope = float(np.polyfit(x[positive], np.log(excess[positive]), 1)[0])
            if np.isfinite(trial_slope) and trial_slope > 0:
                slope0 = trial_slope
        if np.count_nonzero(positive) < 3:
            raise ValueError("Electron-retarding fit requires at least three samples above the estimated ion-current baseline; choose a physically justified electron_fit_range.")
        first_positive = int(np.flatnonzero(positive)[0])
        amplitude_at_first_positive = excess[first_positive] * np.exp(-slope0 * (x[first_positive] - x_ref))
        amplitude0 = max(float(amplitude_at_first_positive) / current_scale, 1e-12)

        def _electron_branch_model(x_values, baseline_scaled_fit, amplitude_scaled_fit, slope_per_V):
            exponent = np.clip(slope_per_V * (np.asarray(x_values) - x_ref), -700.0, 700.0)
            return baseline_scaled_fit + amplitude_scaled_fit * np.exp(exponent)

        # Fit the ion-current offset and exponential branch together. Subtracting a
        # noisy median ion-tail estimate before taking log(current) biases Te,
        # especially when the electron signal is small near the lower fit boundary.
        parameters, covariance = optimize.curve_fit(
            _electron_branch_model,
            x,
            measured / current_scale,
            p0=(baseline_scaled, amplitude0, slope0),
            bounds=([-np.inf, 0.0, 1e-12], [np.inf, np.inf, np.inf]),
            maxfev=20000,
        )
        baseline_fit_A = float(parameters[0] * current_scale)
        slope = float(parameters[2])
        fitted_current = _electron_branch_model(x, *parameters) * current_scale
        out["electron_retarding_fit_R2"] = _r2(measured, fitted_current)
        residual = measured - fitted_current
        out["electron_retarding_fit_residual_rms_A"] = float(np.sqrt(np.mean(residual ** 2)))
        signal_scale = float(np.ptp(measured))
        out["electron_retarding_fit_normalized_rmse"] = (
            float(np.sqrt(np.mean(residual ** 2)) / signal_scale) if signal_scale > 0 else None
        )
        out["electron_retarding_fit_points"] = int(np.count_nonzero(m))
        out["electron_fit_points"] = int(np.count_nonzero(m))  # Backward-compatible field name.
        electron_nrmse = out["electron_retarding_fit_normalized_rmse"]
        electron_fit_good = bool(
            np.isfinite(out["electron_retarding_fit_R2"])
            and out["electron_retarding_fit_R2"] >= 0.98
            and electron_nrmse is not None and np.isfinite(electron_nrmse)
            and electron_nrmse <= 0.1
        )
        out["electron_retarding_fit_status"] = (
            "numerical_fit_review_pass_not_physical_validation" if electron_fit_good
            else "weak_or_undefined_numerical_fit_review_required"
        )
        out["electron_retarding_regime_review"] = {
            "numerical_fit_quality_pass": electron_fit_good,
            "physical_regime_confirmed": False,
            "selected_voltage_span_V": float(np.ptp(x)),
            "note": "An acceptable exponential fit is necessary but not sufficient to establish a Maxwellian electron-retarding regime."
        }
        out["ion_current_baseline_subtracted_A"] = baseline_fit_A
        out["electron_retarding_log_slope_per_V"] = slope
        out["electron_retarding_fit_intercept"] = float(np.log(max(parameters[1] * current_scale, np.finfo(float).tiny)) - slope * x_ref)
        if not np.isfinite(slope) or slope <= 0:
            raise ValueError("Electron-retarding exponential fit returned a non-positive or non-finite slope; electron temperature is undefined.")
        if not np.isfinite(out["electron_retarding_fit_R2"]):
            raise ValueError("Electron-retarding fit quality is non-finite; check the selected fit range and data.")
        out["electron_temperature_eV"] = 1.0 / slope
        slope_se = float(np.sqrt(max(float(covariance[2, 2]), 0.0))) if covariance.shape == (3, 3) and np.isfinite(covariance[2, 2]) else None
        out["electron_temperature_standard_error_eV"] = (slope_se / slope**2) if slope_se is not None else None
        te_relative_components = {
            "fit_statistical": ((out["electron_temperature_standard_error_eV"] / out["electron_temperature_eV"])
                                if out["electron_temperature_standard_error_eV"] is not None else 0.0),
            "user_supplied_nonstatistical": float(electron_temperature_relative_uncertainty or 0.0),
            "calibration": float(electron_temperature_calibration_relative_uncertainty or 0.0),
            "fit_window": float(electron_temperature_fit_window_relative_uncertainty or 0.0),
        }
        for component_name, component_value in te_relative_components.items():
            if not _is_finite_real_scalar(component_value) or component_value < 0:
                raise ValueError(f"{component_name} relative electron-temperature uncertainty must be finite and non-negative.")
        te_total_relative = math.sqrt(sum(value ** 2 for value in te_relative_components.values()))
        out["electron_temperature_relative_standard_uncertainty"] = float(te_total_relative)
        out["electron_temperature_total_standard_uncertainty_eV"] = float(out["electron_temperature_eV"] * te_total_relative)
        out["electron_temperature_uncertainty_components_relative"] = te_relative_components
        out["electron_temperature_uncertainty_status"] = "quadrature_of_fit_statistical_and_explicitly_supplied_components"
        out["electron_temperature_uncertainty_note"] = (
            "Combined uncertainty includes fit covariance and only user-supplied nonstatistical, calibration, "
            "and fit-window components. Unspecified effects remain unquantified.")
    density_args = (probe_area_m2, ion_mass_kg, ion_saturation_coefficient, sheath_model)
    if any(value is not None for value in density_args):
        if not all(value is not None for value in density_args):
            raise ValueError("Density inversion requires probe_area_m2, ion_mass_kg, ion_saturation_coefficient, and sheath_model together.")
        if electron_fit_range is None or ion_fit_range is None:
            raise ValueError("Density inversion requires analyst-selected electron_fit_range and ion_fit_range; descriptive tail fallbacks are not sufficient.")
        for name, value in (("probe_area_m2", probe_area_m2), ("ion_mass_kg", ion_mass_kg), ("ion_saturation_coefficient", ion_saturation_coefficient)):
            if not _is_finite_real_scalar(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite (a real numeric scalar).")
        if not isinstance(sheath_model, str) or not sheath_model.strip():
            raise ValueError("sheath_model must be a non-empty label for the selected physical model.")
        te = out.get("electron_temperature_eV")
        ion_current = out.get("ion_saturation_current_A")
        if not _is_finite_real_scalar(te) or te <= 0:
            raise ValueError("A positive fitted electron temperature is required for ion-density inversion.")
        if not _is_finite_real_scalar(ion_current) or ion_current >= 0:
            raise ValueError("Ion-density inversion requires a negative ion-saturation current after normalization to the electron-positive convention; inspect polarity and ion_fit_range.")
        from .constants import E_CHARGE
        denominator = float(ion_saturation_coefficient) * E_CHARGE * float(probe_area_m2) * math.sqrt(E_CHARGE * float(te) / float(ion_mass_kg))
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError("The selected density model produced an invalid denominator.")
        density = abs(float(ion_current)) / denominator
        if not math.isfinite(density) or density <= 0:
            raise ValueError("The selected density model produced a non-finite or non-positive density.")
        out["ion_density_estimate_m-3"] = float(density)
        out["ion_density_status"] = "model_based_estimate_review_assumptions"
        uncertainty_inputs = {
            "ion_current_uncertainty_A": ion_current_uncertainty_A,
            "probe_area_relative_uncertainty": probe_area_relative_uncertainty,
            "ion_mass_relative_uncertainty": ion_mass_relative_uncertainty,
            "ion_saturation_coefficient_relative_uncertainty": ion_saturation_coefficient_relative_uncertainty,
            "electron_temperature_relative_uncertainty": electron_temperature_relative_uncertainty,
            "electron_temperature_calibration_relative_uncertainty": electron_temperature_calibration_relative_uncertainty,
            "electron_temperature_fit_window_relative_uncertainty": electron_temperature_fit_window_relative_uncertainty,
            "density_model_relative_uncertainty": density_model_relative_uncertainty,
        }
        for name, value in uncertainty_inputs.items():
            if value is not None and (not _is_finite_real_scalar(value) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative.")
        out["ion_density_model"] = {
            "sheath_model": sheath_model.strip(),
            "ion_saturation_coefficient": float(ion_saturation_coefficient),
            "probe_area_m2": float(probe_area_m2),
            "ion_mass_kg": float(ion_mass_kg),
            "formula": "n_i = |I_i,sat| / (C_s e A_p sqrt(e T_e / m_i)); T_e in eV",
            "assumption": "Single-ion species; supplied coefficient is appropriate for the declared sheath model, probe geometry and plasma regime.",
            "uncertainty_note": "Only supplied uncertainty components are propagated; unprovided systematic/model uncertainties remain excluded.",
        }
        if any(value is not None for value in uncertainty_inputs.values()):
            rel_i = (float(ion_current_uncertainty_A) / abs(float(ion_current))
                     if ion_current_uncertainty_A is not None else 0.0)
            te_se = out.get("electron_temperature_standard_error_eV")
            rel_te_stat = (float(te_se) / float(te) if _is_finite_real_scalar(te_se) and te_se >= 0 else 0.0)
            rel_te_sys = math.sqrt(
                float(electron_temperature_relative_uncertainty or 0.0) ** 2
                + float(electron_temperature_calibration_relative_uncertainty or 0.0) ** 2
                + float(electron_temperature_fit_window_relative_uncertainty or 0.0) ** 2
            )
            rel_te = math.sqrt(rel_te_stat**2 + rel_te_sys**2)
            rel_components = {
                "ion_current": rel_i,
                "probe_area": float(probe_area_relative_uncertainty or 0.0),
                "ion_mass": 0.5 * float(ion_mass_relative_uncertainty or 0.0),
                "sheath_coefficient": float(ion_saturation_coefficient_relative_uncertainty or 0.0),
                "electron_temperature": 0.5 * rel_te,
                "density_model_discrepancy": float(density_model_relative_uncertainty or 0.0),
            }
            relative_uncertainty = math.sqrt(sum(value**2 for value in rel_components.values()))
            out["ion_density_standard_uncertainty_m-3"] = float(density * relative_uncertainty)
            out["ion_density_relative_standard_uncertainty"] = float(relative_uncertainty)
            out["ion_density_uncertainty_components_relative"] = rel_components
            out["ion_density_uncertainty_status"] = "partial_first_order_propagation_supplied_components_only"
        else:
            out["ion_density_standard_uncertainty_m-3"] = None
            out["ion_density_relative_standard_uncertainty"] = None
            out["ion_density_uncertainty_components_relative"] = None
            out["ion_density_uncertainty_status"] = "not_propagated_uncertainty_inputs_not_supplied"
    out['current_convention_input'] = current_convention
    out['current_convention'] = 'electron_positive'  # All returned current values use this normalized convention.
    if 'ion_saturation_current_A' in out and out['ion_saturation_current_A']>0:
        out['sign_warning']='Ion-saturation current is positive in the electron-positive convention; check current_convention_input.'
    out['analysis_scope'] = "calibrated_single_probe_IV_sweep_only"
    out['interpretation_warning']='Single-probe estimates depend on sweep direction, magnetic field, collisions, secondary emission, sheath expansion, surface condition and fit-window choice. Numerical fit diagnostics do not establish physical regime validity. This workflow does not claim multi-probe, triple-probe, array, or full probe-system analysis.'
    return out


def suggest_langmuir_fit_windows(
    voltage: Any, current: Any, current_convention: str = "electron_positive",
    *, min_points: int = 6, min_voltage_fraction: float = 0.12,
) -> dict[str, Any]:
    """Suggest reviewable I-V fit windows; suggestions do not validate physical regimes.

    Electron candidates are ranked using log(I - low-voltage-tail median) linearity.
    The low-voltage interval is only a candidate, not proof of ion saturation.
    """
    if not isinstance(current_convention, str) or current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    if isinstance(min_points, bool) or not isinstance(min_points, (int, np.integer)) or min_points < 6:
        raise ValueError("min_points must be an integer of at least 6.")
    if not _is_finite_real_scalar(min_voltage_fraction) or not 0 < min_voltage_fraction <= 1:
        raise ValueError("min_voltage_fraction must be finite and in (0, 1].")
    raw_v, raw_i = np.asarray(voltage), np.asarray(current)
    if raw_v.ndim != 1 or raw_i.ndim != 1 or raw_v.shape != raw_i.shape:
        raise ValueError("Voltage and current must be equal-length one-dimensional arrays.")
    if raw_v.dtype.kind not in "iuf" or raw_i.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must contain real numeric values.")
    v, i = raw_v.astype(float, copy=True), raw_i.astype(float, copy=True)
    if current_convention == "ion_positive":
        i = -i
    finite = np.isfinite(v) & np.isfinite(i)
    v, i = v[finite], i[finite]
    if len(v) < min_points:
        raise ValueError(f"At least {min_points} finite paired samples are required to suggest fit windows.")
    order = np.argsort(v)
    v, i = v[order], i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v) < min_points:
        raise ValueError(f"At least {min_points} distinct voltage values are required to suggest fit windows.")
    if len(unique_v) != len(v):
        sums, counts = np.bincount(inverse, weights=i), np.bincount(inverse)
        v, i = unique_v, sums / counts
    # Bound the quadratic window scan for long acquisitions. Suggestions are
    # based on an evenly spaced subset; the final nonlinear fit still uses the
    # complete user-selected interval and all available samples.
    original_unique_count = int(len(v))
    suggestion_sampled = original_unique_count > 250
    if suggestion_sampled:
        sample_indices = np.linspace(0, original_unique_count - 1, 250, dtype=int)
        v, i = v[sample_indices], i[sample_indices]
    n = len(v)
    tail_n = max(3, int(math.ceil(0.1 * n)))
    ion_tail = float(np.median(i[:tail_n]))
    ion_count = max(3, min(n, int(math.ceil(0.12 * n))))
    ion_range = (float(v[0]), float(v[ion_count - 1]))
    min_span = float(v[-1] - v[0]) * float(min_voltage_fraction)
    candidates = []
    for start in range(n - min_points + 1):
        for stop in range(start + min_points, n + 1):
            x, y = v[start:stop], i[start:stop]
            if x[-1] - x[0] < min_span:
                continue
            excess = y - ion_tail
            valid = excess > 0
            if np.count_nonzero(valid) < min_points:
                continue
            lx, ly = x[valid], np.log(excess[valid])
            slope, intercept = np.polyfit(lx, ly, 1)
            if not np.isfinite(slope) or slope <= 0:
                continue
            predicted = intercept + slope * lx
            ss_res = float(np.sum((ly - predicted) ** 2))
            ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("-inf")
            if np.isfinite(r2):
                candidates.append({"start": start, "stop": stop, "r2": r2,
                                   "points": int(np.count_nonzero(valid)), "span": float(x[-1] - x[0])})
    selected = None
    if candidates:
        good = [c for c in candidates if c["r2"] >= 0.98]
        pool = good if good else candidates
        selected = max(pool, key=lambda c: (c["span"], c["r2"], c["points"]))
    return {
        "electron_fit_range_V": ((float(v[selected["start"]]), float(v[selected["stop"] - 1]))
                                  if selected is not None else None),
        "electron_candidate_log_R2": float(selected["r2"]) if selected is not None else None,
        "electron_candidate_points": selected["points"] if selected is not None else 0,
        "electron_candidate_status": (
            "candidate_found_review_curve_and_final_fit" if selected is not None and selected["r2"] >= 0.98
            else "low_quality_candidate_review_manually" if selected is not None
            else "no_defensible_candidate_found_select_window_manually"),
        "ion_fit_range_candidate_V": ion_range,
        "ion_candidate_status": "low_voltage_tail_candidate_only_not_verified_ion_saturation",
        "ion_baseline_tail_median_A": ion_tail,
        "current_convention_normalized": "electron_positive",
        "suggestion_sampled": suggestion_sampled,
        "suggestion_input_unique_voltage_count": original_unique_count,
        "suggestion_points_scanned": int(n),
        "warning": "Automated ranges are suggestions, not proof of a physical regime. Inspect the I-V curve and fit residuals.",
    }

def floating_potential(voltage: Any, current: Any) -> float | None:
    """Return the unique interpolated floating potential, or None if absent/ambiguous.

    Inputs are calibrated voltage/current arrays. Non-finite pairs are ignored;
    duplicate voltages are combined by averaging their current values.
    """
    raw_v, raw_i = np.asarray(voltage), np.asarray(current)
    if raw_v.ndim != 1 or raw_i.ndim != 1:
        raise ValueError("Voltage and current inputs must be one-dimensional arrays.")
    if raw_v.dtype.kind not in "iuf" or raw_i.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must contain real numeric values.")
    if len(raw_v) != len(raw_i):
        raise ValueError("Voltage and current arrays must have equal lengths.")
    v, i = raw_v.astype(float, copy=True), raw_i.astype(float, copy=True)
    good = np.isfinite(v) & np.isfinite(i)
    v, i = v[good], i[good]
    if len(v) < 2:
        return None
    order = np.argsort(v)
    v, i = v[order], i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v) < 2:
        return None
    if len(unique_v) != len(v):
        sums = np.bincount(inverse, weights=i)
        counts = np.bincount(inverse)
        v, i = unique_v, sums / counts
    candidates = _floating_potential_candidates(v, i)
    return candidates[0] if len(candidates) == 1 else None


def triple_probe_temperature_eV(v_difference_V: float, model_factor: float = 1.0) -> dict[str, float | str]:
    """First-order triple-probe Te estimate under a user-selected idealized exponential-current model."""
    if (not _is_finite_real_scalar(v_difference_V) or v_difference_V <= 0
            or not _is_finite_real_scalar(model_factor) or model_factor <= 0):
        raise ValueError('Voltage difference and model factor must be positive and finite.')
    te=float(v_difference_V/(np.log(2)*model_factor))
    return {'electron_temperature_estimate_eV':te,'voltage_difference_V':float(v_difference_V),'model_factor':float(model_factor),'assumption':'Idealized triple-probe relation; confirm the actual electrode configuration, current balance, sheath model and circuit definition before use.'}

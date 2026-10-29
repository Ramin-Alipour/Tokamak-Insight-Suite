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
    if len(v)<5: raise ValueError('At least 5 finite I-V points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v)<5: raise ValueError('Voltage must contain at least 5 unique values.')
    if len(unique_v)<len(v):
        sums=np.bincount(inverse,weights=i)
        counts=np.bincount(inverse)
        v,i=unique_v,sums/counts
    dIdV=np.gradient(i,v)
    return {'n_points':int(len(v)),'voltage_min_V':float(v.min()),'voltage_max_V':float(v.max()),
            'current_min_A':float(i.min()),'current_max_A':float(i.max()),
            'floating_potential_interpolated_V':float(np.interp(0, i[np.argsort(i)], v[np.argsort(i)])) if (np.all(np.diff(i)>=0) or np.all(np.diff(i)<=0)) and (i.min()<=0<=i.max()) else None,
            'max_dIdV_A_per_V':float(np.max(dIdV)),'min_dIdV_A_per_V':float(np.min(dIdV)),
            'warning':'Descriptive summary only. Physical interpretation and temperature estimates require a justified probe regime/model; density and particle flux are not inferred.'}

# Expanded diagnostic workflows (v0.3): calculations operate on user-supplied,
# calibrated data. They do not invent device calibration or geometry.

def langmuir_iv_analysis(voltage: Any, current: Any,
                         electron_fit_range: tuple[float, float] | None = None,
                         ion_fit_range: tuple[float, float] | None = None,
                         current_convention: str = "electron_positive") -> dict[str, Any]:
    """Analyze a calibrated single Langmuir I-V sweep without geometry-dependent inversion.

    current_convention: "electron_positive" (default; ion saturation current is negative)
    or "ion_positive" (collected-ion current is positive, as in many probe circuits); the
    latter is flipped internally so the analysis always works in the electron-positive form.
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
    # Always provide explicitly labelled descriptive saturation-tail estimates.
    # These are fallbacks only; a user-supplied ion_fit_range overrides the ion tail.
    # Tail size is bounded to avoid letting a few endpoint samples dominate.
    tail_n = max(3, int(math.ceil(0.1 * len(v))))
    out["ion_saturation_current_A"] = float(np.median(i[:tail_n]))
    out["ion_saturation_current_method"] = "median_lowest_voltage_tail_fallback"
    out["electron_saturation_current_A"] = float(np.median(i[-tail_n:]))
    out["electron_saturation_current_method"] = "median_highest_voltage_tail_descriptive"
    # Find every zero-current crossing in voltage order. A zero-current plateau
    # is represented by its midpoint; multiple crossings are ambiguous and must
    # not be silently collapsed to the first crossing.
    exact_zero = (i == 0.0)
    candidates: list[float] = []
    zero_idx = np.flatnonzero(exact_zero)
    if zero_idx.size:
        groups = np.split(zero_idx, np.flatnonzero(np.diff(zero_idx) > 1) + 1)
        candidates.extend(float(0.5 * (v[g[0]] + v[g[-1]])) for g in groups)
    for k in range(len(i) - 1):
        if not exact_zero[k] and not exact_zero[k + 1] and i[k] * i[k + 1] < 0:
            candidates.append(float(v[k] - i[k] * (v[k + 1] - v[k]) / (i[k + 1] - i[k])))
    candidates.sort()
    unique_candidates: list[float] = []
    for candidate in candidates:
        if not unique_candidates or not np.isclose(candidate, unique_candidates[-1], rtol=1e-10, atol=1e-12):
            unique_candidates.append(candidate)
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
    out['max_dIdV_A_per_V']=float(np.max(didv)); out['plasma_potential_proxy_V']=float(v[np.argmax(didv)])
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
        p=np.polyfit(v[m],i[m],1); out['ion_saturation_linear_slope_A_per_V']=float(p[0]); out['ion_saturation_intercept_A']=float(p[1]); out['ion_saturation_fit_R2']=_r2(i[m],np.polyval(p,v[m]))
        out['ion_saturation_current_A']=float(np.median(i[m])); out['ion_saturation_current_method']='median_user_selected_ion_fit_range'
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
        out["electron_retarding_fit_status"] = (
            "fit_completed_review_R2_and_residuals"
            if np.isfinite(out["electron_retarding_fit_R2"])
            else "fit_quality_undefined"
        )
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
        out["electron_temperature_uncertainty_note"] = (
            "Local covariance from a joint offset-plus-exponential fit; excludes calibration, "
            "fit-window choice, sheath expansion and RF/noise systematics."
        )
    out['current_convention']=current_convention
    if 'ion_saturation_current_A' in out and out['ion_saturation_current_A']>0:
        out['sign_warning']='Ion-saturation current is positive in the electron-positive convention; check current_convention.'
    out['interpretation_warning']='Single-probe estimates depend on sweep direction, magnetic field, collisions, secondary emission, sheath expansion, surface condition and fit-window choice. Check fit R² and device-specific probe theory.'
    return out


def suggest_langmuir_fit_windows(
    voltage: Any, current: Any, current_convention: str = "electron_positive",
    *, min_points: int = 6, min_voltage_fraction: float = 0.12,
) -> dict[str, Any]:
    """Suggest reviewable I-V fit windows; suggestions do not validate physical regimes.

    Electron candidates are ranked using log(I - low-voltage-tail median) linearity.
    The low-voltage interval is only a candidate, not proof of ion saturation.
    """
    if current_convention not in {"electron_positive", "ion_positive"}:
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
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
    m=np.isfinite(v)&np.isfinite(i); v,i=v[m],i[m]
    if len(v)<2: return None
    o=np.argsort(v); v,i=v[o],i[o]; ix=np.where(i[:-1]*i[1:]<=0)[0]
    if not len(ix): return None
    k=ix[0]
    if i[k+1]==i[k]: return float((v[k]+v[k+1])/2)
    return float(v[k]-i[k]*(v[k+1]-v[k])/(i[k+1]-i[k]))

def triple_probe_temperature_eV(v_difference_V: float, model_factor: float = 1.0) -> dict[str, float | str]:
    """First-order triple-probe Te estimate under a user-selected idealized exponential-current model."""
    if (not _is_finite_real_scalar(v_difference_V) or v_difference_V <= 0
            or not _is_finite_real_scalar(model_factor) or model_factor <= 0):
        raise ValueError('Voltage difference and model factor must be positive and finite.')
    te=float(v_difference_V/(np.log(2)*model_factor))
    return {'electron_temperature_estimate_eV':te,'voltage_difference_V':float(v_difference_V),'model_factor':float(model_factor),'assumption':'Idealized triple-probe relation; confirm the actual electrode configuration, current balance, sheath model and circuit definition before use.'}

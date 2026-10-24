"""Electrostatic probe I-V analysis and temperature estimates."""
from __future__ import annotations
import math
from typing import Any
import numpy as np
from scipy import optimize
from .constants import E_CHARGE, AMU
from .spectral_analysis import _r2

def probe_iv_summary(voltage: Any, current: Any) -> dict[str, Any]:
    """Descriptive I-V preprocessing only; does not silently infer Te/ne without a validated regime model."""
    v=np.asarray(voltage,float).ravel(); i=np.asarray(current,float).ravel()
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
    good=np.isfinite(v)&np.isfinite(i); v,i=v[good],i[good]
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
            'warning':'Descriptive summary only. Plasma potential, Te and density require a justified probe regime/sheath model, collection geometry and uncertainty.'}

# Expanded diagnostic workflows (v0.3): calculations operate on user-supplied,
# calibrated data. They do not invent device calibration or geometry.

def langmuir_iv_analysis(voltage: Any, current: Any, electron_fit_range: tuple[float,float] | None = None,
                         ion_fit_range: tuple[float,float] | None = None, area_m2: float | None = None,
                         ion_mass_amu: float | None = None, ti_eV: float = 0.0,
                         sheath_factor: float = 0.61, gamma_i: float = 1.0,
                         current_convention: str = "electron_positive") -> dict[str, Any]:
    """Analyze a single Langmuir I-V sweep with explicit model-input validation.

    current_convention: "electron_positive" (default; ion saturation current is negative)
    or "ion_positive" (collected-ion current is positive, as in many probe circuits); the
    latter is flipped internally so the analysis always works in the electron-positive form.
    """
    if current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    for name, fit_range in (("electron_fit_range", electron_fit_range), ("ion_fit_range", ion_fit_range)):
        if fit_range is not None:
            try:
                limits = np.asarray(fit_range, dtype=float)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must contain exactly two finite voltage limits.") from exc
            if limits.shape != (2,) or not np.all(np.isfinite(limits)) or limits[1] <= limits[0]:
                raise ValueError(f"{name} must contain exactly two finite voltage limits in strictly increasing order.")
    if (area_m2 is None) != (ion_mass_amu is None):
        raise ValueError("area_m2 and ion_mass_amu must be supplied together for density estimation.")
    if area_m2 is not None and (not np.isfinite(area_m2) or area_m2 <= 0):
        raise ValueError("Collection area must be positive and finite.")
    if ion_mass_amu is not None and (not np.isfinite(ion_mass_amu) or ion_mass_amu <= 0):
        raise ValueError("Ion mass must be positive and finite.")
    if not np.all(np.isfinite([ti_eV, sheath_factor, gamma_i])) or ti_eV < 0 or sheath_factor <= 0 or gamma_i < 0:
        raise ValueError("ti_eV and gamma_i must be finite and nonnegative; sheath_factor must be positive and finite.")
    raw_v=np.asarray(voltage,float); raw_i=np.asarray(current,float)
    if raw_v.ndim != 1 or raw_i.ndim != 1:
        raise ValueError("Voltage and current inputs must be one-dimensional arrays.")
    v=raw_v.copy(); i=raw_i.copy()
    if len(v) != len(i): raise ValueError('Voltage and current arrays must have equal lengths.')
    if current_convention=="ion_positive": i=-i
    good=np.isfinite(v)&np.isfinite(i); v,i=v[good],i[good]
    if len(v)<8: raise ValueError('At least 8 valid voltage points are required.')
    order=np.argsort(v); v,i=v[order],i[order]
    unique_v, inverse = np.unique(v, return_inverse=True)
    if len(unique_v)<8: raise ValueError('At least 8 distinct voltage points are required after duplicate aggregation.')
    if len(unique_v)<len(v):
        sums=np.bincount(inverse,weights=i)
        counts=np.bincount(inverse)
        v,i=unique_v,sums/counts
    out=probe_iv_summary(v,i)
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
        if max(ion_fit_range) < min(v) or min(ion_fit_range) > max(v):
            raise ValueError("ion_fit_range must overlap the measured voltage sweep and contain at least 3 valid samples.")
        m=(v>=min(ion_fit_range))&(v<=max(ion_fit_range))
        if m.sum()>=3:
            p=np.polyfit(v[m],i[m],1); out['ion_saturation_linear_slope_A_per_V']=float(p[0]); out['ion_saturation_intercept_A']=float(p[1]); out['ion_saturation_fit_R2']=_r2(i[m],np.polyval(p,v[m]))
            out['ion_saturation_current_A']=float(np.median(i[m])); out['ion_saturation_current_method']='median_user_selected_ion_fit_range'
        else:
            raise ValueError("ion_fit_range must overlap the measured voltage sweep and contain at least 3 valid samples.")
    if electron_fit_range is not None:
        if max(electron_fit_range) <= min(v) or min(electron_fit_range) >= max(v):
            raise ValueError("electron_fit_range must overlap the measured voltage sweep.")
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
        first_positive = int(np.flatnonzero(positive)[0]) if np.any(positive) else 0
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
        out["ion_current_baseline_subtracted_A"] = baseline_fit_A
        out["electron_fit_points"] = int(np.count_nonzero(m))
        out["electron_retarding_log_slope_per_V"] = slope
        out["electron_retarding_fit_intercept"] = float(np.log(max(parameters[1] * current_scale, np.finfo(float).tiny)) - slope * x_ref)
        if slope > 0 and np.isfinite(slope):
            out["electron_temperature_eV"] = 1.0 / slope
            slope_se = float(np.sqrt(max(float(covariance[2, 2]), 0.0))) if covariance.shape == (3, 3) and np.isfinite(covariance[2, 2]) else None
            out["electron_temperature_standard_error_eV"] = (slope_se / slope**2) if slope_se is not None else None
        out["electron_temperature_uncertainty_note"] = (
            "Local covariance from a joint offset-plus-exponential fit; excludes calibration, "
            "fit-window choice, sheath expansion and RF/noise systematics."
        )
    if area_m2 is not None and ion_fit_range is None:
        out['ion_density_status'] = "not_estimated: provide a physically selected ion_fit_range containing at least three points; tail fallback is descriptive only"
    if area_m2 is not None and ion_fit_range is not None and 'ion_saturation_current_A' in out:
        te_value = out.get('electron_temperature_eV')
        if te_value is None or not np.isfinite(te_value) or te_value <= 0:
            raise ValueError("Ion-density estimation requires a valid positive electron temperature from the electron-retarding fit; refusing to substitute Te=0.")
        te=float(te_value)
        cs=math.sqrt(E_CHARGE*(ti_eV*gamma_i + te)/(ion_mass_amu*AMU))
        denom=sheath_factor*E_CHARGE*area_m2*cs
        out['ion_sound_speed_m_s']=cs
        out['ion_density_estimate_m-3']=abs(out['ion_saturation_current_A'])/denom if denom>0 else None
        out['ion_density_model']='n_i≈|I_isat|/(sheath_factor·e·A·c_s); depends on sheath model, geometry and ion temperature.'
    out['current_convention']=current_convention
    if 'ion_saturation_current_A' in out and out['ion_saturation_current_A']>0:
        out['sign_warning']='Ion-saturation current is positive in the electron-positive convention; check current_convention.'
    out['interpretation_warning']='Single-probe estimates depend on sweep direction, magnetic field, collisions, secondary emission, sheath expansion, surface condition and fit-window choice. Check fit R² and device-specific probe theory.'
    return out

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
    if not np.isfinite(v_difference_V) or v_difference_V<=0 or not np.isfinite(model_factor) or model_factor<=0:
        raise ValueError('Voltage difference and model factor must be positive and finite.')
    te=float(v_difference_V/(np.log(2)*model_factor))
    return {'electron_temperature_estimate_eV':te,'voltage_difference_V':float(v_difference_V),'model_factor':float(model_factor),'assumption':'Idealized triple-probe relation; confirm the actual electrode configuration, current balance, sheath model and circuit definition before use.'}

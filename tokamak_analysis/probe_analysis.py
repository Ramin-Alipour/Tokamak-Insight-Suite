"""Basic Langmuir, ball-pen and multi-pin probe data analysis."""
import numpy as np


def analyze_probe_voltage_current(voltage_V, current_A, *, electron_fit_range=None, ion_fit_range=None):
    """Summarize a swept single-probe I-V trace with explicit fit diagnostics.

    Voltage/current must already be calibrated and use a consistent current polarity.
    Returned saturation values are descriptive tail medians, not sheath-model fits.
    """
    v = np.asarray(voltage_V, dtype=float)
    i = np.asarray(current_A, dtype=float)
    if v.ndim != 1 or i.shape != v.shape or len(v) < 5:
        raise ValueError("V and I must be equal-length 1-D arrays with at least 5 points.")
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(i)):
        raise ValueError("V and I must contain only finite values.")
    order = np.argsort(v)
    v, i = v[order], i[order]
    if np.any(np.diff(v) <= 0):
        raise ValueError("Voltage values must be unique.")
    dvdi = np.gradient(i, v, edge_order=2)

    zero = i == 0.0
    candidates = []
    zero_idx = np.flatnonzero(zero)
    if zero_idx.size:
        groups = np.split(zero_idx, np.flatnonzero(np.diff(zero_idx) > 1) + 1)
        candidates.extend(float((v[g[0]] + v[g[-1]]) / 2) for g in groups)
    for k in range(len(i) - 1):
        if not zero[k] and not zero[k + 1] and i[k] * i[k + 1] < 0:
            candidates.append(float(v[k] - i[k] * (v[k + 1] - v[k]) / (i[k + 1] - i[k])))
    candidates.sort()
    unique_candidates = []
    for candidate in candidates:
        if not unique_candidates or not np.isclose(candidate, unique_candidates[-1], rtol=1e-10, atol=1e-12):
            unique_candidates.append(candidate)

    ion_mask = v <= np.quantile(v, 0.2)
    electron_mask = v >= np.quantile(v, 0.8)
    out = {
        "voltage_V": v, "current_A": i, "dI_dV_S": dvdi,
        "floating_potential_V": unique_candidates[0] if len(unique_candidates) == 1 else None,
        "floating_potential_candidates_V": unique_candidates,
        "floating_potential_status": ("unique_crossing" if len(unique_candidates) == 1 else
                                      "no_crossing" if not unique_candidates else "ambiguous_multiple_crossings"),
        "ion_saturation_current_A": float(np.median(i[ion_mask])),
        "electron_saturation_current_A": float(np.median(i[electron_mask])),
        "sample_count": int(len(v)),
        "voltage_span_V": float(v[-1] - v[0]),
        "current_span_A": float(np.ptp(i)),
        "monotonic_current_fraction": float(np.mean(np.diff(i) >= 0)),
        "caveat": ("Descriptive I-V metrics only. Tail medians are not validated saturation-region fits; "
                   "Te, plasma potential, and density require justified fit regions, current polarity, "
                   "probe geometry, sheath assumptions, and calibration."),
    }
    if electron_fit_range is not None:
        try:
            lo, hi = map(float, electron_fit_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("electron_fit_range must contain two finite increasing bounds.") from exc
        if not np.all(np.isfinite([lo, hi])) or hi <= lo:
            raise ValueError("electron_fit_range must contain two finite increasing bounds.")
        if lo < v[0] or hi > v[-1]:
            raise ValueError("electron_fit_range must lie within the measured voltage range.")
        mask = (v >= lo) & (v <= hi)
        points = int(np.count_nonzero(mask))
        out["electron_fit_points"] = points
        if points < 4:
            out["electron_temperature_estimate_eV"] = None
            out["electron_fit_status"] = "insufficient_positive_points"
        else:
            # Fit the retarding branch as I(V) = I_ion + A exp(V/Te).
            # A free offset avoids biasing Te when the selected voltage interval
            # does not reach a true ion-saturation plateau. This is a diagnostic
            # empirical fit, not a substitute for a validated sheath model.
            from scipy.optimize import curve_fit

            x, current = v[mask], i[mask]
            baseline0 = float(np.min(current))
            amplitude0 = max(float(np.ptp(current)), np.finfo(float).tiny)
            slope0 = 1.0 / max(float(hi - lo) / 3.0, np.finfo(float).eps)

            def exponential_branch(voltage, baseline, amplitude, slope):
                return baseline + amplitude * np.exp(slope * (voltage - lo))

            try:
                params, _ = curve_fit(
                    exponential_branch, x, current,
                    p0=(baseline0 - amplitude0 * 0.05, amplitude0, slope0),
                    bounds=([-np.inf, 0.0, np.finfo(float).eps],
                            [np.inf, np.inf, np.inf]),
                    maxfev=20000,
                )
                baseline, amplitude, slope = map(float, params)
                fitted = exponential_branch(x, *params)
                residual = current - fitted
                ss_res = float(np.sum(residual ** 2))
                ss_tot = float(np.sum((current - np.mean(current)) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
                out["electron_temperature_estimate_eV"] = float(1.0 / slope)
                out["electron_fit_status"] = "ok"
                out["electron_fit_slope_per_V"] = slope
                out["electron_fit_intercept"] = float(np.log(amplitude) - slope * lo)
                out["electron_fit_baseline_A"] = baseline
                out["electron_fit_r_squared"] = float(r_squared)
                out["electron_current_used_A"] = current - baseline
                out["electron_fit_note"] = (
                    "Nonlinear exponential fit with a free current offset over the selected interval. "
                    "R² is descriptive; verify fit region, current polarity, saturation behavior, "
                    "and sheath-model assumptions against raw I-V data."
                )
            except (RuntimeError, ValueError, FloatingPointError):
                out["electron_temperature_estimate_eV"] = None
                out["electron_fit_status"] = "fit_failed"
    if ion_fit_range is not None:
        try:
            lo, hi = map(float, ion_fit_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("ion_fit_range must contain two finite increasing bounds.") from exc
        if not np.all(np.isfinite([lo, hi])) or hi <= lo:
            raise ValueError("ion_fit_range must contain two finite increasing bounds.")
        if lo < v[0] or hi > v[-1]:
            raise ValueError("ion_fit_range must lie within the measured voltage range.")
        mask = (v >= lo) & (v <= hi)
        if np.count_nonzero(mask) < 2:
            raise ValueError("ion_fit_range must include at least two measured points.")
        slope, intercept = np.polyfit(v[mask], i[mask], 1)
        fitted = slope * v[mask] + intercept
        ss_res = float(np.sum((i[mask] - fitted) ** 2))
        ss_tot = float(np.sum((i[mask] - np.mean(i[mask])) ** 2))
        out["ion_fit_linear_slope_A_V"] = float(slope)
        out["ion_fit_linear_intercept_A"] = float(intercept)
        out["ion_fit_points"] = int(np.count_nonzero(mask))
        out["ion_fit_r_squared"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else 0.0)
    return out


def electron_retarding_current(V_probe_V, plasma_potential_V, electron_saturation_current_A, electron_temperature_eV):
    """Maxwellian electron-retarding branch: Ie=Ies*exp((V-Vp)/Te[eV])."""
    v = np.asarray(V_probe_V, dtype=float)
    vp, ies, te = float(plasma_potential_V), float(electron_saturation_current_A), float(electron_temperature_eV)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite([vp, ies, te])) or ies <= 0 or te <= 0:
        raise ValueError("Inputs must be finite; Ies and Te must be positive.")
    if np.any(v > vp):
        raise ValueError("This relation applies only to the electron-retarding region V <= Vp.")
    current = ies * np.exp((v - vp) / te)
    return {"electron_current_A": current, "equation": "Ie=Ies*exp((Vprobe-Vplasma)/Te_eV)",
            "assumption": "Ideal Maxwellian electron-retarding branch; sheath geometry, collisions, and ion current are not included."}

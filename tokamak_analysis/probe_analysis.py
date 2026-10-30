"""Basic Langmuir, ball-pen and multi-pin probe data analysis."""
import numpy as np


def analyze_probe_voltage_current(
    voltage_V, current_A, *, electron_fit_range=None, ion_fit_range=None,
    current_convention="electron_positive",
):
    """Compatibility adapter to the canonical calibrated Langmuir I-V pipeline.

    Eight or more distinct samples use the canonical analysis. Short legacy
    sweeps (four to seven points) retain the historical summary-only behavior;
    they cannot be used for the canonical fit/density workflow.
    """
    raw_v, raw_i = np.asarray(voltage_V), np.asarray(current_A)
    if raw_v.ndim != 1 or raw_i.ndim != 1:
        raise ValueError("Voltage and current must be one-dimensional arrays.")
    if raw_v.shape != raw_i.shape or raw_v.dtype.kind not in "iuf" or raw_i.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must be equal-length real numeric arrays.")
    if not isinstance(current_convention, str) or current_convention not in {"electron_positive", "ion_positive"}:
        raise ValueError("current_convention must be 'electron_positive' or 'ion_positive'.")
    v, i = raw_v.astype(float), raw_i.astype(float)
    if not np.all(np.isfinite(v)) or not np.all(np.isfinite(i)):
        raise ValueError("Voltage and current must contain only finite values.")
    if current_convention == "ion_positive":
        i = -i
    if len(v) != len(i) or len(v) < 5 or len(np.unique(v)) != len(v):
        raise ValueError("At least five paired samples with unique voltages are required.")
    order = np.argsort(v)
    v, i = v[order], i[order]

    # Keep the canonical pipeline as the single implementation for supported
    # full sweeps. This small-sweep branch exists only for old summary callers.
    if len(v) >= 8:
        from tokamak_analysis.langmuir_probe import analyze_langmuir_probe
        result = analyze_langmuir_probe(
            v, i, electron_fit_range=electron_fit_range,
            ion_fit_range=ion_fit_range, current_convention="electron_positive",
        )
    else:
        from tis.probe_analysis import probe_iv_summary
        for name, fit_range in (("electron_fit_range", electron_fit_range), ("ion_fit_range", ion_fit_range)):
            if fit_range is not None:
                if not isinstance(fit_range, (tuple, list)) or len(fit_range) != 2:
                    raise ValueError(f"{name} must contain two voltage limits inside the measured voltage range.")
                lo, hi = fit_range
                if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
                    raise ValueError(f"{name} must contain two increasing voltage limits inside the measured voltage range.")
                if lo < v[0] or hi > v[-1]:
                    raise ValueError(f"{name} must be inside the measured voltage range.")
                if name == "ion_fit_range" and np.count_nonzero((v >= lo) & (v <= hi)) < 2:
                    raise ValueError("ion_fit_range must contain at least two measured points.")
        result = probe_iv_summary(v, i)
        result["analysis_scope"] = "descriptive_short_sweep_compatibility_only"
        result["floating_potential_V"] = result["floating_potential_interpolated_V"]
        result["floating_potential_status"] = {
            "single_zero_crossing": "unique_crossing",
            "no_zero_crossing_in_measured_sweep": "no_crossing",
            "ambiguous_multiple_zero_crossings": "ambiguous_multiple_crossings",
        }[result["floating_potential_status"]]
        result.update({
            "processed_voltage_V": v.tolist(),
            "processed_current_A": i.tolist(),
            "voltage_V": v.copy(),
            "current_A": i.copy(),
            "dI_dV_S": np.gradient(i, v),
            "dIdV_A_per_V": np.gradient(i, v).tolist(),
            "sample_count": int(len(v)),
            "n_points": int(len(v)),
            "voltage_span_V": float(v[-1] - v[0]),
            "current_span_A": float(np.ptp(i)),
            "electron_temperature_eV": None,
            "electron_temperature_estimate_eV": None,
            "electron_fit_status": "not_requested",
            "electron_fit_points": 0,
            "electron_fit_r_squared": None,
            "electron_fit_residual_rms_A": None,
            "electron_fit_quality": None,
            "electron_temperature_standard_uncertainty_eV": None,
            "current_convention_input": current_convention,
            "current_convention": "electron_positive",
        })
        if electron_fit_range is not None:
            mask = (v >= electron_fit_range[0]) & (v <= electron_fit_range[1])
            x, y = v[mask], i[mask]
            baseline = float(np.median(i[:max(3, int(np.ceil(0.1 * len(i))))]))
            positive = y - baseline > 0
            result["electron_fit_points"] = int(np.count_nonzero(positive))
            if np.count_nonzero(positive) < 3:
                result["electron_fit_status"] = "insufficient_positive_points"
            else:
                slope, intercept = np.polyfit(x[positive], np.log(y[positive] - baseline), 1)
                if not np.isfinite(slope) or slope <= 0:
                    result["electron_fit_status"] = "invalid_exponential_slope"
                else:
                    fitted = baseline + np.exp(intercept + slope * x[positive])
                    residual = y[positive] - fitted
                    ss_tot = float(np.sum((y[positive] - np.mean(y[positive])) ** 2))
                    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else None
                    rms = float(np.sqrt(np.mean(residual ** 2)))
                    te = 1.0 / float(slope)
                    result.update({
                        "electron_temperature_eV": te,
                        "electron_temperature_estimate_eV": te,
                        "electron_fit_status": "ok",
                        "electron_fit_r_squared": r2,
                        "electron_fit_residual_rms_A": rms,
                        "electron_fit_quality": "strong_empirical_fit" if r2 is not None and r2 >= 0.99 else "weak_empirical_fit",
                        "electron_temperature_standard_uncertainty_eV": None,
                    })
    result.update({
        "voltage_V": np.asarray(result["processed_voltage_V"], dtype=float),
        "current_A": np.asarray(result["processed_current_A"], dtype=float),
        "dI_dV_S": np.asarray(result["dIdV_A_per_V"], dtype=float),
        "floating_potential_status_legacy": {
            "single_zero_crossing": "unique_crossing",
            "no_zero_crossing_in_measured_sweep": "no_crossing",
            "ambiguous_multiple_zero_crossings": "ambiguous_multiple_crossings",
        }.get(result.get("floating_potential_status"), result.get("floating_potential_status")),
        "sample_count": result.get("n_points", len(v)),
        "voltage_span_V": result.get("voltage_max_V", float(v[-1])) - result.get("voltage_min_V", float(v[0])),
        "current_span_A": result.get("current_max_A", float(np.max(i))) - result.get("current_min_A", float(np.min(i))),
        "electron_temperature_estimate_eV": result.get("electron_temperature_eV"),
        "electron_fit_status": result.get("electron_fit_status", "not_requested" if electron_fit_range is None else result.get("electron_retarding_fit_status")),
        "electron_fit_r_squared": result.get("electron_fit_r_squared", result.get("electron_retarding_fit_R2")),
        "electron_fit_residual_rms_A": result.get("electron_fit_residual_rms_A", result.get("electron_retarding_fit_residual_rms_A")),
        "electron_fit_quality": result.get("electron_fit_quality", (
            "strong_empirical_fit" if result.get("electron_retarding_fit_R2") is not None and result["electron_retarding_fit_R2"] >= 0.99
            else "weak_empirical_fit" if result.get("electron_retarding_fit_R2") is not None else None
        )),
        "electron_temperature_standard_uncertainty_eV": result.get("electron_temperature_total_standard_uncertainty_eV", result.get("electron_temperature_standard_error_eV")),
        "ion_fit_linear_slope_A_V": result.get("ion_saturation_linear_slope_A_per_V"),
        "ion_fit_linear_intercept_A": result.get("ion_saturation_intercept_A"),
        "ion_fit_points": result.get("ion_saturation_fit_points"),
        "ion_fit_r_squared": result.get("ion_saturation_fit_R2"),
    })
    return result

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

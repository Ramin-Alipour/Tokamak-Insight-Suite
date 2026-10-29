"""Advanced calibrated Langmuir-probe analysis extensions.

All functions consume already-calibrated, ordered or orderable voltage/current
samples. They do not perform instrument calibration or experimental validation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import curve_fit


def fit_langmuir_electron_models(
    voltage_V: Any,
    current_A: Any,
    fit_range_V: tuple[float, float],
    *,
    model: str = "single_maxwellian",
) -> dict[str, Any]:
    """Fit a single-temperature or two-temperature exponential electron branch.

    The two-temperature model is an empirical mixture of two exponentials. It is
    not proof of a physical bi-Maxwellian distribution; component identifiability
    must be assessed from residuals, covariance, and the selected voltage range.
    """
    v0, i0 = np.asarray(voltage_V), np.asarray(current_A)
    if v0.ndim != 1 or i0.ndim != 1 or v0.shape != i0.shape:
        raise ValueError("Voltage and current must be equal-length one-dimensional arrays.")
    if v0.dtype.kind not in "iuf" or i0.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must be real numeric arrays.")
    v, i = v0.astype(float), i0.astype(float)
    finite = np.isfinite(v) & np.isfinite(i)
    v, i = v[finite], i[finite]
    if len(v) < 8:
        raise ValueError("At least eight finite paired samples are required.")
    order = np.argsort(v)
    v, i = v[order], i[order]
    if not isinstance(fit_range_V, (tuple, list)) or len(fit_range_V) != 2:
        raise ValueError("fit_range_V must contain two increasing voltage bounds.")
    lo, hi = map(float, fit_range_V)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo >= v[0] and hi <= v[-1]):
        raise ValueError("fit_range_V must be finite, increasing, and inside the measured voltage range.")
    mask = (v >= lo) & (v <= hi)
    x, y = v[mask], i[mask]
    min_count = 4 if model == "single_maxwellian" else 8
    if model not in {"single_maxwellian", "two_temperature"}:
        raise ValueError("model must be 'single_maxwellian' or 'two_temperature'.")
    if len(x) < min_count:
        raise ValueError(f"The selected model requires at least {min_count} samples in the fit window.")
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
            pars = np.asarray([pars[0], pars[3], pars[4], pars[1], pars[2]])
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
    return {
        "model": model,
        "fit_range_V": [lo, hi],
        "fit_points": int(len(x)),
        "electron_temperature_components_eV": temperatures,
        "component_amplitudes_A": amps,
        "fit_R2": r2,
        "fit_residual_rms_A": rms,
        "fit_normalized_rmse": nrmse,
        "covariance_finite": bool(np.all(np.isfinite(cov))),
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
    ion_saturation_current_A: float,
    electron_temperature_eV: float,
    probe_area_m2: float,
    species: Sequence[Mapping[str, float]],
) -> dict[str, Any]:
    """Estimate total ion density from supplied species fractions and sheath coefficients.

    Each species item must contain mass_kg, fraction, and sheath_coefficient.
    Fractions must sum to one. The formula assumes a common electron temperature
    and an explicitly supplied species-resolved ion-saturation current model.
    """
    from tis.constants import E_CHARGE

    vals = [ion_saturation_current_A, electron_temperature_eV, probe_area_m2]
    if not all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x)) for x in vals):
        raise ValueError("Current, temperature, and area must be finite real scalars.")
    if ion_saturation_current_A >= 0 or electron_temperature_eV <= 0 or probe_area_m2 <= 0:
        raise ValueError("Ion current must be negative in electron-positive convention; temperature and area must be positive.")
    if not species:
        raise ValueError("At least one ion species is required.")
    normalized = []
    for item in species:
        if not all(k in item for k in ("mass_kg", "fraction", "sheath_coefficient")):
            raise ValueError("Each species requires mass_kg, fraction, and sheath_coefficient.")
        mass, fraction, coeff = (float(item[k]) for k in ("mass_kg", "fraction", "sheath_coefficient"))
        if not all(math.isfinite(z) and z > 0 for z in (mass, fraction, coeff)):
            raise ValueError("Species mass, fraction, and sheath coefficient must be positive and finite.")
        normalized.append({"mass_kg": mass, "fraction": fraction, "sheath_coefficient": coeff})
    total_fraction = sum(x["fraction"] for x in normalized)
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Ion species fractions must sum to 1.")
    factor = sum(x["fraction"] * x["sheath_coefficient"] / math.sqrt(x["mass_kg"]) for x in normalized)
    denominator = E_CHARGE * probe_area_m2 * math.sqrt(E_CHARGE * electron_temperature_eV) * factor
    density = abs(ion_saturation_current_A) / denominator
    if not math.isfinite(density) or density <= 0:
        raise ValueError("Multispecies density estimate is not finite and positive.")
    return {
        "ion_density_estimate_m-3": density,
        "ion_species": normalized,
        "ion_species_fraction_sum": total_fraction,
        "model": "fraction-weighted_multi-species_sheath_current",
        "assumptions": ["common electron temperature", "supplied species fractions and sheath coefficients are appropriate", "single effective collection area"],
        "physical_regime_confirmed": False,
        "uncertainty_propagation_performed": False,
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
    electron_saturation_current_A: float,
    electron_temperature_eV: float,
    probe_area_m2: float,
) -> dict[str, Any]:
    """Estimate electron density from positive electron saturation current.

    Uses I_e,sat = e A n_e sqrt(e T_e / (2 pi m_e)); the user must supply an
    effective collection area appropriate to the probe regime and geometry.
    """
    from tis.constants import E_CHARGE, M_E

    vals = (electron_saturation_current_A, electron_temperature_eV, probe_area_m2)
    if not all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x)) for x in vals):
        raise ValueError("Current, temperature, and area must be finite real scalars.")
    if electron_saturation_current_A <= 0 or electron_temperature_eV <= 0 or probe_area_m2 <= 0:
        raise ValueError("Electron saturation current, temperature, and area must be positive.")
    thermal_speed_factor = math.sqrt(E_CHARGE * electron_temperature_eV / (2.0 * math.pi * M_E))
    density = electron_saturation_current_A / (E_CHARGE * probe_area_m2 * thermal_speed_factor)
    return {
        "electron_density_estimate_m-3": density,
        "formula": "n_e = I_e,sat / (e A_p sqrt(e T_e / (2 pi m_e)))",
        "assumption": "Maxwellian electron collection with the supplied effective area and applicable electron-saturation regime.",
        "physical_regime_confirmed": False,
        "uncertainty_propagation_performed": False,
    }


def estimate_ion_particle_flux(
    ion_density_m3: float,
    electron_temperature_eV: float,
    ion_mass_kg: float,
    *,
    sheath_factor: float = 1.0,
) -> dict[str, Any]:
    """Estimate Bohm-like ion particle flux from supplied density and Te."""
    from tis.constants import E_CHARGE

    vals = (ion_density_m3, electron_temperature_eV, ion_mass_kg, sheath_factor)
    if not all(isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_)) and math.isfinite(float(x)) for x in vals):
        raise ValueError("Density, temperature, mass, and sheath_factor must be finite real scalars.")
    if ion_density_m3 <= 0 or electron_temperature_eV <= 0 or ion_mass_kg <= 0 or sheath_factor <= 0:
        raise ValueError("Density, temperature, mass, and sheath_factor must be positive.")
    sound_speed = math.sqrt(E_CHARGE * electron_temperature_eV / ion_mass_kg)
    flux = float(sheath_factor) * float(ion_density_m3) * sound_speed
    return {
        "ion_sound_speed_m_s": sound_speed,
        "ion_particle_flux_m-2_s-1": flux,
        "formula": "Gamma_i = sheath_factor * n_i * sqrt(e T_e / m_i)",
        "assumption": "Bohm-like flux estimate; sheath_factor and the applicability of the selected edge/sheath model are supplied by the user.",
        "physical_regime_confirmed": False,
        "uncertainty_propagation_performed": False,
    }


def estimate_smoothed_plasma_potential_candidates(
    voltage_V: Any,
    current_A: Any,
    *,
    window_length: int = 7,
    polynomial_order: int = 2,
) -> dict[str, Any]:
    """Compare raw-gradient and explicitly Savitzky-Golay-smoothed derivative candidates."""
    from scipy.signal import savgol_filter

    v0, i0 = np.asarray(voltage_V), np.asarray(current_A)
    if v0.ndim != 1 or i0.ndim != 1 or v0.shape != i0.shape:
        raise ValueError("Voltage and current must be equal-length one-dimensional arrays.")
    if v0.dtype.kind not in "iuf" or i0.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must be real numeric arrays.")
    finite = np.isfinite(v0) & np.isfinite(i0)
    v, i = v0[finite].astype(float), i0[finite].astype(float)
    order = np.argsort(v)
    v, i = v[order], i[order]
    unique, inv = np.unique(v, return_inverse=True)
    if len(unique) != len(v):
        sums, counts = np.bincount(inv, weights=i), np.bincount(inv)
        v, i = unique, sums / counts
    if len(v) < 5 or not np.all(np.diff(v) > 0):
        raise ValueError("At least five distinct finite voltage values are required.")
    if isinstance(window_length, bool) or not isinstance(window_length, int) or window_length < 5 or window_length % 2 == 0 or window_length > len(v):
        raise ValueError("window_length must be an odd integer between 5 and the number of samples.")
    if isinstance(polynomial_order, bool) or not isinstance(polynomial_order, int) or polynomial_order < 1 or polynomial_order >= window_length:
        raise ValueError("polynomial_order must be an integer in [1, window_length).")
    raw_didv = np.gradient(i, v)
    raw_d2 = np.gradient(raw_didv, v)
    smooth_i = savgol_filter(i, window_length=window_length, polyorder=polynomial_order, mode="interp")
    smooth_didv = np.gradient(smooth_i, v)
    smooth_d2 = np.gradient(smooth_didv, v)
    return {
        "raw_max_dIdV_candidate_V": float(v[int(np.argmax(raw_didv))]),
        "raw_max_d2IdV2_candidate_V": float(v[int(np.argmax(raw_d2))]),
        "smoothed_max_dIdV_candidate_V": float(v[int(np.argmax(smooth_didv))]),
        "smoothed_max_d2IdV2_candidate_V": float(v[int(np.argmax(smooth_d2))]),
        "smoothing_method": "Savitzky-Golay",
        "window_length_samples": window_length,
        "polynomial_order": polynomial_order,
        "smoothed_current_A": smooth_i.tolist(),
        "physical_plasma_potential_confirmed": False,
        "interpretation": "Compare candidates and parameter sensitivity; smoothing is explicit and does not certify the physical plasma potential.",
    }


def fit_langmuir_stretched_exponential(
    voltage_V: Any,
    current_A: Any,
    fit_range_V: tuple[float, float],
) -> dict[str, Any]:
    """Fit a generalized stretched-exponential shape as a non-Maxwellian-shape diagnostic.

    The shape exponent is empirical; this is not a kinetic kappa-distribution model
    and must not be interpreted as a unique non-Maxwellian electron distribution.
    """
    v0, i0 = np.asarray(voltage_V), np.asarray(current_A)
    if v0.ndim != 1 or i0.ndim != 1 or v0.shape != i0.shape:
        raise ValueError("Voltage and current must be equal-length one-dimensional arrays.")
    if v0.dtype.kind not in "iuf" or i0.dtype.kind not in "iuf":
        raise ValueError("Voltage and current must be real numeric arrays.")
    finite = np.isfinite(v0) & np.isfinite(i0)
    v, i = v0[finite].astype(float), i0[finite].astype(float)
    order = np.argsort(v)
    v, i = v[order], i[order]
    if len(v) < 8:
        raise ValueError("At least eight finite paired samples are required.")
    lo, hi = map(float, fit_range_V)
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi and lo >= v[0] and hi <= v[-1]):
        raise ValueError("fit_range_V must be finite, increasing, and inside the measured voltage range.")
    mask = (v >= lo) & (v <= hi)
    x, y = v[mask], i[mask]
    if len(x) < 5:
        raise ValueError("Stretched-exponential fit requires at least five samples.")
    xref = float(x[0])
    scale = max(float(np.ptp(y)), float(np.max(np.abs(y))), np.finfo(float).tiny)
    def fn(xx, offset, amplitude, slope, shape):
        z = np.maximum(slope * (np.asarray(xx) - xref), 0.0)
        return offset + amplitude * np.exp(np.clip(np.power(z, shape), -700.0, 700.0))
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
    return {
        "model": "stretched_exponential_empirical_shape",
        "fit_range_V": [lo, hi],
        "fit_points": int(len(x)),
        "shape_exponent": float(pars[3]),
        "scale_slope_per_V": float(pars[2]),
        "fit_R2": (1.0 - ss_res / ss_tot if ss_tot > 0 else None),
        "fit_residual_rms_A": float(np.sqrt(np.mean(residual ** 2))),
        "covariance_finite": bool(np.all(np.isfinite(cov))),
        "kinetic_distribution_identified": False,
        "interpretation": "Empirical curve-shape alternative only; not a kappa model or proof of a physical non-Maxwellian distribution.",
    }

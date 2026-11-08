"""Signal repair and conditioning utilities.

The core FFT/filter routines deliberately reject irregular or gappy time bases.
This module supplies the *explicit* repair step that error message asks for, and
records exactly what was changed so that provenance stays honest.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from .core import prepare_time_series


def sampling_audit(time_s: Any) -> dict[str, Any]:
    """Describe sampling regularity: median/mean dt, jitter and the location of gaps."""
    t = np.asarray(time_s, dtype=float).ravel()
    if not np.all(np.isfinite(t)):
        n_invalid = int(np.count_nonzero(~np.isfinite(t)))
        raise ValueError(
            f"Timestamp array contains {n_invalid} non-finite value(s); "
            "resolve missing/invalid timestamps before auditing sampling regularity."
        )
    if len(t) < 3:
        raise ValueError("At least three finite timestamps are required.")
    dt = np.diff(t)
    if np.any(dt <= 0):
        n_bad = int(np.count_nonzero(dt <= 0))
        return {"monotonic": False, "n_nonpositive_intervals": n_bad,
                "note": "Timestamps are not strictly increasing; sort/deduplicate before analysis."}
    med = float(np.median(dt))
    gap_idx = np.flatnonzero(dt > 1.5 * med)
    return {
        "monotonic": True, "n_samples": int(len(t)), "median_dt_s": med, "mean_dt_s": float(np.mean(dt)),
        "median_sampling_rate_Hz": 1.0 / med, "jitter_std_over_mean": float(np.std(dt) / np.mean(dt)),
        "n_gaps": int(len(gap_idx)), "largest_gap_s": float(np.max(dt)),
        "largest_gap_over_median": float(np.max(dt) / med),
        "gap_start_times_s": [float(t[i]) for i in gap_idx[:50]],
        "gap_durations_s": [float(dt[i]) for i in gap_idx[:50]],
    }


def hampel_despike(values: Any, window: int = 11, n_sigmas: float = 6.0) -> dict[str, Any]:
    """Replace isolated spikes (|x - rolling median| > n_sigmas * 1.4826 * rolling MAD) by the median.

    Use only for glitches/pickup spikes. ELMs and other real transients are *physical*
    and will be removed too if the window is short, so inspect `n_replaced` and `indices`.
    The window must be much shorter than the fastest signal period (<~1/10 of it); otherwise the
    median filter itself distorts the signal and curvature is mistaken for spikes.
    """
    x = np.asarray(values, dtype=float).ravel()
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")
    if isinstance(n_sigmas, (bool, np.bool_)) or not isinstance(n_sigmas, (int, float, np.integer, np.floating)) or not np.isfinite(n_sigmas) or n_sigmas <= 0:
        raise ValueError("n_sigmas must be a positive finite real number.")
    if len(x) < window or not np.all(np.isfinite(x)):
        raise ValueError("Signal must be finite and at least one window long.")
    med = ndimage.median_filter(x, size=window, mode="nearest")
    mad = ndimage.median_filter(np.abs(x - med), size=window, mode="nearest")
    scale = 1.4826 * mad
    # Global noise floor from the whole-record residual: on a very smooth signal the *local* MAD
    # collapses to ~0 and the median filter's curvature bias would be misread as spikes.
    global_scale = 1.4826 * float(np.median(np.abs((x - med) - np.median(x - med))))
    # For (nearly) noise-free data no noise scale exists; the median filter's curvature bias at
    # extrema is then the only residual. Never flag deviations below 0.05 % of the dynamic range.
    dynamic = float(np.percentile(x, 99) - np.percentile(x, 1))
    floor = max(global_scale, 5e-4 * dynamic, np.finfo(float).tiny)
    bad = np.abs(x - med) > n_sigmas * np.maximum(scale, floor)
    out = x.copy(); out[bad] = med[bad]
    return {"values": out, "n_replaced": int(bad.sum()), "indices": np.flatnonzero(bad),
            "method": f"Hampel filter, window={window}, threshold={n_sigmas}*1.4826*MAD"}


def noise_floor_detection_limit(
    background_values: Any,
    false_alarm_probability: float = 1e-3,
    two_sided: bool = False,
) -> dict[str, Any]:
    """Estimate a noise floor and per-sample detection threshold from background-only samples.

    The robust noise scale is 1.4826*MAD. The empirical threshold is taken from
    the observed residual distribution, so Gaussianity is not required for that
    threshold. The optional Gaussian threshold is a reference only. Correlation,
    instrument bandwidth, averaging and multiple comparisons are not inferred;
    choose a background-only interval and interpret thresholds for the acquisition.
    """
    x = np.asarray(background_values, dtype=float).ravel()
    if len(x) < 8:
        raise ValueError("At least 8 background samples are required.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Background samples must contain only finite values.")
    if (isinstance(false_alarm_probability, (bool, np.bool_))
            or not isinstance(false_alarm_probability, (int, float, np.integer, np.floating))
            or not np.isfinite(false_alarm_probability)
            or not 0.0 < false_alarm_probability < 0.5):
        raise ValueError("false_alarm_probability must be a finite real number strictly between 0 and 0.5.")
    if not isinstance(two_sided, (bool, np.bool_)):
        raise ValueError("two_sided must be a boolean.")
    alpha = float(false_alarm_probability)
    baseline = float(np.median(x))
    residual = x - baseline
    mad = float(np.median(np.abs(residual - np.median(residual))))
    sigma = 1.4826 * mad
    degenerate = sigma <= np.finfo(float).eps * max(float(np.max(np.abs(x))), 1.0)
    if degenerate:
        sigma = float(np.std(residual, ddof=1))
    tail = np.abs(residual) if two_sided else residual
    empirical_threshold = max(0.0, float(np.quantile(tail, 1.0 - alpha)))
    from scipy.stats import norm
    gaussian_probability = 1.0 - alpha / 2.0 if two_sided else 1.0 - alpha
    gaussian_threshold = float(norm.ppf(gaussian_probability) * sigma)
    return {
        "baseline": baseline,
        "noise_floor_sigma": float(sigma),
        "empirical_detection_threshold_above_baseline": empirical_threshold,
        "gaussian_reference_threshold_above_baseline": gaussian_threshold,
        "false_alarm_probability_per_sample": alpha,
        "two_sided": bool(two_sided),
        "n_background_samples": int(len(x)),
        "noise_estimate_degenerate": bool(degenerate),
        "method": "median baseline; 1.4826*MAD robust scale; empirical residual quantile",
        "note": "Use background-only samples. Threshold is per sample and does not account for bandwidth, temporal correlation, averaging or multiple comparisons; no device calibration is inferred.",
    }


def detrend_polynomial(time_s: Any, values: Any, order: int = 1,
                       fit_window_s: tuple[float, float] | None = None) -> dict[str, Any]:
    """Subtract a polynomial trend fitted (optionally) only inside a quiet window."""
    t = np.asarray(time_s, dtype=float).ravel()
    y = np.asarray(values, dtype=float).ravel()
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, float, np.integer, np.floating)) or not np.isfinite(order) or int(order) != order or not 0 <= int(order) <= 8:
        raise ValueError("order must be an integer in [0, 8].")
    order = int(order)
    if len(t) != len(y):
        raise ValueError("Time and signal must have equal lengths.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)):
        raise ValueError("Time and signal must contain only finite values.")
    if len(t) < order + 2:
        raise ValueError("Too few samples for the requested polynomial order.")
    window = None
    m = np.ones(len(t), bool)
    if fit_window_s is not None:
        try:
            window = np.asarray(fit_window_s, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("fit_window_s must contain exactly two finite time bounds.") from exc
        if window.shape != (2,) or not np.all(np.isfinite(window)):
            raise ValueError("fit_window_s must contain exactly two finite time bounds.")
        lo, hi = float(np.min(window)), float(np.max(window))
        m = (t >= lo) & (t <= hi)
        if m.sum() < order + 2:
            raise ValueError("Fit window contains too few samples for the polynomial order.")
    tc = t - t[m].mean()
    coeff = np.polyfit(tc[m], y[m], order)
    trend = np.polyval(coeff, tc)
    return {"values": y - trend, "trend": trend, "coefficients_high_to_low_centered_time": coeff.tolist(),
            "order": order, "fit_window_s": None if window is None else [float(v) for v in window]}


def resample_uniform(time_s: Any, values: Any, fs: float | None = None,
                     max_interp_gap_s: float | None = None,
                     max_output_samples: int = 2_000_000) -> dict[str, Any]:
    """Linearly interpolate onto a uniform grid, without inventing data inside long gaps.

    Samples falling in a gap longer than `max_interp_gap_s` (default: 3 median intervals)
    are returned as NaN and flagged; nothing is silently filled. Linear interpolation
    low-passes content near Nyquist, so prefer `longest_contiguous_segment` for spectral work.
    """
    t = np.asarray(time_s, dtype=float).ravel(); y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y):
        raise ValueError("Time and signal must have equal lengths.")
    n_input_raw = len(t)
    good = np.isfinite(t) & np.isfinite(y)
    n_nonfinite_dropped = int(np.count_nonzero(~good))
    t, y = t[good], y[good]
    if len(t) < 8:
        raise ValueError("At least 8 finite samples are required.")
    order = np.argsort(t, kind="stable"); t, y = t[order], y[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Duplicate timestamps must be resolved before resampling.")
    med = float(np.median(np.diff(t)))
    if fs is None:
        fs = 1.0 / med
    if (isinstance(fs, (bool, np.bool_))
            or not isinstance(fs, (int, float, np.integer, np.floating))
            or not np.isfinite(fs) or fs <= 0):
        raise ValueError("fs must be a positive finite real number.")
    if (isinstance(max_output_samples, (bool, np.bool_))
            or not isinstance(max_output_samples, (int, np.integer))
            or max_output_samples < 8):
        raise ValueError("max_output_samples must be an integer >= 8.")
    if max_interp_gap_s is None:
        limit = 3.0 * med
    else:
        if isinstance(max_interp_gap_s, (bool, np.bool_)):
            raise ValueError("max_interp_gap_s must be a positive finite real number.")
        try:
            limit = float(max_interp_gap_s)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("max_interp_gap_s must be a positive finite real number.") from exc
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("max_interp_gap_s must be a positive finite real number.")
    span = float(t[-1] - t[0])
    if not np.isfinite(span) or span < 0:
        raise ValueError("Time span must be finite and non-negative.")
    # Bound allocations before constructing the grid: a mistaken fs or tiny
    # timestamp interval must not allocate an unbounded array.
    estimated_samples = span * float(fs)
    if not np.isfinite(estimated_samples) or estimated_samples > max_output_samples:
        raise ValueError(
            f"Uniform grid would exceed the {max_output_samples:,}-sample safety limit; "
            "reduce fs or process a shorter time window."
        )
    step = 1.0 / float(fs)
    endpoint_tol = 1.0 * max(
        abs(float(np.spacing(t[0]))), abs(float(np.spacing(t[-1]))),
        abs(float(np.spacing(step))), np.finfo(float).tiny,
    )
    n_grid = int(np.floor(estimated_samples + endpoint_tol * float(fs))) + 1
    if n_grid > max_output_samples:
        raise ValueError(
            f"Uniform grid would exceed the {max_output_samples:,}-sample safety limit; "
            "reduce fs or process a shorter time window."
        )
    grid = t[0] + np.arange(n_grid, dtype=float) * step
    grid = grid[grid <= t[-1] + endpoint_tol]
    # Snap grid points that differ from a measured timestamp only by floating-
    # point roundoff. Besides preserving measured values, this keeps endpoint
    # identity consistent for downstream masks using strict time comparisons.
    right_idx = np.clip(np.searchsorted(t, grid, side="left"), 0, len(t) - 1)
    left_idx = np.clip(right_idx - 1, 0, len(t) - 1)
    right_dist = np.abs(grid - t[right_idx])
    left_dist = np.abs(grid - t[left_idx])
    use_left = left_dist < right_dist
    nearest_idx = np.where(use_left, left_idx, right_idx)
    nearest_time = t[nearest_idx]
    tol = 1.0 * np.maximum(np.abs(np.spacing(grid)), np.abs(np.spacing(nearest_time)))
    snap = np.abs(grid - nearest_time) <= tol
    grid[snap] = nearest_time[snap]
    out = np.interp(grid, t, y)
    # Mask only interpolated points strictly inside long source intervals.
    # Floating-point grids can represent a source timestamp (e.g. 0.6 s) as
    # 0.6000000000000001 s. Treat values within a few local ULPs of either
    # endpoint as measured samples, not interpolated gap interiors.
    right = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1)
    left_time = t[right - 1]
    right_time = t[right]
    span = right_time - left_time
    left_tol = 1.0 * np.maximum(np.abs(np.spacing(grid)), np.abs(np.spacing(left_time)))
    right_tol = 1.0 * np.maximum(np.abs(np.spacing(grid)), np.abs(np.spacing(right_time)))
    at_left_sample = np.abs(grid - left_time) <= left_tol
    at_right_sample = np.abs(grid - right_time) <= right_tol
    strictly_inside_interval = ~(at_left_sample | at_right_sample)
    inside_gap = (span > limit) & strictly_inside_interval
    out[inside_gap] = np.nan
    return {"time_s": grid, "values": out, "fs_Hz": float(fs), "n_nan_in_long_gaps": int(inside_gap.sum()),
            "n_input": int(len(t)), "n_input_raw": int(n_input_raw),
            "n_nonfinite_dropped": n_nonfinite_dropped, "n_output": int(len(grid)),
            "method": f"linear interpolation; gaps > {limit:.6g} s left as NaN"}


def longest_contiguous_segment(time_s: Any, values: Any, gap_factor: float = 1.5) -> dict[str, Any]:
    """Return the longest run with no gap larger than gap_factor * median dt (cut, never interpolate)."""
    if isinstance(gap_factor, (bool, np.bool_)) or not isinstance(gap_factor, (int, float, np.integer, np.floating)) or not np.isfinite(gap_factor) or gap_factor <= 1:
        raise ValueError("gap_factor must be a finite real number greater than 1.")
    t = np.asarray(time_s, dtype=float).ravel(); y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y):
        raise ValueError("Time and signal must have equal lengths.")
    good = np.isfinite(t) & np.isfinite(y); t, y = t[good], y[good]
    if len(t) < 8:
        raise ValueError("At least 8 finite samples are required.")
    order = np.argsort(t, kind="stable"); t, y = t[order], y[order]
    dt = np.diff(t)
    if np.any(dt <= 0):
        raise ValueError("Duplicate timestamps must be resolved first.")
    cuts = np.flatnonzero(dt > gap_factor * np.median(dt)) + 1
    bounds = np.r_[0, cuts, len(t)]
    k = int(np.argmax(np.diff(bounds)))
    lo, hi = int(bounds[k]), int(bounds[k + 1])
    return {"time_s": t[lo:hi], "values": y[lo:hi], "start_index": lo, "end_index": hi,
            "n_segments": int(len(bounds) - 1), "n_samples_kept": int(hi - lo), "n_samples_total": int(len(t))}


def prepare_time_series_robust(time: Any, values: Any, sample_rate: float | None = None,
                               gap_policy: str = "error",
                               max_interp_gap_s: float | None = None) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    """prepare_time_series with an explicit gap policy.

    gap_policy: "error" (strict, same as core), "longest_segment" (cut, no invention),
    or "interpolate" (uniform resampling; long gaps -> the longest finite run is used).
    Returns (t, y, fs, report); report documents every modification.
    """
    if gap_policy not in {"error", "longest_segment", "interpolate"}:
        raise ValueError("gap_policy must be 'error', 'longest_segment' or 'interpolate'.")
    if time is None or gap_policy == "error":
        t, y, fs = prepare_time_series(time, values, sample_rate)
        return t, y, fs, {"gap_policy": "error", "modified": False}
    tt = np.asarray(time, dtype=float).ravel(); yy = np.asarray(values, dtype=float).ravel()
    try:
        t, y, fs = prepare_time_series(tt, yy, sample_rate)
        return t, y, fs, {"gap_policy": gap_policy, "modified": False, "note": "No gaps; data used as-is."}
    except ValueError as exc:
        if "gap" not in str(exc).lower():
            raise
    audit = sampling_audit(tt[np.isfinite(tt) & np.isfinite(yy)])
    if gap_policy == "longest_segment":
        seg = longest_contiguous_segment(tt, yy)
        t, y, fs = prepare_time_series(seg["time_s"], seg["values"], None)
        return t, y, fs, {"gap_policy": gap_policy, "modified": True, "audit": audit,
                          "samples_kept": seg["n_samples_kept"], "samples_total": seg["n_samples_total"],
                          "note": "Longest gap-free segment retained; no data were interpolated."}
    rs = resample_uniform(tt, yy, None, max_interp_gap_s)
    finite = np.isfinite(rs["values"])
    resampling_audit = {
        "input_samples_raw": rs["n_input_raw"],
        "finite_samples_used": rs["n_input"],
        "nonfinite_samples_dropped": rs["n_nonfinite_dropped"],
        "output_samples": rs["n_output"],
        "nan_samples_in_long_gaps": rs["n_nan_in_long_gaps"],
        "target_sample_rate_Hz": rs["fs_Hz"],
    }
    if finite.all():
        t, y, fs = prepare_time_series(rs["time_s"], rs["values"], None)
        return t, y, fs, {"gap_policy": gap_policy, "modified": True, "audit": audit,
                          "resampling": resampling_audit,
                          "note": "Short gaps linearly interpolated; spectra near Nyquist are attenuated."}
    seg = longest_contiguous_segment(rs["time_s"][finite], rs["values"][finite])
    t, y, fs = prepare_time_series(seg["time_s"], seg["values"], None)
    return t, y, fs, {"gap_policy": gap_policy, "modified": True, "audit": audit,
                      "resampling": resampling_audit,
                      "samples_kept": seg["n_samples_kept"],
                      "note": "Long gaps present: interpolated short gaps, then kept the longest finite run."}

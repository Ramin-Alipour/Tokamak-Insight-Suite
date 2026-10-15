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
    t = t[np.isfinite(t)]
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
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")
    if n_sigmas <= 0 or not np.isfinite(n_sigmas):
        raise ValueError("n_sigmas must be positive and finite.")
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


def detrend_polynomial(time_s: Any, values: Any, order: int = 1,
                       fit_window_s: tuple[float, float] | None = None) -> dict[str, Any]:
    """Subtract a polynomial trend fitted (optionally) only inside a quiet window."""
    t = np.asarray(time_s, dtype=float).ravel(); y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y) or len(t) < order + 2:
        raise ValueError("Time/signal length mismatch or too few samples for the polynomial order.")
    if isinstance(order, bool) or int(order) != order or not 0 <= order <= 8:
        raise ValueError("order must be an integer in [0, 8].")
    m = np.ones(len(t), bool)
    if fit_window_s is not None:
        m = (t >= min(fit_window_s)) & (t <= max(fit_window_s))
        if m.sum() < order + 2:
            raise ValueError("Fit window contains too few samples for the polynomial order.")
    tc = t - t[m].mean()
    coeff = np.polyfit(tc[m], y[m], int(order))
    trend = np.polyval(coeff, tc)
    return {"values": y - trend, "trend": trend, "coefficients_high_to_low_centered_time": coeff.tolist(),
            "order": int(order), "fit_window_s": None if fit_window_s is None else [float(v) for v in fit_window_s]}


def resample_uniform(time_s: Any, values: Any, fs: float | None = None,
                     max_interp_gap_s: float | None = None) -> dict[str, Any]:
    """Linearly interpolate onto a uniform grid, without inventing data inside long gaps.

    Samples falling in a gap longer than `max_interp_gap_s` (default: 3 median intervals)
    are returned as NaN and flagged; nothing is silently filled. Linear interpolation
    low-passes content near Nyquist, so prefer `longest_contiguous_segment` for spectral work.
    """
    t = np.asarray(time_s, dtype=float).ravel(); y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y):
        raise ValueError("Time and signal must have equal lengths.")
    good = np.isfinite(t) & np.isfinite(y)
    t, y = t[good], y[good]
    if len(t) < 8:
        raise ValueError("At least 8 finite samples are required.")
    order = np.argsort(t, kind="stable"); t, y = t[order], y[order]
    if np.any(np.diff(t) <= 0):
        raise ValueError("Duplicate timestamps must be resolved before resampling.")
    med = float(np.median(np.diff(t)))
    if fs is None:
        fs = 1.0 / med
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be positive and finite.")
    limit = 3.0 * med if max_interp_gap_s is None else float(max_interp_gap_s)
    if limit <= 0 or not np.isfinite(limit):
        raise ValueError("max_interp_gap_s must be positive and finite.")
    grid = np.arange(t[0], t[-1] + 0.5 / fs, 1.0 / fs)
    grid = grid[grid <= t[-1] + 1e-12 * abs(t[-1])]
    out = np.interp(grid, t, y)
    # Flag grid points whose bracketing source samples are further apart than the limit.
    right = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1)
    span = t[right] - t[right - 1]
    inside_gap = span > limit
    out[inside_gap] = np.nan
    return {"time_s": grid, "values": out, "fs_Hz": float(fs), "n_nan_in_long_gaps": int(inside_gap.sum()),
            "n_input": int(len(t)), "n_output": int(len(grid)),
            "method": f"linear interpolation; gaps > {limit:.6g} s left as NaN"}


def longest_contiguous_segment(time_s: Any, values: Any, gap_factor: float = 1.5) -> dict[str, Any]:
    """Return the longest run with no gap larger than gap_factor * median dt (cut, never interpolate)."""
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
    if finite.all():
        t, y, fs = prepare_time_series(rs["time_s"], rs["values"], None)
        return t, y, fs, {"gap_policy": gap_policy, "modified": True, "audit": audit,
                          "note": "Short gaps linearly interpolated; spectra near Nyquist are attenuated."}
    seg = longest_contiguous_segment(rs["time_s"][finite], rs["values"][finite])
    t, y, fs = prepare_time_series(seg["time_s"], seg["values"], None)
    return t, y, fs, {"gap_policy": gap_policy, "modified": True, "audit": audit,
                      "samples_kept": seg["n_samples_kept"],
                      "note": "Long gaps present: interpolated short gaps, then kept the longest finite run."}

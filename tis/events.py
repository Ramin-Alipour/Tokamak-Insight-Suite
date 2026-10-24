"""Transient-event analysis: ELMs, sawteeth and plasma-current quench (disruption).

Thresholds are *statistical* (robust MAD) rather than device constants, and every
result reports what was detected and how, so the analyst can overrule it.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, signal


def _clean(time_s: Any, values: Any, min_n: int = 16) -> tuple[np.ndarray, np.ndarray, float]:
    t = np.asarray(time_s, dtype=float).ravel(); y = np.asarray(values, dtype=float).ravel()
    if len(t) != len(y):
        raise ValueError("Time and signal must have equal lengths.")
    g = np.isfinite(t) & np.isfinite(y); t, y = t[g], y[g]
    if len(t) < min_n:
        raise ValueError(f"At least {min_n} finite samples are required.")
    if np.any(np.diff(t) <= 0):
        raise ValueError("Time must be strictly increasing.")
    intervals = np.diff(t)
    dt = float(np.median(intervals))
    # Timestamp quantization can alternate adjacent clock ticks (e.g. 2 and
    # 3 microseconds for a 400 kHz signal stored at 1 us resolution). Such
    # deterministic quantization is not the same as dropped-sample gaps.
    # Retain the strict gap check, but tolerate interval variation when every
    # interval is within 1.5x of the median and the intervals are consistent
    # with integer multiples of a common clock tick.
    max_ratio = float(np.max(intervals) / dt)
    cv = float(np.std(intervals) / dt)
    # Estimate the timestamp clock quantum from the smallest positive
    # separation between distinct intervals. Using min(intervals) as the tick
    # fails for a 400 kHz signal timestamped at 1 us resolution: intervals
    # alternate between 2 and 3 us, while the actual clock quantum is 1 us.
    # Cluster interval values that differ only by floating-point roundoff
    # before estimating the clock tick. np.unique alone treats 2.0000000000000
    # and 2.0000000000001 microseconds as distinct values, producing a false
    # sub-femtosecond tick and rejecting otherwise valid quantized timestamps.
    interval_tol = max(
        1e-9 * dt,
        2.0 * np.finfo(float).eps * max(float(np.max(np.abs(t))), float(np.max(intervals)), 1e-12),
    )
    sorted_intervals = np.sort(intervals)
    clusters: list[list[float]] = []
    for value in sorted_intervals:
        if not clusters or value - clusters[-1][-1] > interval_tol:
            clusters.append([float(value)])
        else:
            clusters[-1].append(float(value))
    interval_levels = np.asarray([float(np.median(cluster)) for cluster in clusters])
    differences = np.diff(interval_levels)
    positive_differences = differences[differences > interval_tol]
    tick = float(np.min(positive_differences)) if positive_differences.size else float(np.min(intervals))
    tick_ratios = intervals / tick
    quantized_ticks = np.all(np.abs(tick_ratios - np.rint(tick_ratios)) <= 0.05)
    quantized_adjacent = (
        quantized_ticks
        and max_ratio <= 1.5
        and float(np.max(tick_ratios) - np.min(tick_ratios)) <= 1.0 + 1e-9
    )
    if max_ratio > 1.5 or (cv > 0.02 and not quantized_adjacent):
        raise ValueError("Event detection requires approximately uniform sampling; resample irregular timestamps and handle missing samples explicitly first.")
    return t, y, dt


def _mad_sigma(x: np.ndarray) -> float:
    return float(1.4826 * np.median(np.abs(x - np.median(x))))


def _real_parameter(name: str, value: Any, *, positive: bool = False,
                    minimum: float | None = None, maximum: float | None = None) -> float:
    """Validate a real scalar without accepting booleans or coercing strings."""
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))):
        raise ValueError(f"{name} must be a finite real number.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive and finite.")
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be <= {maximum}.")
    return result


def interval_statistics(event_times_s: Any) -> dict[str, Any]:
    """Repetition statistics: mean/median period, frequency, coefficient of variation."""
    ev = np.asarray(event_times_s, dtype=float).ravel()
    if not np.all(np.isfinite(ev)):
        raise ValueError("Event times must contain only finite values.")
    if np.any(np.diff(ev) <= 0):
        raise ValueError("Event times must be strictly increasing and unique.")
    if len(ev) < 2:
        return {"n_events": int(len(ev)), "note": "At least two events are needed for interval statistics."}
    d = np.diff(ev)
    return {"n_events": int(len(ev)), "mean_interval_s": float(d.mean()), "median_interval_s": float(np.median(d)),
            "std_interval_s": float(d.std(ddof=1)) if len(d) > 1 else 0.0,
            "coefficient_of_variation": float(d.std(ddof=1) / d.mean()) if len(d) > 1 and d.mean() > 0 else None,
            "mean_frequency_Hz": float(1.0 / d.mean()), "min_interval_s": float(d.min()), "max_interval_s": float(d.max())}


def detect_elms(time_s: Any, d_alpha: Any, baseline_window_s: float | None = None,
                n_sigmas: float = 5.0, min_separation_s: float | None = None,
                window_s: tuple[float, float] | None = None) -> dict[str, Any]:
    """Detect ELM-like bursts in a D_alpha / divertor-emission trace.

    A slowly varying baseline (rolling median over `baseline_window_s`, default 50 median
    event-free scales ~ 5% of the record) is subtracted; peaks exceeding
    n_sigmas * MAD-sigma of the residual are events. Amplitude = peak above baseline.
    Peak-height detection cannot distinguish ELM types or reject pellet/ disruption spikes:
    restrict `window_s` to the H-mode flat-top and verify visually.
    """
    t, y, dt = _clean(time_s, d_alpha, 32)
    if window_s is not None:
        try:
            bounds = np.asarray(window_s, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("window_s must contain two finite time bounds.") from exc
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("window_s must contain two finite time bounds.")
        window_s = (float(bounds[0]), float(bounds[1]))
        m = (t >= min(window_s)) & (t <= max(window_s))
        t, y = t[m], y[m]
        if len(t) < 32:
            raise ValueError("Analysis window contains too few samples.")
    n_sigmas = _real_parameter("n_sigmas", n_sigmas, positive=True)
    if baseline_window_s is not None:
        baseline_window_s = _real_parameter("baseline_window_s", baseline_window_s, positive=True)
    if min_separation_s is not None:
        min_separation_s = _real_parameter("min_separation_s", min_separation_s, positive=True)
    span = t[-1] - t[0]
    bw = baseline_window_s if baseline_window_s is not None else max(span * 0.05, 20 * dt)
    size = max(3, int(round(bw / dt)) | 1)
    size = min(size, (len(y) // 2) * 2 - 1) if len(y) > 4 else 3
    base = ndimage.median_filter(y, size=size, mode="nearest")
    res = y - base
    sigma = _mad_sigma(res)
    if sigma <= 0:
        sigma = float(np.std(res)) or 1e-300
    sep = min_separation_s if min_separation_s is not None else 10 * dt
    distance = max(1, int(round(sep / dt)))
    # Require prominence as well as height: noise ripples riding on the decaying tail of a
    # large burst exceed the height threshold but are not independent events.
    idx, props = signal.find_peaks(res, height=n_sigmas * sigma, prominence=n_sigmas * sigma, distance=distance)
    times = t[idx]
    out = {"event_times_s": times, "amplitudes_above_baseline": res[idx], "n_elms": int(len(idx)),
           "noise_sigma_robust": sigma, "threshold": float(n_sigmas * sigma),
           "baseline_window_s": float(size * dt), "analysis_span_s": float(span),
           "statistics": interval_statistics(times),
           "method": "rolling-median baseline; MAD-threshold peak detection",
           "warning": "Statistical detector: confirm against the raw trace; ELM type, pellets and disruptions are not classified."}
    if len(idx) >= 2:
        out["instantaneous_frequency_Hz"] = 1.0 / np.diff(times)
        out["instantaneous_frequency_time_s"] = 0.5 * (times[1:] + times[:-1])
    return out


def detect_sawteeth(time_s: Any, core_signal: Any, smooth_window_s: float | None = None,
                    n_sigmas: float = 5.0, min_period_s: float | None = None,
                    min_relative_amplitude: float = 0.3) -> dict[str, Any]:
    """Detect sawtooth crashes in a core signal (SXR, core T_e from ECE, ...).

    Crashes are rapid *decreases*: peaks of -d(smoothed y)/dt above n_sigmas * MAD-sigma.
    Crash amplitude = max(y) in the preceding half-period minus min(y) within the crash.
    Signals where sawteeth *increase* the measurement (outside the inversion radius) must be negated first.

    A purely statistical threshold on noisy derivatives yields occasional false peaks (a 4-sigma
    exceedance is *expected* a few times per 10^4 Gaussian samples), so candidates whose drop is smaller
    than `min_relative_amplitude` x the median candidate drop are rejected and counted in `n_rejected_small`.
    """
    n_sigmas = _real_parameter("n_sigmas", n_sigmas, positive=True)
    if smooth_window_s is not None:
        smooth_window_s = _real_parameter("smooth_window_s", smooth_window_s, positive=True)
    if min_period_s is not None:
        min_period_s = _real_parameter("min_period_s", min_period_s, positive=True)
    min_relative_amplitude = _real_parameter(
        "min_relative_amplitude", min_relative_amplitude, minimum=0.0, maximum=1.0
    )
    t, y, dt = _clean(time_s, core_signal, 64)
    sw = smooth_window_s if smooth_window_s is not None else 5 * dt
    k = max(1, int(round(sw / dt)))
    ys = ndimage.uniform_filter1d(y, size=k, mode="nearest") if k > 1 else y
    d = -np.gradient(ys, t)
    sigma = _mad_sigma(d) or float(np.std(d)) or 1e-300
    dist = max(1, int(round((min_period_s if min_period_s is not None else 20 * dt) / dt)))
    idx, _ = signal.find_peaks(d, height=n_sigmas * sigma, distance=dist)
    times = t[idx]
    stats = interval_statistics(times)
    half = max(2, int(round(0.5 * stats["median_interval_s"] / dt))) if "median_interval_s" in stats else max(2, 10 * k)
    amps = []; rates = []
    for i in idx:
        lo = max(0, i - half); hi = min(len(y), i + k + 2)
        amps.append(float(np.max(ys[lo:i + 1]) - np.min(ys[i:hi])))
        rates.append(float(d[i]))
    amps = np.asarray(amps); rates = np.asarray(rates)
    n_rejected = 0
    if len(amps) >= 3 and min_relative_amplitude > 0:
        keep = amps >= min_relative_amplitude * float(np.median(amps))
        n_rejected = int((~keep).sum())
        times, amps, rates, idx = times[keep], amps[keep], rates[keep], idx[keep]
        stats = interval_statistics(times)
    return {"crash_times_s": times, "crash_amplitudes": amps, "max_drop_rates_per_s": rates,
            "n_crashes": int(len(idx)), "n_rejected_small": n_rejected, "period_statistics": stats, "noise_sigma_robust": sigma,
            "method": "MAD-thresholded peaks of negative derivative of boxcar-smoothed signal",
            "warning": "Choose a core channel inside the inversion radius; ELMs/disruptions also trigger this detector."}


def current_quench_analysis(time_s: Any, ip_A: Any, reference_current_A: float | None = None,
                            reference_window_s: tuple[float, float] | None = None,
                            upper_fraction: float = 0.8, lower_fraction: float = 0.2) -> dict[str, Any]:
    """Characterize a plasma-current quench (disruption) from I_p(t).

    Reference current I0: `reference_current_A`, else the median of |Ip| in `reference_window_s`,
    else the 95th percentile of |Ip| (flat-top proxy). The quench is found after the *last*
    time |Ip| >= I0 * upper_fraction before the final collapse. Conventional duration
    tau_CQ = (t_20 - t_80)/(0.6) (ITER convention, 80%->20%). Linear-extrapolated 100%->0 time
    is the same number by construction. Runaway plateaus will keep |Ip| above the lower level and
    are reported as `disrupted=False, runaway_or_incomplete_quench=True`.
    """
    lower_fraction = _real_parameter("lower_fraction", lower_fraction, minimum=0.0, maximum=1.0)
    upper_fraction = _real_parameter("upper_fraction", upper_fraction, minimum=0.0, maximum=1.0)
    if not 0 < lower_fraction < upper_fraction < 1:
        raise ValueError("Need 0 < lower_fraction < upper_fraction < 1.")
    t, y, dt = _clean(time_s, ip_A, 16)
    a = np.abs(y)
    if reference_current_A is not None:
        i0 = abs(_real_parameter("reference_current_A", reference_current_A))
    elif reference_window_s is not None:
        try:
            bounds = np.asarray(reference_window_s, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("reference_window_s must contain two finite time bounds.") from exc
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)):
            raise ValueError("reference_window_s must contain two finite time bounds.")
        m = (t >= float(np.min(bounds))) & (t <= float(np.max(bounds)))
        if m.sum() < 2:
            raise ValueError("Reference window contains too few samples.")
        i0 = float(np.median(a[m]))
    else:
        i0 = float(np.percentile(a, 95))
    if i0 <= 0:
        raise ValueError("Reference current must be positive.")
    hi, lo = upper_fraction * i0, lower_fraction * i0
    above = np.flatnonzero(a >= hi)
    if len(above) == 0:
        raise ValueError("Current never reaches the upper threshold; check the reference current.")
    last_hi = int(above[-1])
    below_lo = np.flatnonzero(a[last_hi:] <= lo)
    result: dict[str, Any] = {"reference_current_A": i0, "upper_level_A": hi, "lower_level_A": lo,
                              "final_current_fraction": float(a[-1] / i0)}

    def cross(level: float, start: int, stop: int) -> float:
        seg_a = a[start:stop + 1]; seg_t = t[start:stop + 1]
        k = int(np.argmax(seg_a <= level))
        if k == 0:
            return float(seg_t[0])
        return float(seg_t[k - 1] + (level - seg_a[k - 1]) * (seg_t[k] - seg_t[k - 1]) / (seg_a[k] - seg_a[k - 1]))

    if len(below_lo) == 0:
        result.update({"disrupted": False, "runaway_or_incomplete_quench": bool(a[-1] < hi),
                       "t_upper_s": cross(hi, last_hi, len(a) - 1) if a[-1] < hi else None,
                       "note": "|Ip| never falls to the lower level: no complete quench (flat-top, ramp-down in progress or runaway plateau)."})
        return result
    end = last_hi + int(below_lo[0])
    t_hi = cross(hi, last_hi, end) if last_hi < end else float(t[last_hi])
    t_lo = cross(lo, last_hi, end)
    dur = (t_lo - t_hi) / (upper_fraction - lower_fraction)
    didt = np.gradient(y, t)
    seg = slice(max(0, last_hi - 1), end + 1)
    k = int(np.argmax(np.abs(didt[seg]))) + seg.start
    result.update({"disrupted": True, "t_upper_s": t_hi, "t_lower_s": t_lo,
                   "current_quench_time_s": float(dur),
                   "convention": f"tau_CQ = (t_{int(lower_fraction*100)} - t_{int(upper_fraction*100)})/{upper_fraction-lower_fraction:.2g}",
                   "max_decay_rate_A_per_s": float(abs(didt[k])), "time_of_max_decay_s": float(t[k]),
                   "mean_decay_rate_A_per_s": float(i0 * (upper_fraction - lower_fraction) / (t_lo - t_hi)) if t_lo > t_hi else None,
                   "warning": "Sampling must resolve the quench (dt << tau_CQ); Ip from Rogowski integrators saturates/lags in fast quenches."})
    return result

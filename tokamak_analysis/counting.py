"""Shared, explicit semantics for binned detector counts and cumulative counters."""
from __future__ import annotations
import numpy as np


def count_rate_from_timestamps(time_s, counts, *, timestamp_semantics,
                              counts_are_cumulative=False, live_time_s=None):
    """Return normalized times, rate, total counts and a traceable method description.

    Supported timestamp_semantics:
    - bin_centers: one timestamp per independent bin count; width inferred from centers
      or supplied explicitly as live_time_s.
    - bin_edges: one more timestamp than bin counts; exact widths are used.
    - counter_read_times: one timestamp per cumulative counter reading.
    """
    allowed = {"bin_centers", "bin_edges", "counter_read_times"}
    if timestamp_semantics not in allowed:
        raise ValueError("Set timestamp_semantics explicitly to bin_centers, bin_edges, or counter_read_times")
    if counts_are_cumulative and timestamp_semantics != "counter_read_times":
        raise ValueError("Cumulative counters require timestamp_semantics='counter_read_times'")
    if not counts_are_cumulative and timestamp_semantics == "counter_read_times":
        raise ValueError("counter_read_times is valid only when counts_are_cumulative=True")
    t = np.asarray(time_s, dtype=float)
    c = np.asarray(counts, dtype=float)
    if c.ndim != 1 or len(c) < 2 or not np.all(np.isfinite(c)) or np.any(c < 0):
        raise ValueError("Counts must be a finite nonnegative 1-D array with at least two samples")
    if t.ndim != 1 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Time coordinates must be finite and strictly increasing")
    if timestamp_semantics == "bin_edges":
        if counts_are_cumulative or len(t) != len(c) + 1:
            raise ValueError("bin_edges requires per-bin counts and exactly len(counts)+1 time edges")
        if live_time_s is not None:
            raise ValueError("Do not supply live_time_s when exact bin_edges are provided")
        width = np.diff(t)
        return {"time_s": (t[:-1] + t[1:]) / 2, "count_rate_s-1": c / width,
                "total_counts": float(np.sum(c)), "timestamp_semantics": timestamp_semantics,
                "counts_are_cumulative": False, "count_rate_method": "per-bin counts divided by exact edge-to-edge bin widths"}
    if len(t) != len(c):
        raise ValueError(f"{timestamp_semantics} requires one timestamp per count sample")
    if counts_are_cumulative:
        if live_time_s is not None or np.any(np.diff(c) < 0):
            raise ValueError("Cumulative counts must be nondecreasing and cannot use live_time_s")
        rate = np.gradient(c, t, edge_order=2 if len(t) > 2 else 1)
        return {"time_s": t, "count_rate_s-1": rate, "total_counts": float(c[-1] - c[0]),
                "timestamp_semantics": timestamp_semantics, "counts_are_cumulative": True,
                "count_rate_method": "derivative of cumulative counter at counter read times"}
    if live_time_s is None:
        width = np.gradient(t)
    else:
        width = np.asarray(live_time_s, dtype=float)
        if not np.all(np.isfinite(width)) or np.any(width <= 0):
            raise ValueError("live_time_s must be finite and positive")
        try:
            width = np.broadcast_to(width, c.shape)
        except ValueError as exc:
            raise ValueError("live_time_s must be scalar or match counts") from exc
    return {"time_s": t, "count_rate_s-1": c / width, "total_counts": float(np.sum(c)),
            "timestamp_semantics": timestamp_semantics, "counts_are_cumulative": False,
            "count_rate_method": "counts divided by supplied live time" if live_time_s is not None else
                                 "counts divided by local bin widths estimated from bin-center timestamps"}

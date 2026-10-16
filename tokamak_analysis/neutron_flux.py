"""Neutron detector count-rate and flux estimates with explicit time semantics."""
import numpy as np


def analyze_neutron_flux(time_s, counts, *, timestamp_semantics=None, efficiency=None,
                         effective_area_m2=None, live_time_s=None,
                         counts_are_cumulative=False):
    """Estimate count rate and optionally flux from detector counts.

    timestamp_semantics must be one of:
      - "bin_centers": time_s has one timestamp per per-bin count; widths are inferred
        from center spacing unless live_time_s is supplied.
      - "bin_edges": time_s has len(counts)+1 bin boundaries; exact bin widths are used.
      - "counter_read_times": time_s gives the read time of each cumulative counter value.
    """
    if timestamp_semantics not in {"bin_centers", "bin_edges", "counter_read_times"}:
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
        bin_width = np.diff(t)
        output_time = (t[:-1] + t[1:]) / 2.0
        rate = c / bin_width
        total_counts = float(np.sum(c))
        rate_method = "per-bin counts divided by exact edge-to-edge bin widths"
    elif counts_are_cumulative:
        if len(t) != len(c):
            raise ValueError("counter_read_times requires one read time per cumulative counter value")
        if np.any(np.diff(c) < 0):
            raise ValueError("Cumulative counts must be nondecreasing")
        if live_time_s is not None:
            raise ValueError("live_time_s applies to per-bin counts, not cumulative counters")
        rate = np.gradient(c, t, edge_order=2 if len(t) > 2 else 1)
        total_counts = float(c[-1] - c[0])
        output_time = t
        rate_method = "derivative of cumulative counter at counter read times"
    else:
        if len(t) != len(c):
            raise ValueError("bin_centers requires one timestamp per per-bin count")
        if live_time_s is None:
            bin_width = np.gradient(t)
        else:
            bin_width = np.asarray(live_time_s, dtype=float)
            if not np.all(np.isfinite(bin_width)) or np.any(bin_width <= 0):
                raise ValueError("live_time_s must be finite and positive")
            try:
                bin_width = np.broadcast_to(bin_width, c.shape)
            except ValueError as exc:
                raise ValueError("live_time_s must be scalar or match counts") from exc
        rate = c / bin_width
        total_counts = float(np.sum(c))
        output_time = t
        rate_method = ("counts divided by supplied live time" if live_time_s is not None else
                       "counts divided by local bin widths estimated from bin-center timestamps")

    out = {"time_s": output_time, "counts": c, "count_rate_s-1": rate,
           "total_counts": total_counts, "count_rate_method": rate_method,
           "timestamp_semantics": timestamp_semantics,
           "counts_are_cumulative": bool(counts_are_cumulative)}
    if efficiency is not None or effective_area_m2 is not None:
        if efficiency is None or effective_area_m2 is None:
            raise ValueError("Provide detector efficiency and effective area")
        eff, area = float(efficiency), float(effective_area_m2)
        if not np.isfinite(eff) or not np.isfinite(area) or not 0 < eff <= 1 or area <= 0:
            raise ValueError("Efficiency must be finite in (0,1] and area finite positive")
        flux = rate / (eff * area)
        out["neutron_flux_m-2_s-1"] = flux
        out["flux_estimate_m-2_s-1"] = flux  # backward-compatible alias
    out["caveat"] = ("Count semantics and timestamp semantics are explicit. Flux is an estimate and also requires "
                     "detector geometry, efficiency, background, dead-time and energy-response corrections.")
    return out


def neutron_flux_from_integrated_counts(counts, live_time_s, detector_efficiency, effective_area_m2):
    """Integrated-count flux estimator Phi=N/(T*epsilon*A) with Poisson counting uncertainty."""
    n, t, eff, area = np.broadcast_arrays(*[
        np.asarray(v, dtype=float)
        for v in (counts, live_time_s, detector_efficiency, effective_area_m2)])
    if any(not np.all(np.isfinite(v)) for v in (n, t, eff, area)) or np.any(n < 0) or np.any(t <= 0) or np.any(eff <= 0) or np.any(eff > 1) or np.any(area <= 0):
        raise ValueError("Counts non-negative; time/area positive; efficiency in (0,1]; inputs finite")
    denom = t * eff * area
    flux = n / denom
    return {"neutron_flux_m-2_s-1": flux, "flux_estimate_m-2_s-1": flux,
            "poisson_uncertainty_m-2_s-1": np.sqrt(n) / denom,
            "equation": "Phi=N/(T*epsilon*A); sigma_Phi=sqrt(N)/(T*epsilon*A)",
            "assumption": "Poisson counting statistics and constant efficiency/area over live time; background and dead-time corrections must be applied separately."}
